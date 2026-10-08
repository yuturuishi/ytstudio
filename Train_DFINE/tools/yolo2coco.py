# -*- coding: utf-8 -*-
"""把 YOLO 格式检测数据集批量转成 D-FINE 可训练的 COCO json + 数据集配置。

用法: python tools/yolo2coco.py
输出: E:/datasets/datasets_dfine/<name>/
      ├─ annotations/instances_train.json / instances_val.json
      ├─ dataset.yml            (D-FINE dataset 配置, num_classes/remap 已设好)
      └─ report.txt             (转换统计)
图片不复制，dataset.yml 里 img_folder 直接指向原始 images 目录。

注意: D-FINE 在 remap_mscoco_category: False 时要求 category_id 为 0 起始连续，
本转换器按 YOLO class_id 原值写入（0 起）。
"""
import os, sys, json, time
import cv2
import numpy as np
import yaml

SRC_ROOTS = [r"E:\datasets", r"E:\datasets\datasets_yolo26"]
OUT_ROOT = r"E:\datasets\datasets_dfine"

# 无 data.yaml 的数据集手工声明; names=None 则自动用 class_N 占位
MANUAL = {
    "detect_student6type": dict(names=None, nc=6, only_train_split=True),
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def load_names(root, manual=None):
    for f in ("data.yaml", "data_local.yaml"):
        p = os.path.join(root, f)
        if os.path.isfile(p):
            try:
                y = yaml.safe_load(open(p, encoding="utf-8"))
                names = y.get("names")
                if isinstance(names, dict):
                    names = [names[k] for k in sorted(names)]
                if names:
                    return list(names)
            except Exception:
                pass
    if manual is not None:
        return None  # 占位
    return None

def find_dataset_dirs():
    out = []
    for src in SRC_ROOTS:
        if not os.path.isdir(src):
            continue
        for d in sorted(os.listdir(src)):
            full = os.path.join(src, d)
            if not os.path.isdir(full):
                continue
            if d.startswith("detect") and "-检测" not in d and "classify" not in d \
               and os.path.isdir(os.path.join(full, "train", "images")):
                out.append(full)
    return out

def convert_split(img_dir, lab_dir, names, cat_ids=None):
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
    images, anns, negs, miss, bad = [], [], 0, 0, []
    aid = 1
    for iid, f in enumerate(files, 1):
        stem = os.path.splitext(f)[0]
        lp = None
        for e in (".txt",):
            cand = os.path.join(lab_dir, stem + e)
            if os.path.isfile(cand):
                lp = cand
                break
        path = os.path.join(img_dir, f)
        img = cv2.imread(path)
        if img is None:
            bad.append(f)
            continue
        h, w = img.shape[:2]
        images.append(dict(id=iid, file_name=f, width=w, height=h))
        if lp is None:
            miss += 1
            continue
        rows = []
        try:
            for line in open(lp, encoding="utf-8"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
                if cat_ids is not None and c not in cat_ids:
                    continue
                bw = max(0.0, min(bw, 1.0)); bh = max(0.0, min(bh, 1.0))
                cx = max(0.0, min(cx, 1.0)); cy = max(0.0, min(cy, 1.0))
                x1 = max(0.0, (cx - bw / 2) * w); y1 = max(0.0, (cy - bh / 2) * h)
                aw = min(w - x1, bw * w); ah = min(h - y1, bh * h)
                if aw <= 0 or ah <= 0:
                    continue
                rows.append(dict(id=aid, image_id=iid, category_id=c,
                                 bbox=[round(x1, 2), round(y1, 2), round(aw, 2), round(ah, 2)],
                                 area=round(aw * ah, 2), iscrowd=0))
                aid += 1
        except Exception:
            pass
        anns.extend(rows)
        if not rows:
            negs += 1
    return images, anns, negs, miss, bad

def write_coco(images, anns, categories, path):
    coco = dict(info=dict(description="converted from YOLO", date_created=time.strftime("%Y-%m-%d")),
                licenses=[], images=images, annotations=anns, categories=categories)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

DATASET_YML = """\
task: detection

num_classes: {nc}
remap_mscoco_category: False

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', ]

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {img_train}
    ann_file: {ann_train}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {img_val}
    ann_file: {ann_val}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
"""

def main():
    all_lines = []
    dirs = find_dataset_dirs()
    print("发现 %d 个检测数据集" % len(dirs), flush=True)
    seen_names = set()
    for full in dirs:
        name = os.path.basename(full)
        if name in seen_names:
            print("跳过重复 %s（已由更上层同名数据集转换，内容一致）" % full)
            continue
        seen_names.add(name)
        manual = MANUAL.get(name)
        names = load_names(full, manual)
        nc = None
        if names is None:
            if manual and manual.get("nc"):
                nc = manual["nc"]
                names = ["class_%d" % i for i in range(nc)]
                note = " [类别名待补: 无yml, id 0..%d]" % (nc - 1)
            else:
                print("跳过 %s: 无 names" % name)
                continue
        else:
            nc = len(names)
            note = ""
        if manual and manual.get("names") is None and names and names[0].startswith("class_"):
            note = " [类别名待补: 无yml, id 0..%d]" % (nc - 1)

        out_dir = os.path.join(OUT_ROOT, name)
        ann_dir = os.path.join(out_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        categories = [dict(id=i, name=n, supercategory="object") for i, n in enumerate(names)]

        # train
        img_tr = os.path.join(full, "train", "images")
        lab_tr = os.path.join(full, "train", "labels")
        t0 = time.time()
        imgs_tr, anns_tr, neg_tr, miss_tr, bad_tr = convert_split(img_tr, lab_tr, names)
        # val
        if manual and manual.get("only_train_split"):
            # 自动切 1/7 当 val（按文件名排序，每 7 张取 1）
            val_ids = set(imgs_tr[i]["id"] for i in range(0, len(imgs_tr), 7))
            imgs_v = [im for im in imgs_tr if im["id"] in val_ids]
            anns_v = [a for a in anns_tr if a["image_id"] in val_ids]
            imgs_tr2 = [im for im in imgs_tr if im["id"] not in val_ids]
            anns_tr2 = [a for a in anns_tr if a["image_id"] not in val_ids]
            imgs_tr, anns_tr = imgs_tr2, anns_tr2
            split_note = "val=auto-split(1/7)"
        else:
            img_v = os.path.join(full, "valid", "images")
            lab_v = os.path.join(full, "valid", "labels")
            imgs_v, anns_v, neg_v, miss_v, bad_v = convert_split(img_v, lab_v, names)
            split_note = "neg=%d miss_label=%d bad_img=%d" % (neg_v, miss_v, len(bad_v))

        write_coco(imgs_tr, anns_tr, categories, os.path.join(ann_dir, "instances_train.json"))
        write_coco(imgs_v, anns_v, categories, os.path.join(ann_dir, "instances_val.json"))

        yml_txt = DATASET_YML.format(nc=nc,
                                     img_train=img_tr.replace("\\", "/"),
                                     ann_train=os.path.join(ann_dir, "instances_train.json").replace("\\", "/"),
                                     img_val=(img_tr if split_note.startswith("val=auto") else img_v).replace("\\", "/"),
                                     ann_val=os.path.join(ann_dir, "instances_val.json").replace("\\", "/"))
        open(os.path.join(out_dir, "dataset.yml"), "w", encoding="utf-8").write(yml_txt)

        rep = ("train: img=%d ann=%d neg=%d miss_label=%d bad_img=%d | val: img=%d ann=%d (%s)%s | %.0fs"
               % (len(imgs_tr), len(anns_tr), neg_tr, miss_tr, len(bad_tr),
                  len(imgs_v), len(anns_v), split_note, note, time.time() - t0))
        open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8").write(rep + "\nnames: " + str(names))
        line = "%-34s nc=%-2d %s" % (name, nc, rep)
        print(line, flush=True)
        all_lines.append((name, nc, names, len(imgs_tr), len(anns_tr), len(imgs_v), len(anns_v), note))

    # 总 README
    md = ["# datasets_dfine — D-FINE 批量训练数据集（COCO 格式）", "",
          "转换自 YOLO 格式（tools/yolo2coco.py），图片不复制，`dataset.yml` 的 `img_folder` 指向原始 images 目录。",
          "category_id 从 0 起（YOLO class_id 原值），训练配置必须 `remap_mscoco_category: False`。", "",
          "| 数据集 | nc | 类别 | train img/ann | val img/ann | 备注 |",
          "|---|---|---|---|---|---|"]
    for name, nc, names, ti, ta, vi, va, note in all_lines:
        md.append("| %s | %d | %s | %d / %d | %d / %d | %s |" % (name, nc, ", ".join(names), ti, ta, vi, va, note.strip() or "-"))
    md += ["", "## 用法（D-FINE 训练）",
           "复制 `D:\\project\\ytstudio\\Train_DFINE\\configs\\dfine\\dfine_hgnetv2_n_coco.yml` 为场景配置：",
           "1. `num_classes` 改成上表 nc；2. `dataset` 段指向本目录对应 `<name>/dataset.yml`；",
           "3. 训练命令 `python train.py -c configs/dfine/<场景>.yml --use-amp --seed=0`。"]
    open(os.path.join(OUT_ROOT, "README.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\nREADME ->", os.path.join(OUT_ROOT, "README.md"))

if __name__ == "__main__":
    main()
