# -*- coding: utf-8 -*-
"""D-FINE N 场景模型批量训练编排器（100 轮变体 train_all_100）。

参考 Train_yolo26 的 best_models 归档约定：
  best_detect_<ds>_dfine_n_imgsz640_ep<EPOCHS>_done<done>_mAP50-95_<0pXXXXX>.pth/.onnx/.txt

流程（逐数据集）：
  1. 生成缺失的场景配置 configs/dfine/scenes/dfine_hgnetv2_n_<ds>.yml
  2. 训练（幂等：best_models 里已有归档则跳过）
     - 输出目录在 C 盘临时目录（D 盘沙箱禁止向已存在文件追加，TensorBoard/log 会炸）
  3. 解析 log.txt：best_stat（全局最优 mAP50-95 + epoch）+ 各轮 COCO summary（取 AP50）
  4. 取 best_stg2.pth（无则 best_stg1.pth）→ 归档 .pth + 导出 .onnx + 写 .txt 元数据
  5. 更新 best_models/_train_status.json，继续下一个

用法：
  venv\\Scripts\\python.exe tools\\train_all.py                # 跑全部 15 个
  venv\\Scripts\\python.exe tools\\train_all.py detect_fire_smoke   # 只跑指定集
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(PROJ, "venv", "Scripts", "python.exe")
DATA_ROOT = "E:/datasets/datasets_dfine"
BEST_DIR = os.path.join(PROJ, "best_models")
SCENES_DIR = os.path.join(PROJ, "configs", "dfine", "scenes")

OUT_ROOT = os.path.join(tempfile.gettempdir(), "dfine_runs100")
STATUS_FILE = os.path.join(OUT_ROOT, "_train_status.json")  # C 盘：D 盘禁止覆盖写

EPOCHS = 100          # 100 轮变体（150 轮全量版见 train_all.py）
# fp16 AMP 前向溢出反复发散的数据集：改用 fp32 训练（D-FINE 系 fp16 中间值易溢出）
NO_AMP = {"detect_smoke", "detect_sleep", "detect_phone", "detect_climb_noclimb",
          "detect_stand_fall_sit_squat_run", "detect_head_safehat", "detect_dense_person", "detect_kitchen"}
STOP_EPOCH = 93       # 增强停止/EMA 刷新段起点（比例参照官方 custom 220/200）
BATCH = 32
VAL_BATCH = 64
IMGSZ = 640
ARCH = "dfine_n"

# 顺序：A/B 优先级 + 从小到大，最大两个压轴
DATASETS = [
    "detect_fire_smoke", "detect_smoke", "detect_sleep", "detect_fight",
    "detect_dust", "detect_rat", "detect_clothes", "detect_phone",
    "detect_climb_noclimb", "detect_stand_fall_sit_squat_run",
    "detect_student3type", "detect_student6type",
    "detect_head_safehat", "detect_dense_person", "detect_kitchen",
]

CFG_TEMPLATE = """__include__: [
  '{dataset_yml}',
  '../../runtime.yml',
  '../include/dataloader.yml',
  '../include/optimizer.yml',
  '../include/dfine_hgnetv2.yml',
]

output_dir: ./output/n_{tag}

DFINE:
  backbone: HGNetv2

HGNetv2:
  name: 'B0'
  return_idx: [2, 3]
  freeze_at: -1
  freeze_norm: False
  use_lab: True

HybridEncoder:
  in_channels: [512, 1024]
  feat_strides: [16, 32]
  hidden_dim: 128
  use_encoder_idx: [1]
  dim_feedforward: 512
  expansion: 0.34
  depth_mult: 0.5

DFINETransformer:
  feat_channels: [128, 128]
  feat_strides: [16, 32]
  hidden_dim: 128
  dim_feedforward: 512
  num_levels: 2
  num_layers: 3
  eval_idx: -1
  num_points: [6, 6]

optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.0004
    -
      params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: 0.0004
      weight_decay: 0.
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.
  lr: 0.0008
  betas: [0.9, 0.999]
  weight_decay: 0.0001

epochs: {epochs}
train_dataloader:
  total_batch_size: {batch}
  dataset:
    transforms:
      policy:
        epoch: {stop_epoch}
  collate_fn:
    stop_epoch: {stop_epoch}
    ema_restart_decay: 0.9999
    base_size_repeat: ~

val_dataloader:
  total_batch_size: {val_batch}
"""


def log(msg):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)


def load_status():
    if os.path.isfile(STATUS_FILE):
        with open(STATUS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"done": {}, "failed": {}, "running": None}


def save_status(st):
    os.makedirs(BEST_DIR, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def gen_config(ds):
    os.makedirs(SCENES_DIR, exist_ok=True)
    tag = ds.replace("detect_", "")
    path = os.path.join(SCENES_DIR, "dfine_hgnetv2_n100_%s.yml" % tag)
    if os.path.isfile(path):
        return path
    dataset_yml = os.path.join(DATA_ROOT, ds, "dataset.yml").replace("\\", "/")
    if not os.path.isfile(dataset_yml):
        raise FileNotFoundError(dataset_yml)
    with open(path, "w", encoding="utf-8") as f:
        f.write(CFG_TEMPLATE.format(
            dataset_yml=dataset_yml, tag=tag, epochs=EPOCHS,
            stop_epoch=STOP_EPOCH, batch=BATCH, val_batch=VAL_BATCH))
    log("生成配置 %s" % path)
    return path


def parse_best(log_txt):
    """解析 JSON-lines 格式的 log.txt，取 test_coco_eval_bbox[0](mAP50-95) 全局最优轮。
    返回 (mAP50-95, mAP50, best_epoch)。"""
    best = None
    if not os.path.isfile(log_txt):
        return None, None, None
    with open(log_txt, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            arr = d.get("test_coco_eval_bbox")
            if not arr or len(arr) < 2:
                continue
            a95, a50 = float(arr[0]), float(arr[1])
            if best is None or a95 > best[0]:
                best = (a95, a50, d.get("epoch"))
    if best is None:
        return None, None, None
    return best[0], best[1], best[2]


def load_sd_for_export(ckpt_path):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and ck.get("ema"):
        ema = ck["ema"]
        state = ema.get("module", ema) if isinstance(ema, dict) else ema
    elif isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
    else:
        state = ck
    return state, (ck.get("epoch") if isinstance(ck, dict) else None)


def export_onnx(cfg_path, ckpt_path, onnx_path):
    """best 权重 → ONNX（与 tests/export2onnx.py 同口径：opset17/batch1/端到端）。"""
    import torch
    import torch.nn as nn
    sys.path.insert(0, PROJ)
    os.chdir(PROJ)
    from src.core import YAMLConfig

    cfg = YAMLConfig(cfg_path)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.model.eval()
    state, _ = load_sd_for_export(ckpt_path)
    cfg.model.load_state_dict(state, strict=True)

    class Net(nn.Module):
        def __init__(self, m, pp):
            super().__init__()
            self.model = m
            self.postprocessor = pp

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    data = torch.rand(1, 3, 640, 640)
    size = torch.tensor([[640, 640]], dtype=torch.int64)
    net = Net(cfg.model, cfg.postprocessor).eval()
    with torch.no_grad():
        torch.onnx.export(
            net, (data, size), onnx_path, dynamo=False,
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes={"images": {0: "N"}, "orig_target_sizes": {0: "N"}},
            opset_version=17)
    return os.path.getsize(onnx_path)


def out_dir_of(ds):
    return os.path.join(OUT_ROOT, "n_%s" % ds.replace("detect_", ""))


def archive(ds, cfg_path):
    """训练完成 → 归档 best 权重 + onnx + 元数据。"""
    out_dir = out_dir_of(ds)
    log_txt = os.path.join(out_dir, "log.txt")
    mAP95, mAP50, best_ep = parse_best(log_txt)
    if mAP95 is None:
        return None, "log.txt 无 best_stat（训练可能中断）"

    src = os.path.join(out_dir, "best_stg2.pth")
    if not os.path.isfile(src):
        src = os.path.join(out_dir, "best_stg1.pth")
    if not os.path.isfile(src):
        return None, "未找到 best_stg1/2.pth"

    done_ep = EPOCHS
    try:
        import torch
        c = torch.load(src, map_location="cpu", weights_only=False)
        ep = c.get("last_epoch", c.get("epoch"))
        if isinstance(ep, int) and ep >= 0:
            done_ep = ep + 1
    except Exception:
        pass

    name = "best_%s_%s_imgsz%d_ep%d_done%d_mAP50-95_0p%05d" % (
        ds, ARCH, IMGSZ, EPOCHS, done_ep, int(round(mAP95 * 100000)))
    pth_dst = os.path.join(BEST_DIR, name + ".pth")
    onnx_dst = os.path.join(BEST_DIR, name + ".onnx")
    txt_dst = os.path.join(BEST_DIR, name + ".pt.txt")

    shutil.copy2(src, pth_dst)
    onnx_ok = False
    try:
        export_onnx(cfg_path, src, onnx_dst)
        onnx_ok = True
    except Exception as e:
        log("ONNX 导出失败（不影响权重归档）：%r" % e)
        if os.path.isfile(onnx_dst):
            try:
                os.remove(onnx_dst)
            except OSError:
                pass

    meta = {
        "dataset_tag": ds,
        "data_yaml": os.path.join(DATA_ROOT, ds, "dataset.yml"),
        "base_model": "PPHGNetV2_B0_stage1.pth (D-FINE 官方 stage1)",
        "arch": ARCH, "imgsz": IMGSZ, "batch": BATCH,
        "total_epochs": EPOCHS, "actual_epochs": done_ep, "best_epoch": best_ep,
        "mAP50": ("%.5f" % mAP50) if mAP50 is not None else "",
        "mAP50-95": "%.5f" % mAP95,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_command": "train.py -c %s %s --seed=0" % (cfg_path, "" if ds in NO_AMP else "--use-amp"),
        "run_dir": out_dir, "archived_file": os.path.basename(pth_dst),
        "onnx_exported": onnx_ok,
    }
    with open(txt_dst, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log("归档完成：%s (mAP50-95=%.5f, mAP50=%s)" % (name, mAP95, meta["mAP50"]))
    return {"name": name, "mAP50-95": mAP95, "mAP50": mAP50,
            "epochs_done": done_ep, "best_epoch": best_ep,
            "onnx": onnx_ok}, None


def train_one(ds, st):
    cfg_path = gen_config(ds)
    out_dir = out_dir_of(ds)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    st["running"] = {"dataset": ds, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_status(st)
    cmd = [VENV_PY, os.path.join(PROJ, "train.py"), "-c", cfg_path,
           "--seed=0", "--output-dir", out_dir]
    if ds not in NO_AMP:
        cmd += ["--use-amp"]
    # [fix] 断点续训：output 目录里有 last.pth 就从上次中断处继续（重试不再浪费已训周期）
    last_pth = os.path.join(out_dir, "last.pth")
    if os.path.isfile(last_pth):
        cmd += ["-r", last_pth]
        log("%s 检测到 last.pth，断点续训" % ds)
    log("开始训练 %s  ->  %s" % (ds, out_dir))
    with open(os.path.join(out_dir, "console.log"), "w", encoding="utf-8") as cf:
        p = subprocess.run(cmd, cwd=PROJ, stdout=cf,
                           stderr=subprocess.STDOUT, encoding="utf-8",
                           errors="ignore")
    hours = (time.time() - t0) / 3600
    log("%s 训练进程退出 rc=%d，耗时 %.2f h" % (ds, p.returncode, hours))
    st["running"] = None
    if p.returncode != 0:
        st["failed"][ds] = {"rc": p.returncode, "hours": round(hours, 2),
                            "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_status(st)
        return False
    info, err = archive(ds, cfg_path)
    if info:
        info["hours"] = round(hours, 2)
        st["done"][ds] = info
    else:
        st["failed"][ds] = {"error": err, "hours": round(hours, 2),
                            "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_status(st)
    return True


def main(argv):
    targets = argv[1:] or DATASETS
    os.makedirs(BEST_DIR, exist_ok=True)
    st = load_status()
    log("=== D-FINE N 批量训练开始，共 %d 个数据集 ===" % len(targets))
    for i, ds in enumerate(targets, 1):
        if ds in st["done"]:
            log("(%d/%d) 跳过 %s：已归档 %s" % (i, len(targets), ds, st["done"][ds]["name"]))
            continue
        log("(%d/%d) ===== %s =====" % (i, len(targets), ds))
        try:
            train_one(ds, st)
        except Exception as e:
            st["running"] = None
            st["failed"][ds] = {"error": repr(e),
                                "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_status(st)
            log("%s 异常：%r" % (ds, e))
    log("=== 批量结束 done=%d failed=%d ===" % (len(st["done"]), len(st["failed"])))
    for ds, d in sorted(st["done"].items()):
        log("  [OK] %-38s mAP50-95=%s  %.1fh" % (ds, d["mAP50-95"], d.get("hours", 0)))
    for ds, d in sorted(st["failed"].items()):
        log("  [FAIL] %-38s %s" % (ds, d))


if __name__ == "__main__":
    main(sys.argv)
