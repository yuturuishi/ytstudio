# -*- coding: utf-8 -*-
"""用 ONNX Runtime 推理测试图：检出效果 + 每张图片的推理耗时（+ 可选保真复核）。

模型位置：weights/onnx/dfine_<档>.onnx（由 export/20_export_onnx.py 生成）
测试图：  tests/bus.jpg 等本目录图片，或 --images 指定
结果图：  tests/out/<运行时间戳>/out_onnx_<图>_<档>.jpg
日志：    tests/logs/test_onnx_<时间戳>_<pid>.log

用法（在工程根目录执行）：
    venv\\Scripts\\python.exe tests\\test_onnx.py                # 五档全跑
    venv\\Scripts\\python.exe tests\\test_onnx.py n x            # 指定档
    venv\\Scripts\\python.exe tests\\test_onnx.py --conf 0.3 --repeat 20
    venv\\Scripts\\python.exe tests\\test_onnx.py --verify           # 加 .pth 保真比对
    venv\\Scripts\\python.exe tests\\test_onnx.py --images a.jpg b.jpg

注意：--verify 会建 PyTorch 模型，而 D-FINE 的 YAMLConfig 有全局配置残留，
多档位时本脚本会自动逐档起独立子进程（与 export/_isolate.py 同一套逻辑）。
"""
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ONNX_DIR = os.path.join(PROJ, "weights", "onnx")
WEIGHTS = os.path.join(PROJ, "weights")
RUN_TAG = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR = os.path.join(HERE, "out", RUN_TAG)
VARIANTS = ["n", "s", "m", "l", "x"]
SIZE = 640
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
COCO80_CN = [
    "人", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船",
    "交通信号灯", "消防栓", "停车标志", "停车计时器", "长椅", "鸟", "猫",
    "狗", "马", "羊", "牛", "大象", "熊", "斑马", "长颈鹿", "背包",
    "雨伞", "手提包", "领带", "行李箱", "飞盘", "双板滑雪板", "单板滑雪板", "运动球",
    "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍",
    "瓶子", "酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果",
    "三明治", "橙子", "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子",
    "沙发", "盆栽", "床", "餐桌", "马桶", "电视", "笔记本", "鼠标",
    "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机", "水槽", "冰箱",
    "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷",
]
COLORS = [
    (60, 90, 220), (90, 170, 70), (230, 150, 40), (200, 80, 200), (40, 190, 200),
    (200, 120, 60), (150, 80, 200), (70, 140, 230), (110, 180, 120), (190, 90, 110),
]


def setup_log():
    """stdout/stderr 镜像到 UTF-8 日志；名字带 pid，避免同秒派发时撞名被沙箱拦。"""
    import io

    d = os.path.join(HERE, "logs")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(
        d, "test_onnx_%s_%d.log" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid())
    )
    f = open(path, "w", encoding="utf-8", newline="\n")

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self._s = [x for x in streams if x is not None]

        def write(self, s):
            for st in self._s:
                try:
                    st.write(s)
                    st.flush()
                except Exception:  # noqa: BLE001
                    pass
            return len(s)

        def flush(self):
            for st in self._s:
                try:
                    st.flush()
                except Exception:  # noqa: BLE001
                    pass

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    return path


LOG = setup_log()
CHILD_FLAG = "_DFINE_TEST_ONNX_CHILD"


def preprocess(img, cv2, np):
    r = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def postprocess(labels, boxes, scores, W, H, conf, max_det, dedup_iou=0.65):
    import numpy as np

    sc = np.asarray(scores).reshape(-1)
    lb = np.asarray(labels).reshape(-1)
    bx = np.asarray(boxes).reshape(-1, 4)
    keep = np.where(sc >= conf)[0]
    keep = keep[np.argsort(-sc[keep])][: max(max_det * 3, 60)]
    dets = []
    for i in keep:
        x1, y1, x2, y2 = (float(v) for v in bx[i])
        x1, y1 = max(0.0, min(x1, W)), max(0.0, min(y1, H))
        x2, y2 = max(0.0, min(x2, W)), max(0.0, min(y2, H))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        dets.append({"cls": int(lb[i]), "conf": float(sc[i]), "box": [x1, y1, x2, y2]})
    out = []
    for d in sorted(dets, key=lambda z: -z["conf"]):
        if any(k["cls"] == d["cls"] and iou(k["box"], d["box"]) > dedup_iou for k in out):
            continue
        out.append(d)
        if len(out) >= max_det:
            break
    return out


def draw(img, dets, cv2, title=""):
    out = img.copy()
    for d in dets:
        c = COLORS[d["cls"] % len(COLORS)]
        x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        txt = "%s %.2f" % (COCO80[d["cls"]], d["conf"])
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 6
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + bl), c, -1)
        cv2.putText(out, txt, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def bench_ort(path, x, wh, repeat, np):
    import onnxruntime as ort

    so = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inames = [i.name for i in so.get_inputs()]
    feeds = {inames[0]: x, inames[1]: np.ascontiguousarray(wh)}
    for _ in range(3):
        so.run(None, feeds)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        so.run(None, feeds)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {
        "mean": statistics.mean(ts),
        "median": statistics.median(ts),
        "min": ts[0],
        "max": ts[-1],
        "p90": ts[min(len(ts) - 1, int(round(0.9 * (len(ts) - 1))))],
        "n": len(ts),
    }


def verify_vs_pth(variant, img_path, x, size, out_ort, thr=0.3):
    """保真复核：同图同输入比对 PyTorch(.pth) 与 ONNX。

    只统计 score>=thr 的有效槽位 + 后处理后的逐框配对 —— 随机噪声/全 300 槽位
    的坐标差没有意义（分数≈0 的退化槽位 1 个 ULP 就能摆几百像素）。
    """
    import numpy as np
    import torch
    import torch.nn as nn

    sys.path.insert(0, PROJ)
    from src.core import YAMLConfig  # noqa: PLC0415

    cfg_path = os.path.join(PROJ, "configs", "dfine", "dfine_hgnetv2_%s_coco.yml" % variant)
    pth_path = os.path.join(WEIGHTS, "dfine_%s_coco.pth" % variant)
    cfg = YAMLConfig(cfg_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg.model.load_state_dict(state, strict=True)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    net = Model().eval()
    with torch.no_grad():
        out_pt = net(torch.from_numpy(x), torch.from_numpy(size))
    pt = [o.numpy() for o in out_pt]

    pt_sc = pt[2].reshape(-1)
    ort_sc = np.asarray(out_ort[2]).reshape(-1)
    pt_lb = pt[0].reshape(-1).astype("int64")
    ort_lb = np.asarray(out_ort[0]).reshape(-1).astype("int64")
    pt_bx = pt[1].reshape(-1, 4).astype("float64")
    ort_bx = np.asarray(out_ort[1]).reshape(-1, 4).astype("float64")
    print("[%s] 全 300 槽位 boxes 最大差 %.3e（含分数≈0 的退化槽位，仅供参考）" % (
        variant, np.abs(pt_bx - ort_bx).max()))

    mask = np.maximum(pt_sc, ort_sc) >= thr
    if mask.sum():
        print("[%s] score>=%.2f 有效槽位 %d 个：标签一致 %d/%d | boxes 最大差 %.3e px" % (
            variant, thr, int(mask.sum()), int((pt_lb[mask] == ort_lb[mask]).sum()),
            int(mask.sum()), np.abs(pt_bx[mask] - ort_bx[mask]).max()))

    H, W = int(size[0, 1]), int(size[0, 0])
    d_pt = postprocess(pt[0], pt[1], pt[2], W, H, 0.4, 60)
    d_ort = postprocess(out_ort[0], out_ort[1], out_ort[2], W, H, 0.4, 60)
    print("[%s] 后处理(conf>=0.40) PyTorch %d 个 -> ONNX %d 个" % (variant, len(d_pt), len(d_ort)))
    if d_pt and len(d_pt) == len(d_ort):
        print("[%s] 逐框配对：标签全一致 %s | 坐标最大偏差 %.2f px | 置信度最大偏差 %.5f" % (
            variant,
            all(a["cls"] == b["cls"] for a, b in zip(d_pt, d_ort)),
            max(max(abs(u - v) for u, v in zip(a["box"], b["box"])) for a, b in zip(d_pt, d_ort)),
            max(abs(a["conf"] - b["conf"]) for a, b in zip(d_pt, d_ort))))


def main(argv):
    import cv2
    import numpy as np

    conf = 0.4
    repeat = 15
    do_verify = False
    variants = []
    imgs = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--conf":
            i += 1
            conf = float(argv[i])
        elif a == "--repeat":
            i += 1
            repeat = int(argv[i])
        elif a == "--verify":
            do_verify = True
        elif a == "--images":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                imgs.append(argv[i])
                i += 1
            continue
        elif a in VARIANTS:
            variants.append(a)
        i += 1
    variants = variants or VARIANTS
    if not imgs:
        for c in [os.path.join(HERE, "bus.jpg")]:
            if os.path.isfile(c):
                imgs.append(c)
    imgs = [p for p in imgs if os.path.isfile(p)]
    if not imgs:
        print("!! 没有可用测试图，请用 --images 指定，或把 bus.jpg 放到 tests/ 下")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)

    # --verify 且多档位：YAMLConfig 全局残留，必须逐档独立进程
    if do_verify and len(variants) > 1 and not os.environ.get(CHILD_FLAG):
        print("检测到多档位 %s + --verify -> 逐档起独立进程（规避 YAMLConfig 全局配置污染）" % variants)
        rc = 0
        for v in variants:
            env = dict(os.environ)
            env[CHILD_FLAG] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            p = __import__("subprocess").run(
                [sys.executable, os.path.abspath(__file__), v, "--verify",
                 "--conf", str(conf), "--repeat", str(repeat)] + imgs,
                cwd=PROJ, env=env)
            rc |= p.returncode
        return rc

    print("ONNX Runtime 推理测试 | 档位 %s | conf=%.2f | 重复 %d 次 | 保真复核 %s" % (
        variants, conf, repeat, "开" if do_verify else "关"))
    print("测试图 %d 张：" % len(imgs))
    for p in imgs:
        print("   %s" % p)
    print("日志: %s" % LOG)
    import onnxruntime as ort
    print("onnxruntime %s | CPU: %s | 逻辑核 %d" % (
        ort.__version__, os.environ.get("PROCESSOR_IDENTIFIER", "?"), os.cpu_count() or 0))
    print("=" * 96)

    summary = []
    for v in variants:
        onnx_p = os.path.join(ONNX_DIR, "dfine_%s.onnx" % v)
        if not os.path.isfile(onnx_p):
            print("[%s] 缺少 %s（先跑 tests\\export2onnx.py）" % (v, os.path.basename(onnx_p)))
            continue
        sess = ort.InferenceSession(onnx_p, providers=["CPUExecutionProvider"])
        inames = [i.name for i in sess.get_inputs()]
        print("[%s] 模型 %.1f MB | 输入 %s / %s" % (
            v, os.path.getsize(onnx_p) / 1048576.0, inames[0], inames[1]))

        for path in imgs:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            t0 = time.perf_counter()
            x = preprocess(img, cv2, np)
            t_pre = (time.perf_counter() - t0) * 1000
            wh = np.array([[W, H]], np.int64)
            feeds = {inames[0]: x, inames[1]: np.ascontiguousarray(wh)}
            t0 = time.perf_counter()
            out = sess.run(None, feeds)
            t_first = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dets = postprocess(out[0], out[1], out[2], W, H, conf, 60)
            t_post = (time.perf_counter() - t0) * 1000
            st = bench_ort(onnx_p, x, wh, repeat, np)
            print("-" * 96)
            print("[%s] %s  %dx%d | 预处理 %.1f ms | 首次推理 %.1f ms | 后处理 %.1f ms" % (
                v, os.path.basename(path), W, H, t_pre, t_first, t_post))
            print("[%s] 纯推理(%.0f 次): 平均 %.1f ms | 中位 %.1f | 最快 %.1f | 最慢 %.1f | P90 %.1f" % (
                v, st["n"], st["mean"], st["median"], st["min"], st["max"], st["p90"]))
            print("[%s] 端到端(预处理+推理+后处理) 约 %.1f ms" % (v, t_pre + st["median"] + t_post))
            uniq = {}
            for d in dets:
                uniq[d["cls"]] = uniq.get(d["cls"], 0) + 1
            print("[%s] 检出 %d 个目标 / %d 类：%s" % (
                v, len(dets), len(uniq),
                " / ".join("%s x%d" % (COCO80_CN[k], c) for k, c in sorted(uniq.items(), key=lambda z: -z[1])) or "无"))
            for d in dets[:8]:
                b = d["box"]
                print("      %-14s %.3f   [%.0f, %.0f, %.0f, %.0f]" % (
                    COCO80[d["cls"]] + "/" + COCO80_CN[d["cls"]], d["conf"], b[0], b[1], b[2], b[3]))
            outp = os.path.join(OUT_DIR, "out_onnx_%s_%s.jpg" % (
                os.path.splitext(os.path.basename(path))[0], v))
            cv2.imencode(".jpg", draw(img, dets, cv2, "ONNXRuntime D-FINE-%s  %s  %.1f ms" % (
                v, os.path.basename(path), st["median"])))[1].tofile(outp)
            print("[%s] 结果图 %s" % (v, os.path.relpath(outp, PROJ)))
            if do_verify:
                try:
                    verify_vs_pth(v, path, x, wh, out)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    print("[%s] 保真复核失败: %s" % (v, e))
                    traceback.print_exc()
            summary.append({"variant": v, "image": os.path.basename(path), "ndet": len(dets), **st})

    print("=" * 96)
    print("汇总：纯推理耗时（ms，同一张图、同一 CPU）")
    print("%-6s %-14s %8s %8s %8s %8s %8s %7s" % (
        "档位", "图片", "平均", "中位", "最快", "最慢", "P90", "目标数"))
    for s in summary:
        print("%-6s %-14s %8.1f %8.1f %8.1f %8.1f %8.1f %7d" % (
            s["variant"], s["image"], s["mean"], s["median"], s["min"], s["max"], s["p90"], s["ndet"]))
    print("说明：目标数为 conf>=%.2f 且同类去重后的数量。" % conf)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
