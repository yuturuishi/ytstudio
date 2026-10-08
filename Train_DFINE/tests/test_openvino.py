# -*- coding: utf-8 -*-
"""用 OpenVINO 推理测试图：检出效果 + 每张图片的推理耗时。

模型位置：weights/openvino/dfine_<档>_fp32.xml（由 export/30_to_openvino.py 生成）
测试图：  tests/bus.jpg 等本目录图片，或 --images 指定
结果图：  tests/out/<运行时间戳>/out_openvino_<图>_<档>.jpg
日志：    tests/logs/test_openvino_<时间戳>_<pid>.log

用法（在工程根目录执行）：
    venv\\Scripts\\python.exe tests\\test_openvino.py                # 五档全跑
    venv\\Scripts\\python.exe tests\\test_openvino.py n x            # 指定档
    venv\\Scripts\\python.exe tests\\test_openvino.py --conf 0.3 --repeat 20
    venv\\Scripts\\python.exe tests\\test_openvino.py --ort              # 加 ONNX Runtime 对照
    venv\\Scripts\\python.exe tests\\test_openvino.py --images a.jpg b.jpg
"""
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OV_DIR = os.path.join(PROJ, "weights", "openvino")
ONNX_DIR = os.path.join(PROJ, "weights", "onnx")
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
        d, "test_openvino_%s_%d.log" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid())
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


def bench(req, feeds, repeat):
    for _ in range(3):
        req.infer(feeds)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        req.infer(feeds)
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


def bench_ort(path, x, wh, repeat, np):
    """可选对照：同一张图走 ONNX Runtime CPU，用来看 OpenVINO 相对提速多少。"""
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
    return {"median": statistics.median(ts)}


def main(argv):
    import cv2
    import numpy as np
    import openvino as ov

    conf = 0.4
    repeat = 15
    use_ort = False
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
        elif a == "--ort":
            use_ort = True
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

    print("OpenVINO 推理测试 | 档位 %s | conf=%.2f | 重复 %d 次 | ORT 对照 %s" % (
        variants, conf, repeat, "开" if use_ort else "关"))
    print("测试图 %d 张：" % len(imgs))
    for p in imgs:
        print("   %s" % p)
    print("日志: %s" % LOG)

    core = ov.Core()
    cpu_name = "?"
    for key in ("FULL_DEVICE_NAME", "CPU_MODEL_FULLNAME", "DEVICE_NAME"):
        try:
            cpu_name = core.get_property("CPU", key).strip()
            break
        except Exception:  # noqa: BLE001
            continue
    print("OpenVINO %s | CPU: %s | 逻辑核 %d" % (ov.__version__, cpu_name, os.cpu_count() or 0))
    print("=" * 96)

    summary = []
    for v in variants:
        xml = os.path.join(OV_DIR, "dfine_%s_fp32.xml" % v)
        if not os.path.isfile(xml):
            print("[%s] 缺少 %s（先跑 tests\\export2openvino.py）" % (v, os.path.basename(xml)))
            continue
        t0 = time.perf_counter()
        model = core.read_model(xml)
        in_names = [i.get_any_name() for i in model.inputs]
        n_img = next((n for n in in_names if "images" in n), in_names[0])
        n_sz = next((n for n in in_names if n != n_img), in_names[1])
        try:
            model.reshape({n_img: [1, 3, SIZE, SIZE], n_sz: [1, 2]})
        except Exception as e:  # noqa: BLE001
            print("[%s] reshape 跳过: %s" % (v, e))
        cm = core.compile_model(model, "CPU")
        t_compile = (time.perf_counter() - t0) * 1000
        req = cm.create_infer_request()
        print("[%s] IR 载入+编译耗时 %.0f ms | 输入名 %s / %s" % (v, t_compile, n_img, n_sz))

        for path in imgs:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            t0 = time.perf_counter()
            x = preprocess(img, cv2, np)
            t_pre = (time.perf_counter() - t0) * 1000
            feeds = {n_img: x, n_sz: np.array([[W, H]], np.int64)}
            t0 = time.perf_counter()
            res = req.infer(feeds)
            t_first = (time.perf_counter() - t0) * 1000
            rv = list(res.values())
            t0 = time.perf_counter()
            dets = postprocess(rv[0], rv[1], rv[2], W, H, conf, 60)
            t_post = (time.perf_counter() - t0) * 1000
            st = bench(req, feeds, repeat)
            print("-" * 96)
            print("[%s] %s  %dx%d | 预处理 %.1f ms | 首次推理 %.1f ms | 后处理 %.1f ms" % (
                v, os.path.basename(path), W, H, t_pre, t_first, t_post))
            print("[%s] 纯推理(%.0f 次): 平均 %.1f ms | 中位 %.1f | 最快 %.1f | 最慢 %.1f | P90 %.1f" % (
                v, st["n"], st["mean"], st["median"], st["min"], st["max"], st["p90"]))
            print("[%s] 端到端(预处理+推理+后处理) 约 %.1f ms" % (v, t_pre + st["median"] + t_post))
            ort_ms = None
            if use_ort:
                onnx_p = os.path.join(ONNX_DIR, "dfine_%s.onnx" % v)
                if os.path.isfile(onnx_p):
                    try:
                        st2 = bench_ort(onnx_p, x, np.array([[W, H]], np.int64), repeat, np)
                        ort_ms = st2["median"]
                        print("[%s] 对照 ONNX Runtime CPU 中位 %.1f ms -> OpenVINO 快 %.2fx" % (
                            v, ort_ms, ort_ms / st["median"]))
                    except Exception as e:  # noqa: BLE001
                        print("[%s] ORT 对照跳过: %s" % (v, e))
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
            outp = os.path.join(OUT_DIR, "out_openvino_%s_%s.jpg" % (
                os.path.splitext(os.path.basename(path))[0], v))
            cv2.imencode(".jpg", draw(img, dets, cv2, "OpenVINO D-FINE-%s  %s  %.1f ms" % (
                v, os.path.basename(path), st["median"])))[1].tofile(outp)
            print("[%s] 结果图 %s" % (v, os.path.relpath(outp, PROJ)))
            summary.append({"variant": v, "image": os.path.basename(path), "ndet": len(dets),
                            "pre": t_pre, "first": t_first, "post": t_post, "ort": ort_ms, **st})

    print("=" * 96)
    print("汇总：纯推理耗时（ms，同一张图、同一 CPU）")
    print("%-6s %-14s %8s %8s %8s %8s %8s %7s %10s" % (
        "档位", "图片", "平均", "中位", "最快", "最慢", "P90", "目标数", "ORT中位"))
    for s in summary:
        ort = ("%.1f" % s["ort"]) if s.get("ort") else "-"
        print("%-6s %-14s %8.1f %8.1f %8.1f %8.1f %8.1f %7d %10s" % (
            s["variant"], s["image"], s["mean"], s["median"], s["min"], s["max"], s["p90"],
            s["ndet"], ort))
    print("说明：目标数为 conf>=%.2f 且同类去重后的数量；“ORT中位”是同一 ONNX 走 ONNX Runtime 的中位耗时，用于对照。" % conf)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
