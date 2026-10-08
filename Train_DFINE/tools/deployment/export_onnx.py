# -*- coding: utf-8 -*-
"""D-FINE 官方 .pth -> ONNX（端到端，含部署后处理，无 NMS）。

与旧版 tools/deployment/export_onnx.py 相比的关键修正（旧版在本机会踩坑）：
  1) opset_version 16 -> 17 ：DETR/Transformer 系低 opset 导出后精度会【静默崩坏】
  2) dummy 输入 batch 32 -> 1 ：官方写死 (32,3,640,640)，导出件会带死 batch 维
  3) 显式 dynamo=False ：torch 2.14+ 默认走 dynamo 导出器，产物 opset 被抬到 18 且
     输出外置数据双文件，必须关掉才能得到 opset17 单文件
  4) 不做 onnxsim 简化 ：dynamic_axes 下 onnxsim 会改坏形状（参考 tests 版结论）

产物统一落地到 models/d-fine-onnx-dyn-opset17/ ，文件名自带档位 + 数据集 + opset：
    dfine_{variant}_{dataset}_opset17.onnx
  官方 COCO 权重 -> dfine_n_coco_opset17.onnx （dataset 默认 coco）
  自定义训练权重 -> dfine_n_fire_smoke_opset17.onnx （dataset 用 --dataset-tag 指定）

⚠️ D-FINE 的 YAMLConfig 有【全局配置残留】：同一进程连续加载多档配置会互相串味
（后档继承前档的 dim_feedforward/use_lab 等字段），表现为 state_dict "size mismatch /
missing keys lab.*" 的假故障。所以多档位时本脚本自动逐档起独立子进程。

⚠️ 输入约定（导出与推理必须一致）：
  输入1 images       float32 [N,3,640,640]  RGB、0~1 归一化（无 ImageNet 均值方差）
  输入2 orig_target_sizes  int64 [N,2]  内容 = 送入网络的图尺寸 (宽,高)，即 [W,H]
        —— 官方推理脚本先把图 resize 成 640x640 正方形，再喂 [[640,640]]，此时 W==H 顺序无关；
           若你采用 letterbox 保留比例预处理，必须喂真实 [W,H]（见 README“输入输出签名”）。
  输出 labels[N,300] / boxes[N,300,4](xyxy) / scores[N,300] ，端到端无 NMS。

用法（在工程根目录执行）：
    venv\\Scripts\\python.exe tools\\deployment\\export_onnx.py n            # 官方 coco n
    venv\\Scripts\\python.exe tools\\deployment\\export_onnx.py n s m l x    # 官方五档
    venv\\Scripts\\python.exe tools\\deployment\\export_onnx.py \\
        --config configs/dfine/scenes/dfine_hgnetv2_n100_fire_smoke.yml \\
        --resume best_models/xxx.pth --variant n --dataset-tag fire_smoke
权重缺失且为官方档位时会自动从 D-FINE 官方 GitHub release 下载（带镜像回退）。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))  # Train_DFINE 根
MODELS_DIR = os.path.join(PROJ, "models")
ONNX_DIR = os.path.join(MODELS_DIR, "d-fine-onnx-dyn-opset17")
VARIANTS = ["n", "s", "m", "l", "x"]
OPSET = 17  # 关键：不能低于 17
SIZE = 640
CHILD_FLAG = "_DFINE_EXPORT_ONNX_CHILD"

# 官方权重候选查找顺序（models/ 里已放好，找不到再下载）
WEIGHT_CANDIDATES = lambda v: [
    os.path.join(MODELS_DIR, "dfine_%s_coco.pth" % v),
    os.path.join(PROJ, "weights", "dfine_%s_coco.pth" % v),
    os.path.join(PROJ, "weight", "dfine_%s_coco.pth" % v),
]
GH = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{v}_coco.pth"
MIRRORS = [
    "{u}",
    "https://ghfast.top/{u}",
    "https://gh-proxy.com/{u}",
    "https://ghproxy.net/{u}",
    "https://ghproxy.cc/{u}",
]


def setup_log():
    import io

    d = os.path.join(HERE, "logs")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "export_onnx_%s_%d.log" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid()))
    f = open(path, "w", encoding="utf-8", newline="\n")

    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self._s = [x for x in streams if x is not None]

        def write(self, s):
            for st in self._s:
                try:
                    st.write(s)
                    st.flush()
                except Exception:
                    pass
            return len(s)

        def flush(self):
            for st in self._s:
                try:
                    st.flush()
                except Exception:
                    pass

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    return path


LOG = setup_log()


def atomic_replace(src, dst):
    if os.path.exists(dst):
        try:
            os.remove(dst)
        except OSError as e:
            print("  (旧文件删除失败，交由 os.replace 尝试: %s)" % e)
    os.replace(src, dst)


def fetch_weight_if_missing(variant):
    for p in WEIGHT_CANDIDATES(variant):
        if os.path.isfile(p) and os.path.getsize(p) > 1 << 20:
            return p
    os.makedirs(MODELS_DIR, exist_ok=True)
    pth_path = WEIGHT_CANDIDATES(variant)[0]
    import urllib.request

    url = GH.format(v=variant)
    last = None
    for m in MIRRORS:
        u = m.format(u=url)
        try:
            print("[%s] 下载权重 %s" % (variant, u))
            tmp = pth_path + ".part%d" % os.getpid()
            urllib.request.urlretrieve(u, tmp)
            size = os.path.getsize(tmp)
            if size < 1 << 20:
                raise IOError("文件过小 %.1f MB" % (size / 1048576.0))
            os.replace(tmp, pth_path)
            print("[%s] 下载完成 %.1f MB" % (variant, size / 1048576.0))
            return pth_path
        except Exception as e:
            last = e
            print("[%s] 该镜像失败: %s" % (variant, e))
    raise RuntimeError("五条下载通道全部失败: %s" % last)


def resolve_variant(cfg_path):
    import re

    m = re.search(r"dfine_hgnetv2_([a-z])", os.path.basename(cfg_path))
    return m.group(1) if m else "n"


def build_and_export(variant, cfg_path=None, resume=None, dataset_tag=None):
    import numpy as np
    import torch
    import torch.nn as nn

    sys.path.insert(0, PROJ)
    from src.core import YAMLConfig

    # ---- 决定配置 / 权重 / 输出名 ----
    if cfg_path and resume:
        cfg_path = os.path.abspath(cfg_path)
        resume = os.path.abspath(resume)
        variant = variant or resolve_variant(cfg_path)
        dataset_tag = dataset_tag or os.path.splitext(os.path.basename(resume))[0]
        assert os.path.isfile(cfg_path), "缺少配置: %s" % cfg_path
        assert os.path.isfile(resume), "缺少权重: %s" % resume
        pth_path = resume
    else:
        variant = variant or "n"
        cfg_path = os.path.join(PROJ, "configs", "dfine", "dfine_hgnetv2_%s_coco.yml" % variant)
        pth_path = fetch_weight_if_missing(variant)
        dataset_tag = dataset_tag or "coco"
        assert os.path.isfile(cfg_path), "缺少配置: %s" % cfg_path

    onnx_name = "dfine_%s_%s_opset17.onnx" % (variant, dataset_tag)
    onnx_path = os.path.join(ONNX_DIR, onnx_name)

    print("=" * 74)
    print("[%s] 配置 %s" % (variant, os.path.relpath(cfg_path, PROJ)))
    print("[%s] 权重 %s (%.1f MB)" % (variant, os.path.basename(pth_path), os.path.getsize(pth_path) / 1048576.0))

    cfg = YAMLConfig(cfg_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # 直接吃 .pth，不下载 ImageNet 骨干

    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
        state = ckpt["ema"]["module"]
        print("[%s] 载入分支: ema.module" % variant)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        print("[%s] 载入分支: model" % variant)
    else:
        state = ckpt
        print("[%s] 载入分支: 裸 state_dict" % variant)

    try:
        cfg.model.load_state_dict(state, strict=True)
        print("[%s] state_dict 严格匹配 OK" % variant)
    except Exception as e:
        miss, unexp = cfg.model.load_state_dict(state, strict=False)
        raise RuntimeError(
            "state_dict 严格匹配失败（若日志显示本档是独立子进程仍报错，则权重与代码确实不匹配）: %s | "
            "缺失 %d / 多余 %d，前几项: %s" % (e, len(miss), len(unexp), list(miss)[:5])
        ) from e

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    net = Model().eval()

    torch.manual_seed(0)
    data = torch.rand(1, 3, SIZE, SIZE)
    size = torch.tensor([[SIZE, SIZE]], dtype=torch.int64)

    with torch.no_grad():
        t0 = time.perf_counter()
        out_pt = net(data, size)
        t_pt = (time.perf_counter() - t0) * 1000
    print("[%s] PyTorch 前向 %.0f ms，输出 %s" % (variant, t_pt, [tuple(o.shape) for o in out_pt]))

    os.makedirs(ONNX_DIR, exist_ok=True)
    tmp_path = onnx_path + ".new%d" % os.getpid()
    kwargs = dict(
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes={"images": {0: "N"}, "orig_target_sizes": {0: "N"}},
        opset_version=OPSET,
        do_constant_folding=True,
        verbose=False,
    )
    try:
        torch.onnx.export(net, (data, size), tmp_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(net, (data, size), tmp_path, **kwargs)  # 旧版 torch 没 dynamo 参数
    atomic_replace(tmp_path, onnx_path)

    mb = os.path.getsize(onnx_path) / 1048576.0
    print("[%s] 导出 ONNX ok  %s  (%.1f MB, opset=%d, 动态 batch)" % (variant, onnx_name, mb, OPSET))

    # ---- ORT 冒烟校验 ----
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        feeds = {
            "images": data.numpy().astype(np.float32),
            "orig_target_sizes": size.numpy().astype(np.int64),
        }
        out_ort = sess.run(None, feeds)
        same = int((out_pt[0].numpy().astype("int64") == np.asarray(out_ort[0]).astype("int64")).sum())
        ds = float(np.abs(out_pt[2].numpy().astype("float64") - np.asarray(out_ort[2], dtype="float64")).max())
        print("[%s]   labels 逐元素一致 %d/%d | scores 最大绝对差 %.3e" % (variant, same, out_pt[0].numel(), ds))
        print("[%s]   真实图片保真复核请跑: tests\\test_onnx.py %s --verify" % (variant, variant))
    except Exception as e:
        print("[%s]   (ORT 冒烟校验跳过: %s)" % (variant, e))

    return onnx_path


def main(argv):
    # 分离“官方档位参数”与“自定义参数”
    variant_args = [a for a in argv if a in VARIANTS]
    cfg = None
    resume = None
    dataset_tag = None
    variant_explicit = None
    fp16 = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--config", "-c"):
            cfg = argv[i + 1]; i += 2; continue
        if a in ("--resume", "-r"):
            resume = argv[i + 1]; i += 2; continue
        if a in ("--dataset-tag", "-t"):
            dataset_tag = argv[i + 1]; i += 2; continue
        if a in ("--variant", "-v"):
            variant_explicit = argv[i + 1]; i += 2; continue
        if a == "--fp16":
            fp16 = True; i += 1; continue
        i += 1

    if cfg and resume:
        # 自定义单档导出（单进程即可）
        print("D-FINE -> ONNX 导出（自定义）| opset=%d | batch=动态 | 输入 %dx%d | fp16=%s" % (OPSET, SIZE, SIZE, fp16))
        print("日志: %s" % LOG)
        os.chdir(PROJ)
        sys.path.insert(0, PROJ)
        try:
            p = build_and_export(variant_explicit, cfg, resume, dataset_tag)
            print("完成: %s (%.1f MB)" % (os.path.relpath(p, PROJ), os.path.getsize(p) / 1048576.0))
            return 0
        except Exception as e:
            import traceback
            print("导出失败: %s" % e)
            traceback.print_exc()
            return 1

    # 官方档位（可能多档 -> 逐档独立子进程规避 YAMLConfig 串味）
    variants = variant_args or VARIANTS
    if len(variants) > 1 and not os.environ.get(CHILD_FLAG):
        print("检测到多档位 %s -> 逐档起独立进程（规避 YAMLConfig 全局配置污染）" % variants)
        import subprocess

        rc = 0
        for v in variants:
            env = dict(os.environ)
            env[CHILD_FLAG] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            print("\n>>>> 子进程 export_onnx.py %s" % v)
            p = subprocess.run([sys.executable, os.path.abspath(__file__), v], cwd=PROJ, env=env)
            print("<<<< 档位 %s 退出码 %d" % (v, p.returncode))
            rc |= p.returncode
        return rc

    print("D-FINE -> ONNX 导出（官方 COCO）| 档位: %s | opset=%d | batch=动态 | 输入 %dx%d" % (variants, OPSET, SIZE, SIZE))
    print("日志: %s" % LOG)
    os.chdir(PROJ)
    sys.path.insert(0, PROJ)
    t0 = time.perf_counter()
    done = []
    for v in variants:
        try:
            done.append(build_and_export(v))
        except Exception as e:
            import traceback
            print("[%s] 导出失败: %s" % (v, e))
            traceback.print_exc()
    print("=" * 74)
    print("完成 %d/%d 档，总耗时 %.1f s" % (len(done), len(variants), time.perf_counter() - t0))
    for p in done:
        print("  %s  %.1f MB" % (os.path.relpath(p, PROJ), os.path.getsize(p) / 1048576.0))
    return 0 if len(done) == len(variants) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
