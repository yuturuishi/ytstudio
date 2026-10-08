# -*- coding: utf-8 -*-
"""D-FINE 官方 .pth -> ONNX（端到端，含部署后处理，无 NMS）。

对官方 tools/deployment/export_onnx.py 做了三处必要修正（它的默认值在本机会踩坑）：
  1) opset_version 16 -> 17 ：DETR/Transformer 系低 opset 导出后精度会【静默崩坏】
  2) dummy 输入 batch 32 -> 1 ：官方写死 (32,3,640,640)，导出件会带死 batch 维
  3) 不做 --check / --simplify ：依赖 pycocotools；且 dynamic_axes 下 onnxsim 会改坏形状

⚠️ D-FINE 的 YAMLConfig 有【全局配置残留】：同一进程连续加载多档配置会互相串味
（后档继承前档的 dim_feedforward/use_lab 等字段），表现为 state_dict "size mismatch /
missing keys lab.*" 的假故障。所以多档位时本脚本自动逐档起独立子进程。

产物：weights/onnx/dfine_<档>.onnx（weights/openvino 由 export2openvino.py 生成）
日志：tests/logs/export2onnx_<时间戳>_<pid>.log
真实图片的保真复核用 tests/test_onnx.py --verify（随机噪声下 300 槽位里大量
分数≈0 的退化槽位坐标没有意义，别被几百像素的 boxes 差吓到）。

用法（在工程根目录执行）：
    venv\\Scripts\\python.exe tests\\export2onnx.py          # 全部五档
    venv\\Scripts\\python.exe tests\\export2onnx.py s m      # 只导指定档
权重缺失时会自动从 D-FINE 官方 GitHub release(dfinev1.0) 下载（带镜像回退），
下载的权重对不对，由下面的 state_dict 严格加载当场把关。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
WEIGHTS = os.path.join(PROJ, "weights")
ONNX_DIR = os.path.join(WEIGHTS, "onnx")
VARIANTS = ["n", "s", "m", "l", "x"]
OPSET = 17  # 关键：不能低于 17
BATCH = 1
SIZE = 640
CHILD_FLAG = "_DFINE_EXPORT_CHILD"

GH = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{v}_coco.pth"
MIRRORS = [
    "{u}",  # 直连 GitHub
    "https://ghfast.top/{u}",
    "https://gh-proxy.com/{u}",
    "https://ghproxy.net/{u}",
    "https://ghproxy.cc/{u}",
]


def setup_log():
    """stdout/stderr 镜像到 UTF-8 日志；名字带 pid，避免父子同秒派发时撞名被沙箱拦。"""
    import io

    d = os.path.join(HERE, "logs")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(
        d, "export2onnx_%s_%d.log" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid())
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


def fetch_weight_if_missing(variant):
    """权重缺失时从官方 release 下载（带镜像回退）。好坏由 strict 加载当场把关。"""
    import urllib.request

    pth_path = os.path.join(WEIGHTS, "dfine_%s_coco.pth" % variant)
    if os.path.isfile(pth_path) and os.path.getsize(pth_path) > 1 << 20:
        return pth_path
    os.makedirs(WEIGHTS, exist_ok=True)
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
        except Exception as e:  # noqa: BLE001
            last = e
            print("[%s] 该镜像失败: %s" % (variant, e))
    raise RuntimeError("五条下载通道全部失败: %s" % last)


def atomic_replace(src, dst):
    """D 盘沙箱可能禁止直接覆盖已存在文件：先删旧的（失败不阻塞），再改名替换。"""
    if os.path.exists(dst):
        try:
            os.remove(dst)
        except OSError as e:
            print("  (旧文件删除失败，交由 os.replace 尝试: %s)" % e)
    os.replace(src, dst)


def build_and_export(variant):
    import numpy as np
    import torch
    import torch.nn as nn

    sys.path.insert(0, PROJ)
    from src.core import YAMLConfig  # noqa: PLC0415

    cfg_path = os.path.join(PROJ, "configs", "dfine", "dfine_hgnetv2_%s_coco.yml" % variant)
    pth_path = fetch_weight_if_missing(variant)
    onnx_path = os.path.join(ONNX_DIR, "dfine_%s.onnx" % variant)
    assert os.path.isfile(cfg_path), "缺少配置: %s" % cfg_path

    print("=" * 74)
    print("[%s] 配置 %s" % (variant, os.path.relpath(cfg_path, PROJ)))
    print("[%s] 权重 %s (%.1f MB)" % (variant, os.path.basename(pth_path), os.path.getsize(pth_path) / 1048576.0))

    cfg = YAMLConfig(cfg_path)
    if "HGNetv2" in cfg.yaml_cfg:
        # 不下载 ImageNet 预训练骨干，直接吃 .pth 的权重
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

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
    except Exception as e:  # noqa: BLE001
        # 多半是「多档同进程配置串味」假故障；单档进程里还报这个就是权重真不匹配
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
    data = torch.rand(BATCH, 3, SIZE, SIZE)
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
        # 旧版 torch 没有 dynamo 参数
        torch.onnx.export(net, (data, size), tmp_path, **kwargs)
    atomic_replace(tmp_path, onnx_path)

    mb = os.path.getsize(onnx_path) / 1048576.0
    print("[%s] 导出 ONNX ok  %s  (%.1f MB, opset=%d)" % (variant, os.path.basename(onnx_path), mb, OPSET))

    # ---- 冒烟校验：ORT 能跑、输出形状对、与 PyTorch 输出同分布 ----
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        feeds = {"images": data.numpy().astype(np.float32), "orig_target_sizes": size.numpy().astype(np.int64)}
        out_ort = sess.run(None, feeds)
        same = int((out_pt[0].numpy().astype("int64") == np.asarray(out_ort[0]).astype("int64")).sum())
        ds = float(np.abs(out_pt[2].numpy().astype("float64") - np.asarray(out_ort[2], dtype="float64")).max())
        db = float(np.abs(out_pt[1].numpy().astype("float64") - np.asarray(out_ort[1], dtype="float64")).max())
        print("[%s]   labels 逐元素一致 %d/%d | scores 最大绝对差 %.3e" % (variant, same, out_pt[0].numel(), ds))
        print("[%s]   boxes 最大绝对差 %.3e（随机噪声输入，含大量分数≈0 的退化槽位，仅供参考）" % (variant, db))
        print("[%s]   真实图片保真复核请跑: tests\\test_onnx.py %s --verify" % (variant, variant))
    except Exception as e:  # noqa: BLE001
        print("[%s]   (ORT 冒烟校验跳过: %s)" % (variant, e))

    return onnx_path


def main(argv):
    variants = [a for a in argv if a in VARIANTS] or VARIANTS
    # 多档位逐档独立进程：D-FINE 的 YAMLConfig 有全局配置残留
    if len(variants) > 1 and not os.environ.get(CHILD_FLAG):
        print("检测到多档位 %s -> 逐档起独立进程（规避 YAMLConfig 全局配置污染）" % variants)
        import subprocess

        rc = 0
        for v in variants:
            env = dict(os.environ)
            env[CHILD_FLAG] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            print("\n>>>> 子进程 export2onnx.py %s" % v)
            p = subprocess.run([sys.executable, os.path.abspath(__file__), v], cwd=PROJ, env=env)
            print("<<<< 档位 %s 退出码 %d" % (v, p.returncode))
            rc |= p.returncode
        return rc

    print("D-FINE -> ONNX 导出 | 档位: %s | opset=%d | batch=%d | 输入 %dx%d" % (variants, OPSET, BATCH, SIZE, SIZE))
    print("日志: %s" % LOG)
    os.chdir(PROJ)  # 配置里的相对路径依赖
    sys.path.insert(0, PROJ)
    t0 = time.perf_counter()
    done = []
    for v in variants:
        try:
            done.append(build_and_export(v))
        except Exception as e:  # noqa: BLE001
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
