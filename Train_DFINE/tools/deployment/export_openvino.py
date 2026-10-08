# -*- coding: utf-8 -*-
"""D-FINE ONNX -> OpenVINO IR（xml + bin），静态 fp32（默认），可静态 fp16。

产物统一落地到 models/d-fine-openvino2024.6-static-fp32/ ，文件名自带版本 + 精度 + 静态：
    dfine_{variant}_{dataset}_ov2024.6_fp32_static.xml + .bin
    dfine_{variant}_{dataset}_ov2024.6_fp16_static.xml + .bin   (--fp16)
“2024.6” 是【目标运行时版本】（rebekah 内置 OpenVINO 2024.6），不代表转换器版本；
任意 >= 2024.6 的 ov 转换器产出的 IR v11 都能被 2024.6 向前兼容加载（已实测）。

为什么要“静态”而不是直接用动态 IR：
  OpenVINO 2024.6 的 GPU 插件对【动态 batch】编译会出错/输出垃圾；CPU 上静态形状也更快。
  所以这里 read_model -> reshape 成 [1,3,640,640] + [1,2] -> serialize，batch 维由 -1 定死为 1。

⚠️ D-FINE fp16 在 GPU 上的坑（rebekah 实测，2026-10-06）：
  S/M/L/X 中间值 fp16 溢出 -> 输出全垃圾；N 档侥幸正常，更具迷惑性。
  即便这里导出 fp16 IR，在 GPU 上仍需强制 f32 推理精度 hint 才安全。
  结论：除非你只在 CPU 上跑或用 2026.4.1+ 运行时，否则默认 fp32 最稳。

用法（在工程根目录执行）：
    venv\\Scripts\\python.exe tools\\deployment\\export_openvino.py \\
        models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx
    venv\\Scripts\\python.exe tools\\deployment\\export_openvino.py \\
        models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx --target-ov 2024.6
    venv\\Scripts\\python.exe tools\\deployment\\export_openvino.py \\
        models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx --fp16
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
MODELS_DIR = os.path.join(PROJ, "models")
OV_DIR = os.path.join(MODELS_DIR, "d-fine-openvino2024.6-static-fp32")
SIZE = 640


def setup_log():
    import io

    d = os.path.join(HERE, "logs")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "export_openvino_%s_%d.log" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid()))
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


def convert_one(onnx_path, target_ov, fp16, dynamic):
    import numpy as np
    import openvino as ov

    assert os.path.isfile(onnx_path), "缺少 ONNX: %s（先跑 tools\\deployment\\export_onnx.py）" % onnx_path
    import re

    stem = os.path.splitext(os.path.basename(onnx_path))[0]  # dfine_n_coco_opset17
    stem = re.sub(r"_opset\d+$", "", stem)  # -> dfine_n_coco（去掉 onnx 的 opset 标记）
    suffix = "%s_%s" % (target_ov, "fp16" if fp16 else "fp32")
    shape_tag = "" if dynamic else "_static"
    xml_name = "%s_ov%s%s.xml" % (stem, suffix, shape_tag)
    xml_path = os.path.join(OV_DIR, xml_name)

    print("=" * 74)
    print("ONNX %.1f MB -> IR(%s%s)" % (os.path.getsize(onnx_path) / 1048576.0, "fp16" if fp16 else "fp32", shape_tag))
    t0 = time.perf_counter()
    model = ov.convert_model(onnx_path)
    t_conv = time.perf_counter() - t0

    ins = [(p.get_any_name(), list(p.partial_shape), str(p.get_element_type())) for p in model.inputs]
    outs = [(p.get_any_name(), list(p.partial_shape), str(p.get_element_type())) for p in model.outputs]
    print("输入 %s" % ins)
    print("输出 %s" % outs)

    # ---- 静态模式：直接在内存里的 model 上 reshape，再一次性 serialize（避免 read/save 回环损坏） ----
    if not dynamic:
        shape = {i.get_any_name(): [1, 3, SIZE, SIZE] if "images" in i.get_any_name() else [1, 2] for i in model.inputs}
        model.reshape(shape)
        print("reshape -> 静态 %s" % shape)

    os.makedirs(OV_DIR, exist_ok=True)
    tmp_xml = os.path.splitext(xml_path)[0] + ".new%d.xml" % os.getpid()
    ov.save_model(model, tmp_xml, compress_to_fp16=fp16)
    atomic_replace(tmp_xml, xml_path)
    bin_path = xml_path.replace(".xml", ".bin")
    atomic_replace(tmp_xml.replace(".xml", ".bin"), bin_path)
    print("转换 %.1f s -> %s (%.1f MB) + .bin (%.1f MB)" % (
        t_conv, xml_name,
        os.path.getsize(xml_path) / 1048576.0, os.path.getsize(bin_path) / 1048576.0))

    # ---- 编译冒烟（读回刚写的静态 IR） ----
    core = ov.Core()
    t0 = time.perf_counter()
    cm = core.compile_model(xml_path, "CPU")
    t_compile = time.perf_counter() - t0
    in_names = [i.get_any_name() for i in cm.inputs]
    req = cm.create_infer_request()
    x = np.zeros((1, 3, SIZE, SIZE), np.float32)
    s = np.array([[SIZE, SIZE]], np.int64)
    t0 = time.perf_counter()
    res = req.infer({in_names[0]: x, in_names[1]: s})
    t_first = time.perf_counter() - t0
    print("编译 %.1f s；首次推理 %.0f ms；输出 %s" % (
        t_compile, t_first * 1000, [np.asarray(v).shape for v in res.values()]))
    return xml_path


def main(argv):
    onnx_paths = [a for a in argv if a.endswith(".onnx")]
    target_ov = "2024.6"
    fp16 = "--fp16" in argv
    dynamic = "--dynamic" in argv
    if "--target-ov" in argv:
        target_ov = argv[argv.index("--target-ov") + 1]
    print("D-FINE ONNX -> OpenVINO IR | 目标运行时 ov%s | 精度: %s | 形状: %s" % (
        target_ov, "fp16" if fp16 else "fp32", "动态" if dynamic else "静态"))
    print("日志: %s" % LOG)
    ok = []
    t0 = time.perf_counter()
    for p in onnx_paths:
        try:
            ok.append(convert_one(p, target_ov, fp16, dynamic))
        except Exception as e:
            import traceback
            print("[FAIL] %s: %s" % (p, e))
            traceback.print_exc()
    print("=" * 74)
    print("完成 %d/%d 个，总耗时 %.1f s" % (len(ok), len(onnx_paths), time.perf_counter() - t0))
    for p in ok:
        print("  %s" % os.path.relpath(p, PROJ))
    return 0 if len(ok) == len(onnx_paths) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
