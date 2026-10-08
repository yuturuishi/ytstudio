# -*- coding: utf-8 -*-
"""D-FINE ONNX -> OpenVINO IR（xml + bin），并做一次编译冒烟。

产物：weights/openvino/dfine_<档>_fp32.xml + .bin（--fp16 则为 *_fp16）
依赖：weights/onnx/dfine_<档>.onnx（由 tests/export2onnx.py 生成）
日志：tests/logs/export2openvino_<时间戳>_<pid>.log

OpenVINO API 要点：
  - convert_model 产物仍是动态 batch；必须 core.read_model() 后 reshape 成静态
    [1,3,640,640] 再 compile_model —— CPU 上静态形状更快；
  - 不能对 CompiledModel 调 reshape（会报 'CompiledModel' object has no attribute 'reshape'）；
  - 输入名会带自动名（如 /postprocessor/Expand_output_0），按名字里有没有 "images" 认；
  - CPU 设备属性用 FULL_DEVICE_NAME（CPU_MODEL_FULLNAME 不支持）。

用法（在工程根目录执行）：
    venv\\Scripts\\python.exe tests\\export2openvino.py        # 全部五档
    venv\\Scripts\\python.exe tests\\export2openvino.py s m    # 指定档
    venv\\Scripts\\python.exe tests\\export2openvino.py --fp16
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ONNX_DIR = os.path.join(PROJ, "weights", "onnx")
OV_DIR = os.path.join(PROJ, "weights", "openvino")
VARIANTS = ["n", "s", "m", "l", "x"]
SIZE = 640


def setup_log():
    """stdout/stderr 镜像到 UTF-8 日志；名字带 pid，避免同秒撞名被沙箱拦。"""
    import io

    d = os.path.join(HERE, "logs")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(
        d, "export2openvino_%s_%d.log" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid())
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


def atomic_replace(src, dst):
    """D 盘沙箱可能禁止直接覆盖已存在文件：先删旧的（失败不阻塞），再改名替换。"""
    if os.path.exists(dst):
        try:
            os.remove(dst)
        except OSError as e:
            print("  (旧文件删除失败，交由 os.replace 尝试: %s)" % e)
    os.replace(src, dst)


def convert_one(variant, fp16=False):
    import numpy as np
    import openvino as ov

    onnx_path = os.path.join(ONNX_DIR, "dfine_%s.onnx" % variant)
    assert os.path.isfile(onnx_path), "缺少 ONNX: %s（先跑 tests\\export2onnx.py）" % onnx_path
    suffix = "fp16" if fp16 else "fp32"
    xml_path = os.path.join(OV_DIR, "dfine_%s_%s.xml" % (variant, suffix))

    print("=" * 74)
    print("[%s] ONNX %.1f MB -> IR(%s)" % (variant, os.path.getsize(onnx_path) / 1048576.0, suffix))
    t0 = time.perf_counter()
    model = ov.convert_model(onnx_path)
    t_conv = time.perf_counter() - t0

    ins = [(p.get_any_name() or p.get_names(), list(p.partial_shape), str(p.get_element_type())) for p in model.inputs]
    outs = [(p.get_any_name() or p.get_names(), list(p.partial_shape), str(p.get_element_type())) for p in model.outputs]
    print("[%s] 输入 %s" % (variant, ins))
    print("[%s] 输出 %s" % (variant, outs))

    os.makedirs(OV_DIR, exist_ok=True)
    # 临时文件名必须仍以 .xml 结尾（ov.save_model 校验扩展名），后缀插在扩展名前
    stem = os.path.splitext(xml_path)[0]
    tmp_xml = "%s.new%d.xml" % (stem, os.getpid())
    ov.save_model(model, tmp_xml, compress_to_fp16=fp16)
    atomic_replace(tmp_xml, xml_path)
    bin_path = xml_path.replace(".xml", ".bin")
    atomic_replace(tmp_xml.replace(".xml", ".bin"), bin_path)
    print("[%s] 转换 %.1f s -> %s (%.1f MB) + .bin (%.1f MB)" % (
        variant, t_conv, os.path.basename(xml_path),
        os.path.getsize(xml_path) / 1048576.0, os.path.getsize(bin_path) / 1048576.0))

    # 编译 + 冒烟（把动态 batch 固定成 1，CPU 上静态形状更快）
    core = ov.Core()
    m2 = core.read_model(xml_path)
    shape = {i.get_any_name(): [1, 3, SIZE, SIZE] if "images" in i.get_any_name() else [1, 2] for i in m2.inputs}
    try:
        m2.reshape(shape)
        print("[%s] reshape -> 静态 %s" % (variant, shape))
    except Exception as e:  # noqa: BLE001
        print("[%s] reshape 跳过: %s" % (variant, e))
    t0 = time.perf_counter()
    cm = core.compile_model(m2, "CPU")
    t_compile = time.perf_counter() - t0
    in_names = [i.get_any_name() for i in cm.inputs]
    req = cm.create_infer_request()
    x = np.zeros((1, 3, SIZE, SIZE), np.float32)
    s = np.array([[SIZE, SIZE]], np.int64)
    t0 = time.perf_counter()
    res = req.infer({in_names[0]: x, in_names[1]: s})
    t_first = time.perf_counter() - t0
    print("[%s] 编译 %.1f s；首次推理 %.0f ms；输出 %s" % (
        variant, t_compile, t_first * 1000, [np.asarray(v).shape for v in res.values()]))
    return xml_path


def main(argv):
    variants = [a for a in argv if a in VARIANTS] or VARIANTS
    fp16 = "--fp16" in argv
    print("D-FINE ONNX -> OpenVINO IR | 档位: %s | 精度: %s" % (variants, "fp16" if fp16 else "fp32"))
    print("日志: %s" % LOG)
    ok = []
    t0 = time.perf_counter()
    for v in variants:
        try:
            ok.append(convert_one(v, fp16))
        except Exception as e:  # noqa: BLE001
            import traceback

            print("[%s] 转换失败: %s" % (v, e))
            traceback.print_exc()
    print("=" * 74)
    print("完成 %d/%d 档，总耗时 %.1f s" % (len(ok), len(variants), time.perf_counter() - t0))
    return 0 if len(ok) == len(variants) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
