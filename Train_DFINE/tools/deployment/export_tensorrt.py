# -*- coding: utf-8 -*-
"""D-FINE ONNX -> TensorRT engine（静态 shape，fp32 默认，可选 fp16）。

产物统一落地到 models/d-fine-tensorrt_trt10-static-fp32/ ，文件名自带 TRT 主版本 + 精度 + 静态：
    dfine_{variant}_{dataset}_trt10_fp32_static.engine
    dfine_{variant}_{dataset}_trt10_fp16_static.engine   (--fp16)
“trt10” 取 tensorrt.__version__ 的主版本号；engine 不可跨 TRT 版本 / 跨 GPU 架构移植，
换环境必须重转。

⚠️ D-FINE fp16 的坑（与 OpenVINO 同理）：
  S/M/L/X 的中间值 fp16 溢出会在 GPU 上输出全垃圾（N 档侥幸正常）。默认 fp32 最稳；
  若要 fp16，务必在推理端也确认结果正常（先用 tools/inference/trt_inf.py 跑一张真图看框）。
  2 核低配 GPU 上 fp16 未必更快，先测速再决定。

依赖：本机需安装 tensorrt（训练 venv 默认不含，需在部署机/带 GPU 的机器上跑）。
      pip install tensorrt   （或随 CUDA/TensorRT SDK 的 whl 安装）

用法（在工程根目录执行）：
    venv\\Scripts\\python.exe tools\\deployment\\export_tensorrt.py \\
        models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx
    venv\\Scripts\\python.exe tools\\deployment\\export_tensorrt.py \\
        models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx --fp16
    venv\\Scripts\\python.exe tools\\deployment\\export_tensorrt.py \\
        models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx --fp16 --workspace 4

也可以用 NVIDIA 官方命令行工具 trtexec（无需 Python/tensorrt 包）：
    trtexec --onnx=models/d-fine-onnx-dyn-opset17/dfine_n_coco_opset17.onnx \\
        --saveEngine=models/d-fine-tensorrt_trt10-static-fp32/dfine_n_coco_trt10_fp32_static.engine \\
        --explicitBatch --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \\
        --optShapes=images:1x3x640x640,orig_target_sizes:1x2 \\
        --maxShapes=images:1x3x640x640,orig_target_sizes:1x2
  fp16 加 --fp16 ；动态 batch 把三个 shapes 的 1x 改成 1x/1x/4x 等。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
MODELS_DIR = os.path.join(PROJ, "models")
TRT_DIR = os.path.join(MODELS_DIR, "d-fine-tensorrt_trt10-static-fp32")
SIZE = 640


def build_engine(onnx_path, fp16, workspace_gb, batch):
    import numpy as np
    import tensorrt as trt

    assert os.path.isfile(onnx_path), "缺少 ONNX: %s（先跑 tools\\deployment\\export_onnx.py）" % onnx_path
    trt_ver = trt.__version__.split(".")[0]  # 主版本号，如 '10'
    stem = os.path.splitext(os.path.basename(onnx_path))[0]  # dfine_n_coco_opset17
    suffix = "fp16" if fp16 else "fp32"
    engine_name = "%s_trt%s_%s_static.engine" % (stem, trt_ver, suffix)
    engine_path = os.path.join(TRT_DIR, engine_name)

    os.makedirs(TRT_DIR, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
            raise RuntimeError("ONNX 解析失败: %s" % "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 静态 shape：images [batch,3,640,640]，orig_target_sizes [batch,2]
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        if "images" in name:
            profile.set_shape(name, (batch, 3, SIZE, SIZE), (batch, 3, SIZE, SIZE), (batch, 3, SIZE, SIZE))
        else:  # orig_target_sizes
            profile.set_shape(name, (batch, 2), (batch, 2), (batch, 2))
    config.add_optimization_profile(profile)

    print("构建 TRT engine: %s (trt%s, %s, batch=%d, workspace=%dGB)" % (
        engine_name, trt_ver, suffix, batch, workspace_gb))
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("engine 构建失败（workspace 不足或算子不支持，尝试调大 --workspace）")

    tmp = engine_path + ".new%d" % os.getpid()
    with open(tmp, "wb") as f:
        f.write(engine)
    if os.path.exists(engine_path):
        try:
            os.remove(engine_path)
        except OSError:
            pass
    os.replace(tmp, engine_path)
    mb = os.path.getsize(engine_path) / 1048576.0
    print("engine 构建完成 %s (%.1f MB)" % (engine_name, mb))
    return engine_path


def main(argv):
    onnx_paths = [a for a in argv if a.endswith(".onnx")]
    fp16 = "--fp16" in argv
    workspace_gb = 4
    batch = 1
    if "--workspace" in argv:
        workspace_gb = int(argv[argv.index("--workspace") + 1])
    if "--batch" in argv:
        batch = int(argv[argv.index("--batch") + 1])
    print("D-FINE ONNX -> TensorRT engine | 精度: %s | batch=%d | workspace=%dGB" % (suffix(fp16), batch, workspace_gb))
    ok = []
    for p in onnx_paths:
        try:
            ok.append(build_engine(p, fp16, workspace_gb, batch))
        except Exception as e:
            import traceback
            print("[FAIL] %s: %s" % (p, e))
            traceback.print_exc()
    print("完成 %d/%d 个" % (len(ok), len(onnx_paths)))
    for p in ok:
        print("  %s" % os.path.relpath(p, PROJ))
    return 0 if len(ok) == len(onnx_paths) else 1


def suffix(fp16):
    return "fp16" if fp16 else "fp32"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
