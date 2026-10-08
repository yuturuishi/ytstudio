"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

# NOTE(本地补丁 2026-10-04): calflops 仅用于统计 FLOPs，纯做 ONNX 导出/转换时并不需要。
# 它的导入链会牵出 transformers 等一大串训练侧依赖，缺失时不应阻塞部署脚本。
try:
    from calflops import calculate_flops
except ModuleNotFoundError:  # pragma: no cover
    calculate_flops = None


def stats(
    cfg,
    input_shape: Tuple = (1, 3, 640, 640),
) -> Tuple[int, dict]:
    if calculate_flops is None:
        raise RuntimeError(
            "calflops 未安装，无法统计 FLOPs（仅部署/导出场景可忽略此功能）"
        )
    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()

    flops, macs, _ = calculate_flops(
        model=model_for_info,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4,
        print_detailed=False,
    )
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" % (flops, macs, params)}
