"""Comfy-Org INT8 ConvRot wire format; W8A16 portable and W8A8 kitchen execution."""
from __future__ import annotations

from functools import lru_cache
import json

import torch
from torch import nn
from torch.nn import functional as F


def best_group_size(features):
    return next((size for size in (256, 64, 16) if features % size == 0), None)


@lru_cache(maxsize=32)
def hadamard(size, device="cpu"):
    # Regular Hadamard used by Comfy-Org, not the Sylvester sign convention.
    if size not in (16, 64, 256):
        raise ValueError(f"Unsupported ConvRot group size: {size}")
    base = torch.ones(4, 4, device=device, dtype=torch.float32)
    base[torch.arange(4, device=device), torch.arange(3, -1, -1, device=device)] = -1
    result = base
    while result.shape[0] < size:
        result = torch.kron(result, base)
    return result / size ** 0.5


def rotate(tensor, group_size, fp32=True):
    if tensor.shape[-1] % group_size:
        raise ValueError("ConvRot group size does not divide input features")
    shape = tensor.shape
    source = tensor.float() if fp32 else tensor
    matrix = hadamard(group_size, str(tensor.device)).to(source.dtype)
    rotated = source.reshape(-1, group_size) @ matrix
    return rotated.reshape(shape)


@torch.no_grad()
def quantize(weight, group_size=None, mseclip=False, device="cpu", rows=128):
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("INT8 ConvRot quantization requires a floating point Linear matrix")
    if type(rows) is not int or rows < 1:
        raise ValueError("INT8 ConvRot row chunk size must be a positive integer")
    group_size = group_size or best_group_size(weight.shape[1])
    if group_size is None:
        raise ValueError("No eligible ConvRot group size")
    # Bound FP32 intermediates even for UMT5's largest matrices.
    result = torch.empty(weight.shape, dtype=torch.int8)
    scales = torch.empty((weight.shape[0], 1), dtype=torch.float32)
    grid = torch.linspace(0.55, 1.0, 80).tolist() if mseclip else [1.0]
    for start in range(0, weight.shape[0], rows):
        rotated = rotate(weight[start:start + rows].to(device), group_size)
        if not torch.isfinite(rotated).all():
            raise ValueError("Non-finite ConvRot weights after FP32 conversion/rotation")
        maxima = rotated.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        best_error = torch.full_like(maxima, float("inf"))
        best_scale, best_q = None, None
        for ratio in grid:
            scale = (maxima * ratio / 127).clamp_min(1e-30)
            q = (rotated / scale).round().clamp(-127, 127)
            error = (q * scale - rotated).square().mean(dim=1, keepdim=True)
            better = error < best_error
            best_error = torch.minimum(best_error, error)
            best_scale = scale if best_scale is None else torch.where(better, scale, best_scale)
            best_q = q if best_q is None else torch.where(better, q, best_q)
        result[start:start + rows] = best_q.to(device="cpu", dtype=torch.int8)
        scales[start:start + rows] = best_scale.cpu()
    config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}
    return result, scales, torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


def decode_config(tensor):
    if tensor.dtype != torch.uint8 or tensor.ndim != 1 or tensor.numel() > 4096:
        raise ValueError("Invalid comfy_quant JSON tensor")
    config = json.loads(bytes(tensor.cpu().tolist()).decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("comfy_quant JSON must be an object")
    if config.get("format") != "int8_tensorwise" or config.get("convrot") is not True:
        raise ValueError(f"Expected INT8 ConvRot, got {config}")
    if config.get("convrot_groupsize") not in (16, 64, 256):
        raise ValueError("Unsupported ConvRot group size")
    return config


class ConvRotLinear(nn.Module):
    def __init__(self, weight, scale, config, bias=None, backend="portable"):
        super().__init__()
        size = config["convrot_groupsize"]
        if weight.ndim != 2 or weight.dtype != torch.int8 or weight.shape[1] % size:
            raise ValueError("Invalid INT8 ConvRot weight shape/dtype")
        if scale.shape not in (torch.Size([]), torch.Size([1]), torch.Size([weight.shape[0], 1])):
            raise ValueError("Invalid INT8 ConvRot scale shape")
        if scale.dtype != torch.float32:
            raise ValueError("INT8 ConvRot scales must be stored in FP32")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("ConvRot scales must be finite and positive")
        if backend not in ("portable", "kitchen"):
            raise ValueError(f"Unknown INT8 backend: {backend}")
        if bias is not None and (bias.shape != (weight.shape[0],) or not bias.is_floating_point() or not torch.isfinite(bias).all()):
            raise ValueError("Invalid ConvRot bias")
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.group_size = size
        self.backend = backend
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scale.float())
        self.register_parameter("bias", nn.Parameter(bias, requires_grad=False) if bias is not None else None)

    def _apply(self, fn, recurse=True):
        # Module.bfloat16()/to(dtype=...) must never truncate the stored FP32 scales.
        scale = self.weight_scale
        super()._apply(fn, recurse)
        self.weight_scale = scale.to(device=self.weight.device, dtype=torch.float32)
        return self

    def forward(self, x):
        if self.backend == "kitchen":
            import comfy_kitchen
            return comfy_kitchen.int8_linear(x, self.weight, self.weight_scale,
                bias=self.bias, out_dtype=x.dtype, convrot=True, convrot_groupsize=self.group_size)
        # W8A16 reference: rotate ACTIVATIONS, preserving the trained function.
        # Dequantized rotated weight is temporary per Linear, not a BF16 model copy.
        rotated = rotate(x, self.group_size, fp32=False)
        weight = (self.weight.float() * self.weight_scale).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(rotated, weight, bias)
