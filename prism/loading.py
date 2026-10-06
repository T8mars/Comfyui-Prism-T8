"""Construct native modules from embedded config, load one tensor at a time."""
from __future__ import annotations

import torch

from .format import Component, TensorReader
from .quantization import ConvRotLinear, decode_config


def make_module(component):
    kind, config = component.kind, component.config
    if kind in ("video_dit", "video_dit_2"):
        from .native.models.modules.wan_video_dit import WanModel
        return WanModel.from_config(config)
    if kind == "audio_dit":
        from .native.models.modules.wan_audio_dit import WanAudioModel
        return WanAudioModel.from_config(config)
    if kind == "dual_tower_bridge":
        from .native.models.modules.interactionv2 import DualTowerConditionalBridge
        return DualTowerConditionalBridge.from_config(config)
    if kind == "video_vae":
        from diffusers import AutoencoderKLWan
        return AutoencoderKLWan.from_config(config)
    if kind == "audio_vae":
        from .native.models.modules.dac_vae import DAC
        return DAC.from_config(config)
    if kind == "text_encoder":
        from transformers import UMT5Config, UMT5EncoderModel
        return UMT5EncoderModel(UMT5Config.from_dict(config))
    raise ValueError(f"Unknown component: {kind}")


def set_tensor(model, key, tensor, dtype):
    parent, _, name = key.rpartition(".")
    module = model.get_submodule(parent) if parent else model
    parameter = name in module._parameters
    current = module._parameters.get(name) if parameter else module._buffers.get(name)
    if current is None:
        raise ValueError(f"Unexpected checkpoint tensor {key}")
    if current.shape != tensor.shape:
        raise ValueError(f"Shape mismatch for {key}: expected {tuple(current.shape)}, got {tuple(tensor.shape)}")
    if current.is_floating_point() and not tensor.is_floating_point():
        raise ValueError(f"Unmarked integer tensor for floating point parameter {key}")
    if tensor.is_floating_point():
        # Keep the official FP32 timestep path explicit across autocast versions.
        target_dtype = torch.float32 if key.startswith(("time_embedding.", "time_projection.")) else dtype
        if tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite component tensor after dtype conversion: {key}")
    if parameter:
        module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=False)
    else:
        module._buffers[name] = tensor


def materialize_nonpersistent(model, kind):
    if kind in ("video_dit", "video_dit_2"):
        from .native.models.modules.wan_video_dit import precompute_freqs_cis_3d
        model.freqs = precompute_freqs_cis_3d(model.dim // model.config.num_heads)
    elif kind == "audio_dit":
        from .native.models.modules.wan_audio_dit import precompute_freqs_cis_1d, legacy_precompute_freqs_cis_1d
        head = model.dim // model.config.num_heads
        model.freqs = precompute_freqs_cis_1d(head) if model.vae_type == "dac" else legacy_precompute_freqs_cis_1d(head)
    elif kind == "dual_tower_bridge":
        rotary = model.rotary
        inv = 1.0 / (rotary.base ** (torch.arange(0, rotary.dim, 2).float() / rotary.dim))
        rotary.inv_freq = inv
        rotary.original_inv_freq = inv
    if hasattr(model, "tie_weights"):
        model.tie_weights()


def load_component(component, dtype=torch.bfloat16, backend="portable", interrupt=None):
    if not isinstance(component, Component):
        component = Component.inspect(component)
    # No large randomly initialized copy. Native configs, no from_pretrained loader.
    with torch.device("meta"):
        model = make_module(component)
    with TensorReader(component.path) as reader:
        keys = set(reader.keys())
        quant_keys = sorted(key for key in keys if key.endswith(".comfy_quant"))
        consumed = set()
        for config_key in quant_keys:
            if interrupt:
                interrupt()
            prefix = config_key[:-len(".comfy_quant")]
            original = model.get_submodule(prefix)
            if not isinstance(original, torch.nn.Linear):
                raise ValueError(f"Quantized tensor {prefix} is not a native Linear")
            config = decode_config(reader.get_tensor(config_key))
            weight_key, scale_key, bias_key = prefix + ".weight", prefix + ".weight_scale", prefix + ".bias"
            if weight_key not in keys or scale_key not in keys:
                raise ValueError(f"Incomplete ConvRot metadata for {prefix}")
            bias = reader.get_tensor(bias_key) if bias_key in keys else None
            if bias is not None:
                if not bias.is_floating_point():
                    raise ValueError(f"Unmarked integer bias for quantized layer {prefix}")
                bias = bias.to(dtype)
            if (original.bias is not None) != (bias is not None):
                raise ValueError(f"Bias mismatch for quantized layer {prefix}")
            layer = ConvRotLinear(reader.get_tensor(weight_key), reader.get_tensor(scale_key), config, bias, backend)
            if layer.in_features != original.in_features or layer.out_features != original.out_features:
                raise ValueError(f"Quantized Linear shape mismatch: {prefix}")
            parent, _, name = prefix.rpartition(".")
            (model.get_submodule(parent) if parent else model)._modules[name] = layer
            consumed.update((config_key, weight_key, scale_key))
            if bias is not None:
                consumed.add(bias_key)
        for key in sorted(keys - consumed):
            if interrupt:
                interrupt()
            tensor = reader.get_tensor(key)
            if tensor.dtype == torch.int8:
                raise ValueError(f"Unmarked INT8 weight: {key}; refusing incorrect integer-to-float loading")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite component tensor: {key}")
            set_tensor(model, key, tensor, dtype)
    materialize_nonpersistent(model, component.kind)
    missing = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    missing += [name for name, buffer in model.named_buffers() if buffer.is_meta]
    if missing:
        raise ValueError(f"Incomplete {component.kind} checkpoint: {missing[:12]}")
    return model.eval().requires_grad_(False)
