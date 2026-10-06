import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prism.format import Component, StreamingWriter, TensorReader
from prism.conversion import split_fused_key, eligible_linear
from prism.quantization import ConvRotLinear, decode_config, hadamard, quantize, rotate
from prism.settings import validate_generation, validate_sparse


@pytest.mark.parametrize("copy", [False, True])
def test_tensor_reader_dtype_content_and_lifetime(tmp_path, copy):
    import gc
    source = {"bf16": torch.randn(3, 7).bfloat16(), "int8": torch.arange(8, dtype=torch.int8),
              "float": torch.tensor([.5, 1.]), "empty": torch.empty(0, dtype=torch.int16)}
    path = tmp_path / "reader.safetensors"
    save_file(source, path, metadata={"test": "readers"})
    with TensorReader(path, copy=copy) as reader:
        assert reader.metadata() == {"test": "readers"}
        result = {key: reader.get_tensor(key) for key in reader.keys()}
    del reader
    gc.collect()
    for key, tensor in result.items():
        assert torch.equal(tensor, source[key])
        assert tensor.dtype == source[key].dtype
    if copy:
        result["float"].fill_(9.)
        with TensorReader(path) as reader:
            assert torch.equal(reader.get_tensor("float"), source["float"])


@pytest.mark.parametrize("size", [16, 64, 256])
def test_regular_convrot_matches_comfy_org(size):
    reference = pytest.importorskip("comfy_kitchen.tensor.int8_utils")
    h = hadamard(size)
    assert torch.equal(h, reference._build_hadamard(size, device="cpu", dtype=torch.float32))
    assert torch.allclose(h @ h.T, torch.eye(size))
    weight = torch.randn(5, size * 2)
    assert torch.allclose(rotate(rotate(weight, size), size), weight, atol=2e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quantized_linear_preserves_function(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(17)
    weight, bias, x = torch.randn(48, 64), torch.randn(48), torch.randn(2, 7, 64)
    q, scale, config = quantize(weight, device=device)
    layer = ConvRotLinear(q, scale, decode_config(config), bias).to(device)
    result = layer(x.to(device)).cpu()
    reference = torch.nn.functional.linear(x, weight, bias)
    relative_error = (result - reference).norm() / reference.norm()
    assert relative_error < 0.015
    # Direct casting of the rotated int8 weight would give the wrong function.
    incorrect = torch.nn.functional.linear(x, q.float() * scale, bias)
    assert (incorrect - reference).norm() / reference.norm() > 0.5
    layer.bfloat16()
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.dtype == torch.float32


def test_mseclip_not_worse_than_absmax():
    torch.manual_seed(2)
    weight = torch.randn(32, 64)
    q, scale, _ = quantize(weight)
    clipped, clip_scale, _ = quantize(weight, mseclip=True)
    rotated = rotate(weight, 64)
    assert (clipped.float() * clip_scale - rotated).square().sum() <= (q.float() * scale - rotated).square().sum() + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kitchen_convrot_gpu_function():
    pytest.importorskip("comfy_kitchen")
    torch.manual_seed(9)
    weight, x = torch.randn(48, 64), torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    q, scale, config = quantize(weight)
    layer = ConvRotLinear(q, scale, decode_config(config), backend="kitchen").cuda()
    result = layer(x)
    reference = torch.nn.functional.linear(x, weight.to(device="cuda", dtype=x.dtype))
    assert torch.isfinite(result).all()
    assert (result.float() - reference.float()).norm() / reference.float().norm() < 0.025


def test_writer_roundtrip_and_no_clobber(tmp_path):
    path = tmp_path / "component.safetensors"
    meta = {"prism.format_version": "1", "prism.component": "video_dit", "prism.config": "{}"}
    tensors = {"bf16": torch.randn(3, 7).bfloat16(), "scalar": torch.tensor(2.), "i8": torch.tensor([-127, 0, 127], dtype=torch.int8)}
    with StreamingWriter(path, meta) as writer:
        for key, tensor in tensors.items():
            writer.add(key, tensor)
        writer.finish()
    with safe_open(str(path), framework="pt") as reader:
        for key, tensor in tensors.items():
            assert torch.equal(reader.get_tensor(key), tensor)
    assert Component.inspect(path, "video_dit").config == {}
    with pytest.raises(FileExistsError):
        StreamingWriter(path, meta)
    with pytest.raises(ValueError, match="expected audio_dit"):
        Component.inspect(path, "audio_dit")


def test_atomic_writer_aborts(tmp_path):
    path = tmp_path / "aborted.safetensors"
    with pytest.raises(ValueError):
        with StreamingWriter(path, {}) as writer:
            writer.add("x", torch.ones(2))
            writer.add("x", torch.ones(2))
    assert not path.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source,kind,target", [
    ("fusion_blocks.7.video_block.self_attn.q.weight", "video_dit", "blocks.7.self_attn.q.weight"),
    ("fusion_blocks.9.audio_block.ffn.2.weight", "audio_dit", "blocks.9.ffn.2.weight"),
    ("fusion_blocks.5.a2v_conditioner.inner.q.weight", "dual_tower_bridge", "audio_to_video_conditioners.5.inner.q.weight"),
    ("fusion_blocks.5.v2a_conditioner.inner.q.weight", "dual_tower_bridge", "video_to_audio_conditioners.5.inner.q.weight"),
    ("remaining_video_blocks.9.ffn.0.weight", "video_dit", "blocks.39.ffn.0.weight"),
    ("video_dit_2.blocks.31.ffn.2.weight", "video_dit_2", "blocks.31.ffn.2.weight"),
    ("audio_dit.time_embedding.0.weight", "audio_dit", "time_embedding.0.weight"),
])
def test_fused_mapping(source, kind, target):
    assert split_fused_key(source, 30) == (kind, target)


def test_only_real_block_linears_quantized():
    assert eligible_linear("video_dit", "blocks.0.self_attn.q.weight", [48, 64])
    assert eligible_linear("text_encoder", "encoder.block.0.layer.1.DenseReluDense.wi_0.weight", [256, 64])
    assert not eligible_linear("text_encoder", "shared.weight", [256, 64])
    assert not eligible_linear("text_encoder", "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", [32, 64])
    assert not eligible_linear("video_dit", "time_embedding.0.weight", [48, 64])
    assert not eligible_linear("video_vae", "decoder.weight", [48, 64])
    with pytest.raises(ValueError):
        split_fused_key("unknown.weight", 30)


def test_generation_validation_and_snap():
    assert validate_generation({"num_frames": 206, "frame_policy": "snap"})["num_frames"] == 205
    for options in ({"height": 1080}, {"num_frames": 206}, {"fps": 0}, {"steps": 0}, {"cfg": float("nan")}):
        with pytest.raises(ValueError):
            validate_generation(options)


def test_sparse_rejects_conflicts_and_typos():
    for options in ({"enable_bsa": "false"}, {"enable_bsaa": True},
                    {"enable_bsa": True, "enable_ivpq_dynamic_block": True, "enable_variance_guidance": True},
                    {"enable_layer_adaptive_dynamic_block": True}, {"bsa_sparsity": 1.0},
                    {"bsa_chunk_3d_shape_q": [0, 4, 4]}, {"bsa_v2a_audio_chunk_size": 5}):
        with pytest.raises(ValueError):
            validate_sparse(options)
    assert validate_sparse({"enable_bsa": True, "enable_ivpq_dynamic_block": True})["enable_ivpq_dynamic_block"]


def test_corrupt_quant_scale_bias_and_bare_integer_rejected():
    q, scale, config = quantize(torch.randn(32, 64))
    for bad_scale, bias in ((scale.half(), None), (scale, torch.zeros(1)), (scale, torch.full((32,), float("nan")))):
        with pytest.raises(ValueError):
            ConvRotLinear(q, bad_scale, decode_config(config), bias=bias)
    from prism.loading import set_tensor
    model = torch.nn.Linear(64, 32)
    with pytest.raises(ValueError, match="Unmarked integer"):
        set_tensor(model, "weight", torch.zeros(32, 64, dtype=torch.uint8), torch.bfloat16)
