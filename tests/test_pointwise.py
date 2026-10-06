"""Native compiled rounding and floating dtype/broadcast compatibility."""
import pytest
import torch

from prism.native.models.modules.wan_video_dit import modulate
from prism.native.models.modules.interactionv2 import apply_rotary_pos_emb, rotate_half


def original_modulate(x, shift, scale):
    # Original pinned native formula, compiled by upstream.
    return x * (1 + scale) + shift


def original_rotary(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


@pytest.mark.parametrize("dtype,x,scale,expected", [
    (torch.bfloat16, 1.5, 1 / 256, 1.5078125),
    (torch.float16, 1.5, 1 / 2048, 1.5009765625),
])
def test_modulate_rounds_once_at_low_precision_output(dtype, x, scale, expected):
    x, scale = torch.tensor(x, dtype=dtype), torch.tensor(scale, dtype=dtype)
    shift = torch.zeros((), dtype=dtype)
    result = modulate(x, shift, scale)
    assert result.dtype == dtype and result.item() == expected
    # A rounded 1+scale intermediate discards a representable contribution.
    assert original_modulate(x, shift, scale).item() != expected


def test_rotary_rounds_once_at_low_precision_output():
    q = torch.tensor([[[[1.5, .5]]]], dtype=torch.bfloat16)
    cos = sin = torch.tensor([[[.70703125, .70703125]]], dtype=torch.bfloat16)
    output, second = apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=2)
    expected = torch.tensor([[[[.70703125, 1.4140625]]]], dtype=torch.bfloat16)
    assert torch.equal(output, expected) and torch.equal(second, expected)
    assert not torch.equal(original_rotary(q, q, cos, sin, unsqueeze_dim=2)[0], expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_modulate_retains_full_precision_and_broadcast(dtype):
    torch.manual_seed(79)
    x = torch.randn(2, 3, 8, dtype=dtype)
    scale = torch.randn(1, 1, 8, dtype=dtype)
    shift = torch.randn(2, 1, 8, dtype=dtype)
    assert torch.equal(modulate(x, shift, scale), original_modulate(x, shift, scale))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotary_retains_full_precision_and_broadcast(dtype):
    torch.manual_seed(19)
    q, k = torch.randn(2, 3, 4, 8, dtype=dtype), torch.randn(2, 3, 4, 8, dtype=dtype)
    cos, sin = torch.randn(1, 3, 8, dtype=dtype), torch.randn(1, 3, 8, dtype=dtype)
    actual, expected = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2), original_rotary(q, k, cos, sin, unsqueeze_dim=2)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_mixed_floating_dtype_matches_original_output_dtype():
    x = torch.ones(2, 3, 8, dtype=torch.bfloat16)
    scale = torch.ones(1, 1, 8, dtype=torch.float64)
    shift = torch.ones(2, 1, 8, dtype=torch.float32)
    actual = modulate(x, shift, scale)
    assert actual.dtype == original_modulate(x, shift, scale).dtype == torch.float64
    q, k = torch.ones(2, 3, 4, 8).bfloat16(), torch.ones(2, 3, 4, 8).double()
    cos, sin = torch.ones(1, 3, 8), torch.ones(1, 3, 8).bfloat16()
    actual = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
    expected = original_rotary(q, k, cos, sin, unsqueeze_dim=2)
    assert [t.dtype for t in actual] == [t.dtype for t in expected] == [torch.float32, torch.float64]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiler comparison requires a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_original_gpu_compiled_pointwise(dtype):
    pytest.importorskip("triton")
    torch.manual_seed(41)
    with torch.inference_mode():
        x = torch.randn(1, 64, 256, dtype=dtype, device="cuda")
        scale, shift = torch.randn(1, 1, 256, dtype=dtype, device="cuda"), torch.randn(1, 1, 256, dtype=dtype, device="cuda")
        expected = torch.compile(original_modulate, fullgraph=True)(x, shift, scale)
        assert torch.equal(modulate(x, shift, scale), expected)
        q, k = torch.randn(1, 64, 4, 128, dtype=dtype, device="cuda"), torch.randn(1, 64, 4, 128, dtype=dtype, device="cuda")
        angles = torch.randn(1, 64, 128, device="cuda")
        cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
        expected = torch.compile(original_rotary, fullgraph=True)(q, k, cos, sin, unsqueeze_dim=2)
        actual = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
