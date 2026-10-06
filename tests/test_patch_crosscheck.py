"""Actual boundary regressions from the independent 0.1.2 cross-check."""
import pytest
import torch

from prism.settings import validate_sparse


@pytest.mark.parametrize("options", [
    {"bsa_chunk_3d_shape_q": [3, 4, 4]},
    {"bsa_chunk_3d_shape_k": [3, 4, 4]},
    {"bsa_chunk_3d_shape_k": [1, 2, 4]},
    {"bsa_v2a_chunk_3d_shape_k": [1, 1, 1]},
    {"bsa_v2a_audio_chunk_size": 192},
    {"bsa_v2a_audio_chunk_size": 320},
])
def test_unsupported_sparse_chunks_fail_before_kernel_or_weight_loading(options):
    with pytest.raises(ValueError, match="power|16 key tokens"):
        validate_sparse(options)


def test_native_sparse_accepts_small_query_and_minimum_key_blocks():
    options = validate_sparse({"bsa_chunk_3d_shape_q": [1, 1, 1], "bsa_chunk_3d_shape_k": [1, 4, 4]})
    assert options["bsa_chunk_3d_shape_q"] == [1, 1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BSA")
def test_pure_topk_keeps_single_key_block_and_matches_dense_attention():
    pytest.importorskip("triton")
    from prism.native.models.modules.wan_video_dit import SelfAttention
    attention = SelfAttention(dim=128, num_heads=1, enable_bsa=True)
    attention.bsa_params = {"sparsity": .75, "cdf_threshold": None,
                            "chunk_3d_shape_q": [4, 4, 4], "chunk_3d_shape_k": [4, 4, 4]}
    generator = torch.Generator(device="cuda").manual_seed(17)
    q, k, v = [torch.randn(1, 1, 64, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)]
    with torch.inference_mode():
        actual = attention._run_bsa(q, k, v, (4, 4, 4))
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert actual.norm() > 0
    torch.testing.assert_close(actual, expected, atol=.002, rtol=.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for native BSA")
def test_padded_audio_queries_with_unpadded_video_keys_preserve_attention():
    pytest.importorskip("triton")
    from prism.native.models.modules.block_sparse_attention import flash_attn_bsa_cross
    generator = torch.Generator(device="cuda").manual_seed(29)
    q = torch.randn(1, 1, 129, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k, v = [torch.randn(1, 1, 64, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(2)]
    with torch.inference_mode():
        actual = flash_attn_bsa_cross(q, k, v, q_structure="1d", k_structure="3d", k_grid_size=(4, 4, 4),
                                    chunk_size_q=64, sparsity=.75, cdf_threshold=.2)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, expected, atol=.002, rtol=.02)
