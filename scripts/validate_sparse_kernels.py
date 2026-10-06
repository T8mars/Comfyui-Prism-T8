"""Exercise native sparse GPU kernels on synthetic tensors, not generation quality."""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch


def main():
    import triton
    from prism.native.models.modules.block_sparse_attention import (
        flash_attn_bsa_3d, flash_attn_bsa_cross, flash_attn_bsa_3d_audio_guided,
        flash_attn_bsa_3d_variance_guided, bsa_taylor_sparse_attn, bsa_rectified_sparse_attn,
    )
    from prism.native.models.modules.block_sparse_attention.dynamic_block_attention import flash_attn_bsa_3d_dynamic
    torch.set_num_threads(4)
    torch.manual_seed(42)
    grid = (8, 8, 16)
    q, k, v = [torch.randn(1, 2, 1024, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    norms = torch.rand(1024, device="cuda")
    blocking = dict(chunk_3d_shape_q=[4, 4, 4], chunk_3d_shape_k=[4, 4, 4])
    cases = {
        "self_all_blocks": lambda: flash_attn_bsa_3d(q, k, v, grid, grid, sparsity=0., **blocking),
        "self_top_p": lambda: flash_attn_bsa_3d(q, k, v, grid, grid, sparsity=None, cdf_threshold=.75, **blocking),
        "audio_guided": lambda: flash_attn_bsa_3d_audio_guided(q, k, v, grid, grid, audio_token_norms=norms,
            audio_gate_value=torch.tensor(.5, device="cuda"), audio_boost_gamma=torch.tensor(1., device="cuda"),
            audio_weighted_lambda=.5, sparsity=.75, **blocking),
        "variance_guided": lambda: flash_attn_bsa_3d_variance_guided(q, k, v, grid, grid, sparsity=.75, **blocking),
        "video_to_audio": lambda: flash_attn_bsa_cross(q[:, :, :129].contiguous(), k, v,
            q_structure="1d", k_structure="3d", k_grid_size=grid, sparsity=.75),
        "audio_to_video": lambda: flash_attn_bsa_cross(q, k[:, :, :129].contiguous(), v[:, :, :129].contiguous(),
            q_structure="3d", k_structure="1d", q_grid_size=grid, sparsity=.75),
        "taylor": lambda: bsa_taylor_sparse_attn(q, k, v, 64, 64, .75, None, 128 ** -.5),
        "rectified": lambda: bsa_rectified_sparse_attn(q, k, v, 64, 64, .75, None, 128 ** -.5),
        "ivpq": lambda: flash_attn_bsa_3d_dynamic(q, k, v, grid, method="ivpq", audio_token_norms=norms, sparsity=.75),
        "penalty": lambda: flash_attn_bsa_3d_dynamic(q, k, v, grid, method="penalty", audio_token_norms=norms, sparsity=.75),
    }
    report = {"scope": "native kernels on synthetic BF16 tensors, not full pipeline or quality acceptance",
              "torch": torch.__version__, "triton": triton.__version__,
              "gpu": torch.cuda.get_device_name(), "cases": {}}
    output = Path("outputs/sparse-kernel-validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        for name, execute in cases.items():
            start = time.monotonic()
            messages = io.StringIO()
            with redirect_stdout(messages):
                result = execute()
            torch.cuda.synchronize()
            if not torch.isfinite(result).all():
                raise RuntimeError(f"{name} returned non-finite values")
            entry = {"shape": list(result.shape), "finite": True, "seconds": time.monotonic() - start}
            if messages.getvalue().strip():
                entry["native_messages"] = messages.getvalue().strip()
            if name == "self_all_blocks":
                entry["dense_max_error"] = float((result - dense).abs().max())
                if entry["dense_max_error"] > .02:
                    raise RuntimeError(f"Native all-block attention disagrees with SDPA: {entry}")
            report["cases"][name] = entry
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(name, entry, flush=True)
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
