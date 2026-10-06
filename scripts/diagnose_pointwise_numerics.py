"""Small native pointwise eager/compile comparison, never a full model quality assertion."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from prism.format import Component
from prism.loading import make_module, materialize_nonpersistent
from prism.native.models.modules.wan_video_dit import modulate
from prism.native.models.modules.interactionv2 import apply_rotary_pos_emb, rotate_half


def stats(a, b):
    diff = a.float() - b.float()
    return {"different": int((a != b).sum()), "total": a.numel(),
            "relative_l2": float(diff.norm() / b.float().norm()), "max_abs": float(diff.abs().max())}


def original_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def original_rotary(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def main():
    torch.set_num_threads(4)
    torch.manual_seed(31)
    torch.cuda.set_per_process_memory_fraction(0.04)
    torch.cuda.reset_peak_memory_stats()
    device = "cuda"
    report = {"scope": "synthetic BF16 values with actual native shape/dtype and rotary positions; not a full pipeline root-cause proof",
              "torch": torch.__version__, "cases": {}}
    with torch.inference_mode():
        x = torch.randn(1, 3120, 5120, device=device, dtype=torch.bfloat16)
        scale = torch.randn(1, 1, 5120, device=device, dtype=torch.bfloat16)
        shift = torch.randn(1, 1, 5120, device=device, dtype=torch.bfloat16)
        eager = modulate(x, shift, scale)
        compiled_fn = torch.compile(original_modulate, fullgraph=True)
        compiled = compiled_fn(x, shift, scale)
        reference = (x.float() * (1 + scale.float()) + shift.float()).bfloat16()
        report["cases"]["modulate"] = {"shape": list(x.shape), "dtype": str(x.dtype),
                                        "fixed_eager_vs_original_compiled": stats(eager, compiled),
                                        "original_eager_vs_compiled": stats(original_modulate(x, shift, scale), compiled),
                                        "fp32_vs_compiled": stats(reference, compiled)}
        del x, scale, shift, eager, compiled, reference
        torch.cuda.empty_cache()
        root = Path(__file__).resolve().parents[1]
        component = Component.inspect(root / "models/standalone/prism_alpha_dual_tower_bridge_int8_convrot.safetensors")
        with torch.device("meta"):
            bridge = make_module(component)
        materialize_nonpersistent(bridge, "dual_tower_bridge")
        visual, audio = bridge.build_aligned_freqs(24., (13, 12, 20), 103, device=torch.device(device), dtype=torch.bfloat16)
        rope = torch.compile(original_rotary, fullgraph=True)
        for label, count, heads, freqs in (("visual_cross_rope", 3120, 40, visual), ("audio_cross_rope", 103, 12, audio)):
            q = torch.randn(1, count, heads, 128, device=device, dtype=torch.bfloat16)
            cos, sin = freqs
            eager = apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=2)[0]
            compiled = rope(q, q, cos, sin, unsqueeze_dim=2)[0]
            reference = (q.float() * cos.unsqueeze(2).float() + rotate_half(q.float()) * sin.unsqueeze(2).float()).bfloat16()
            report["cases"][label] = {"shape": list(q.shape), "dtype": str(q.dtype),
                                        "fixed_eager_vs_original_compiled": stats(eager, compiled),
                                        "original_eager_vs_compiled": stats(original_rotary(q, q, cos, sin, unsqueeze_dim=2)[0], compiled),
                                        "fp32_vs_compiled": stats(reference, compiled)}
            del q, eager, compiled, reference
            torch.cuda.empty_cache()
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    output = Path(__file__).resolve().parents[1] / "outputs/review-gpu-pointwise-fixed-numerics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
