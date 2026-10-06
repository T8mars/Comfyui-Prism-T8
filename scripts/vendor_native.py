"""Reproducible vendor import. Run only against the pinned official checkout."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".research/Prism"
COMMIT = "883e90a5c90dc8b7044c65eba0bb64e9342cb46a"
FILES = [
    "models/modules/mova.py", "models/modules/wan_video_dit.py",
    "models/modules/wan_audio_dit.py", "models/modules/interactionv2.py",
    "models/modules/dac_vae.py", "utils/parallel_states.py", "utils/communications.py",
    "diffusion/pipelines/mova_pipeline.py", "diffusion/schedulers/flow_match_pair.py",
]


def main():
    actual = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f"Expected upstream {COMMIT}, got {actual}")
    selected = FILES + [str(p.relative_to(SOURCE / "hymm")).replace("\\", "/")
                        for p in (SOURCE / "hymm/models/modules/block_sparse_attention").glob("*.py")]
    for name in selected:
        relative = Path(name)
        source = (SOURCE / "hymm" / relative).read_text(encoding="utf-8")
        # Private package: do not occupy the global 'hymm' name in ComfyUI.
        dots = "." * len(relative.parts)
        source = re.sub(r"from hymm\.([\w.]+) import", lambda m: f"from {dots}{m[1]} import", source)
        if name == "models/modules/wan_video_dit.py":
            source = re.sub(r"^@torch\.compile[^\n]*\n", "", source, flags=re.MULTILINE)
            source = source.replace("    return (x * (1 + scale) + shift)\n",
                "    # Match the original compiled BF16/FP16 pointwise kernel:\n"
                "    # calculate intermediates in FP32 and round only the output.\n"
                "    output_dtype = torch.promote_types(torch.promote_types(x.dtype, scale.dtype), shift.dtype)\n"
                "    x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x\n"
                "    scale = scale.float() if scale.dtype in (torch.float16, torch.bfloat16) else scale\n"
                "    shift = shift.float() if shift.dtype in (torch.float16, torch.bfloat16) else shift\n"
                "    return (x * (1 + scale) + shift).to(output_dtype)\n", 1)
            source = source.replace(f"from {dots}models.modules.block_sparse_attention import flash_attn_bsa_3d",
                "def flash_attn_bsa_3d(*args, **kwargs):\n"
                "    from .block_sparse_attention import flash_attn_bsa_3d as native_bsa\n"
                "    return native_bsa(*args, **kwargs)", 1)
            # Do not fetch remote kernel code during module import.
            start = source.index("try:\n    from flash_attn_interface")
            end = source.index("try:\n    import flash_attn", start)
            source = source[:start] + "FLASH_ATTN_3_AVAILABLE = False\n\n" + source[end:]
            source = source.replace("except ModuleNotFoundError:", "except (ImportError, OSError):")
        if name == "models/modules/interactionv2.py":
            source = source.replace("@torch.compile(fullgraph=True)\n", "")
            source = source.replace("    q_embed = (q * cos) + (rotate_half(q) * sin)\n"
                "    k_embed = (k * cos) + (rotate_half(k) * sin)\n"
                "    return q_embed, k_embed\n",
                "    # Preserve the original compiled pointwise rounding while\n"
                "    # keeping dense inference independent of Inductor/Triton.\n"
                "    frequency_dtype = torch.promote_types(cos.dtype, sin.dtype)\n"
                "    q_dtype = torch.promote_types(q.dtype, frequency_dtype)\n"
                "    k_dtype = torch.promote_types(k.dtype, frequency_dtype)\n"
                "    q = q.float() if q.dtype in (torch.float16, torch.bfloat16) else q\n"
                "    k = k.float() if k.dtype in (torch.float16, torch.bfloat16) else k\n"
                "    cos = cos.float() if cos.dtype in (torch.float16, torch.bfloat16) else cos\n"
                "    sin = sin.float() if sin.dtype in (torch.float16, torch.bfloat16) else sin\n"
                "    q_embed = (q * cos) + (rotate_half(q) * sin)\n"
                "    k_embed = (k * cos) + (rotate_half(k) * sin)\n"
                "    return q_embed.to(q_dtype), k_embed.to(k_dtype)\n", 1)
        if name == "models/modules/mova.py":
            source = source.replace('with torch.autocast("cuda", dtype=torch.float32):',
                'with torch.autocast(visual_latents.device.type, enabled=False):')
        if name == "diffusion/pipelines/mova_pipeline.py":
            source = source.replace("        vae_tile_sample_stride=None,\n",
                "        vae_tile_sample_stride=None,\n        callback=None,\n")
            source = source.replace("        for idx_step in tqdm(range(total_steps), disable=not is_main):\n",
                "        for idx_step in tqdm(range(total_steps), disable=not is_main):\n"
                "            if callback is not None:\n                callback(idx_step, total_steps)\n")
            source = source.replace('with torch.autocast("cuda", dtype=torch.bfloat16):',
                'with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):')
            source = source.replace('with torch.autocast("cuda", dtype=torch.float32):',
                'with torch.autocast(device.type, enabled=False):')
            source = source.replace("self.audio_vae.decode(audio_latents)",
                "self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype))")
            source = source.replace('        if offload:\n            self.transformer.to("cpu")',
                '        if not torch.isfinite(latents).all() or not torch.isfinite(audio_latents).all():\n'
                '            raise RuntimeError("Prism produced non-finite video/audio latents")\n\n'
                '        if offload:\n            self.transformer.to("cpu")')
            source = source.replace('        video = self.video_processor.postprocess_video(video, output_type="pil")',
                '        for frame_start in range(0, video.shape[2], 4):\n'
                '            if not torch.isfinite(video[:, :, frame_start:frame_start + 4]).all():\n'
                '                raise RuntimeError("Prism video VAE produced non-finite output before image conversion")\n'
                '        video = self.video_processor.postprocess_video(video, output_type="pil")')
            # Make tile setting independent between repeated ComfyUI executions.
            source = source.replace("        video_latents = self.denormalize_video_latents(latents)",
                "        if not enable_vae_tiling:\n            self.video_vae.disable_tiling()\n\n"
                "        video_latents = self.denormalize_video_latents(latents)")
        if name == "models/modules/block_sparse_attention/dynamic_block_shape.py":
            # Registry E702 preflight: split statements without changing the AST.
            source = source.replace("; off +=", "\n    off +=")
            for axis in "THW":
                source = source.replace(f"; a{axis} =", f"\n        a{axis} =")
        target = ROOT / "prism/native" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Vendored from Tencent-Hunyuan/Prism " + COMMIT + ". See NATIVE_CHANGES.md.\n" + source, encoding="utf-8")
        parent = target.parent
        while parent != ROOT / "prism":
            init = parent / "__init__.py"
            if not init.exists():
                init.write_text("", encoding="utf-8")
            parent = parent.parent
    (ROOT / "LICENSE").write_text((SOURCE / "LICENSE").read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
