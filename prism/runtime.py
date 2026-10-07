"""Assemble native MOVABridge and invoke the official paired video/audio pipeline."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from .format import COMPONENTS, Component, load_tokenizer
from .loading import load_component
from .offload import FrozenOffloadModule, ManagedTransformer


def implementation_fingerprint():
    """Invalidate cached native outputs after changes to inference arithmetic."""
    folder = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ('runtime.py', 'loading.py', 'quantization.py', 'offload.py',
                 'settings.py', 'vae.py', 'format.py',
                 'native/models/modules/mova.py', 'native/models/modules/wan_video_dit.py',
                 'native/models/modules/wan_audio_dit.py', 'native/models/modules/interactionv2.py',
                 'native/models/modules/dac_vae.py', 'native/diffusion/pipelines/mova_pipeline.py',
                 'native/diffusion/schedulers/flow_match_pair.py'):
        digest.update(name.encode('utf-8'))
        digest.update((folder / name).read_bytes())
    return digest.hexdigest()


def check_bundle(parts):
    required = set(COMPONENTS) - {"video_dit_2"}
    if required - parts.keys():
        raise ValueError(f"Missing Prism components: {sorted(required - parts.keys())}")
    for name, component in parts.items():
        if component.kind != name:
            raise ValueError(f"Incorrect component in {name} socket: {component.kind}")
    ids = {component.metadata["prism.bundle_id"] for component in parts.values()}
    if len(ids) != 1:
        raise ValueError("Components are from different Prism variants/conversion bundles")
    return parts


def native_linear_backend(kind, video_backend):
    """Keep original floating activations in audio, bridge and text linears.

    ConvRot weights remain INT8. The optional Kitchen activation quantizer is
    reserved for the much larger video towers in the native pipeline; FreeVideo
    retains its own upstream W8A8 execution in its separate worker.
    """
    return "portable" if kind in ("text_encoder", "audio_dit", "dual_tower_bridge") else video_backend


@contextmanager
def dense_attention(backend):
    # Upstream chooses dense kernels in a module-level helper. Restore after each
    # request so SDPA choice cannot leak into the next workflow.
    if backend != "sdpa":
        yield
        return
    from .native.models.modules import wan_video_dit, interactionv2
    original = wan_video_dit.flash_attention
    other = interactionv2.flash_attention
    def sdpa(q, k, v, num_heads, compatibility_mode=False):
        return original(q, k, v, num_heads, compatibility_mode=True)
    wan_video_dit.flash_attention = sdpa
    interactionv2.flash_attention = sdpa
    try:
        yield
    finally:
        wan_video_dit.flash_attention = original
        interactionv2.flash_attention = other


def crop_reference(image, height, width):
    if isinstance(image, torch.Tensor):
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
            raise ValueError("Prism expects one ComfyUI IMAGE [1,H,W,3/4]")
        if not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError("Reference IMAGE must contain finite floating point pixels")
        import numpy as np
        image = Image.fromarray((image[0, :, :, :3].detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    image = image.convert("RGB")
    w, h = image.size
    ratio = width / height
    if w / h > ratio:
        crop_width = max(1, int(h * ratio))
        offset = (w - crop_width) // 2
        image = image.crop((offset, 0, offset + crop_width, h))
    elif w / h < ratio:
        crop_height = max(1, int(w / ratio))
        offset = (h - crop_height) // 2
        image = image.crop((0, offset, w, offset + crop_height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def run(parts, image, settings, sparse=None, device=None, callback=None, interrupt=None, fsdp_mesh=None):
    from .settings import validate_generation, validate_sparse, configure_sparse

    parts = check_bundle(parts)
    settings = validate_generation(settings)
    sparse = validate_sparse({} if sparse is None else sparse)
    if fsdp_mesh is not None:
        if any(c.metadata.get("prism.precision") == "int8_convrot" for c in parts.values()):
            raise ValueError("FSDP requires BF16 standalone components: INT8 buffers would otherwise be replicated, not sharded. Use SP + block offload for INT8.")
        if settings["offload"] == "block":
            raise ValueError("FSDP manages transformer residency; select cpu (frozen encoders only) or none")
    from .native.models.modules.mova import MOVABridge
    from .native.diffusion.pipelines.mova_pipeline import MOVAPipeline
    from .native.diffusion.schedulers.flow_match_pair import FlowMatchPairScheduler
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError("Prism inference requires a CUDA GPU; CPU is supported for conversion and unit tests")
    if image is None:
        if settings["mode"] != "t2va_white_reference":
            raise ValueError("Connect a reference IMAGE for native I2VA")
        image = Image.new("RGB", (settings["width"], settings["height"]), "white")
    else:
        if settings["mode"] != "i2va":
            raise ValueError("T2VA white-reference mode expects no image; use I2VA to condition on an image")
        image = crop_reference(image, settings["height"], settings["width"])
    modules = {}
    managed = None
    try:
        for kind, component in parts.items():
            if interrupt:
                interrupt()
            backend = native_linear_backend(kind, settings["int8_backend"])
            modules[kind] = load_component(component, dtype=torch.float32 if kind == "audio_vae" else torch.bfloat16,
                                           backend=backend, interrupt=interrupt)
            if kind in ("text_encoder", "video_vae", "audio_vae"):
                modules[kind] = FrozenOffloadModule(modules[kind])
        boundary = float(parts["dual_tower_bridge"].metadata["prism.boundary_ratio"])
        bridge = MOVABridge(video_dit=modules["video_dit"], video_dit_2=modules.get("video_dit_2"),
                            audio_dit=modules["audio_dit"], dual_tower_bridge=modules["dual_tower_bridge"],
                            boundary_ratio=boundary)
        configure_sparse(bridge, sparse)
        if fsdp_mesh is not None:
            from .distributed import shard
            shard(bridge, fsdp_mesh)
        managed = ManagedTransformer(bridge, settings["offload"] == "block", interrupt=interrupt,
                                     fixed_device=device if fsdp_mesh is not None else None)
        scheduler_config = json.loads(parts["dual_tower_bridge"].metadata["prism.scheduler_config"])
        scheduler = FlowMatchPairScheduler.from_config(scheduler_config)
        pipe = MOVAPipeline(managed, modules["video_vae"], modules["audio_vae"], modules["text_encoder"],
                            load_tokenizer(parts["text_encoder"]), scheduler, boundary_ratio=boundary, device=device)
        if settings["offload"] in ("cpu", "block"):
            pipe.enable_cpu_offload(device)
        else:
            pipe.to(device)
        with torch.inference_mode(), dense_attention(settings["attention"]):
            video, audio = pipe(prompt=settings["prompt"], image=image, audio_prompt=settings["audio_prompt"] or None,
                negative_prompt=settings["negative_prompt"], seed=settings["seed"],
                height=settings["height"], width=settings["width"], num_frames=settings["num_frames"],
                video_fps=settings["fps"], num_inference_steps=settings["steps"],
                visual_shift=settings["visual_shift"], audio_shift=settings["audio_shift"], cfg_scale=settings["cfg"],
                enable_vae_tiling=settings["vae_tiling"], vae_tile_sample_min_size=settings["tile_size"],
                vae_tile_sample_stride=settings["tile_stride"], callback=callback)
        import numpy as np
        frames = torch.from_numpy(np.stack([np.asarray(frame) for frame in video[0]]).astype(np.float32) / 255.0)
        waveform = audio.detach().float().cpu()
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3 or waveform.shape[0] != 1:
            raise RuntimeError(f"Unexpected DAC audio shape: {tuple(waveform.shape)}")
        # DAC rounds up to whole codec frames; align AUDIO to the exact video duration.
        samples = int(pipe.audio_sample_rate * settings["num_frames"] / settings["fps"])
        waveform = waveform[..., :samples]
        if not torch.isfinite(frames).all() or not torch.isfinite(waveform).all():
            raise RuntimeError("Prism generated non-finite video/audio")
        return frames, {"waveform": waveform, "sample_rate": pipe.audio_sample_rate}, settings["fps"]
    finally:
        if managed is not None:
            if fsdp_mesh is None:
                managed.close()
        for module in modules.values():
            if fsdp_mesh is None or module is modules.get("video_vae") or module is modules.get("audio_vae") or module is modules.get("text_encoder"):
                module.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
