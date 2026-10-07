"""ComfyUI sockets carry component descriptors; weights load only for sampling."""
from .prism.format import Component
from .prism.settings import GENERATION_DEFAULTS, validate_generation, validate_sparse


def register_standalone_paths():
    # Register before /object_info enters ComfyUI's temporary filename cache.
    # Registering for the first time inside INPUT_TYPES leaves earlier core
    # loader scans cached as empty for this same response.
    import folder_paths
    from pathlib import Path
    model_root = Path(__file__).resolve().parent / "models"
    standalone = model_root / "standalone"
    for category in ("diffusion_models", "text_encoders", "vae"):
        categorized = model_root / category
        # Prefer the standard layout; keep older flat conversion outputs usable.
        # Registering that flat folder too would mix all components in every menu.
        selected = categorized if categorized.is_dir() else standalone
        if selected.is_dir():
            folder_paths.add_model_folder_path(category, str(selected))
    loras = model_root / 'loras'
    if loras.is_dir():
        folder_paths.add_model_folder_path('loras', str(loras))


def component_loader(kind, category):
    socket = "PRISM_" + kind.upper()
    class Loader:
        CATEGORY = "Prism/loaders"
        RETURN_TYPES = (socket,)
        RETURN_NAMES = (kind,)
        FUNCTION = "load"
        @classmethod
        def INPUT_TYPES(cls):
            import folder_paths
            register_standalone_paths()
            return {"required": {"model_name": (folder_paths.get_filename_list(category),)}}
        def load(self, model_name):
            import folder_paths
            path = folder_paths.get_full_path_or_raise(category, model_name)
            return (Component.inspect(path, kind),)
        @classmethod
        def IS_CHANGED(cls, model_name):
            import folder_paths
            from pathlib import Path
            info = Path(folder_paths.get_full_path_or_raise(category, model_name)).stat()
            return f"{info.st_size}:{info.st_mtime_ns}"
    return Loader


class PrismNativePipeline:
    CATEGORY = "Prism"
    RETURN_TYPES = ("PRISM_PIPELINE",)
    FUNCTION = "assemble"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {kind: ("PRISM_" + kind.upper(),) for kind in
            ("video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge", "text_encoder", "video_vae", "audio_vae")}}
    def assemble(self, **components):
        from .prism.runtime import check_bundle
        return (check_bundle(components),)


class PrismSparseOptions:
    CATEGORY = "Prism/options"
    RETURN_TYPES = ("PRISM_SPARSE_OPTIONS",)
    FUNCTION = "configure"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"enable_bsa": ("BOOLEAN", {"default": False}),
            "sparsity": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 0.99, "step": 0.01}),
            "dynamic_blocks": (["fixed", "ivpq", "penalty"],),
            "high_noise_only": ("BOOLEAN", {"default": False}),
            "advanced_json": ("STRING", {"default": "{}", "multiline": True,
                "tooltip": "All native Prism CLI sparse options, using the exact official option names. See examples/sparse_options.json."})}}
    def configure(self, enable_bsa, sparsity, dynamic_blocks, high_noise_only, advanced_json):
        import json
        values = {"enable_bsa": enable_bsa, "bsa_sparsity": sparsity,
                  "enable_ivpq_dynamic_block": dynamic_blocks == "ivpq",
                  "enable_penalty_dynamic_block": dynamic_blocks == "penalty",
                  "sparse_high_noise_only": high_noise_only}
        advanced = json.loads(advanced_json or "{}")
        if not isinstance(advanced, dict):
            raise ValueError("advanced_json must contain a JSON object")
        values.update(advanced)
        return (validate_sparse(values),)


class PrismNativeSampler:
    CATEGORY = "Prism"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("frames", "audio", "fps")
    FUNCTION = "sample"
    @classmethod
    def INPUT_TYPES(cls):
        required = {"pipeline": ("PRISM_PIPELINE",),
            "mode": (["i2va", "t2va_white_reference"], {"tooltip": "T2VA uses MOVA's experimental white reference image convention."}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "audio_prompt": ("STRING", {"default": "", "multiline": True}),
            "negative_prompt": ("STRING", {"default": GENERATION_DEFAULTS["negative_prompt"], "multiline": True}),
            "width": ("INT", {"default": GENERATION_DEFAULTS["width"], "min": 16, "max": 8192, "step": 16}),
            "height": ("INT", {"default": GENERATION_DEFAULTS["height"], "min": 16, "max": 8192, "step": 16}),
            "num_frames": ("INT", {"default": GENERATION_DEFAULTS["num_frames"], "min": 5, "max": 2049, "step": 4}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            "steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
            "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "visual_shift": ("FLOAT", {"default": GENERATION_DEFAULTS["visual_shift"], "min": 0.01, "max": 100.0}),
            "audio_shift": ("FLOAT", {"default": 7.0, "min": 0.01, "max": 100.0}),
            "offload": (["block", "cpu", "none"],),
            "attention": (["sdpa", "auto"],),
            "int8_backend": (["portable", "kitchen"], {"tooltip": "Video towers: portable W8A16 or Kitchen W8A8. Native audio, bridges and T5 retain floating-point activations. T5 supports standalone BF16 or INT8 weights."}),
            "vae_tiling": ("BOOLEAN", {"default": False}),
            "tile_size": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 8}),
            "tile_stride": ("INT", {"default": 192, "min": 8, "max": 2040, "step": 8}),
            "frame_policy": (["strict", "snap"],)}
        return {"required": required, "optional": {"reference_image": ("IMAGE",), "sparse_options": ("PRISM_SPARSE_OPTIONS",)}}
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        from .prism.runtime import implementation_fingerprint
        return implementation_fingerprint()

    def sample(self, pipeline, reference_image=None, sparse_options=None, **settings):
        import comfy.model_management as management
        from comfy.utils import ProgressBar
        from .prism.runtime import run
        settings = validate_generation(settings)
        management.unload_all_models()
        management.soft_empty_cache()
        progress = ProgressBar(settings["steps"])
        def callback(step, total):
            management.throw_exception_if_processing_interrupted()
            progress.update_absolute(step, total)
        result = run(pipeline, reference_image, settings, sparse=sparse_options,
                     device=management.get_torch_device(), callback=callback,
                     interrupt=management.throw_exception_if_processing_interrupted)
        progress.update_absolute(settings["steps"], settings["steps"])
        return result


class PrismSaveVideo:
    CATEGORY = "Prism"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",), "audio": ("AUDIO",),
            "fps": ("FLOAT", {"default": 24., "min": 1., "max": 120.}),
            "filename_prefix": ("STRING", {"default": "Prism/video"})}}
    def save(self, frames, audio, fps, filename_prefix):
        import folder_paths
        import comfy.model_management as management
        from pathlib import Path
        from .prism.media import save_video
        folder, name, counter, subfolder, prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), frames.shape[2], frames.shape[1])
        path = Path(folder) / f"{name}_{counter:05d}.mp4"
        result = save_video(frames, audio, fps, path, interrupt=management.throw_exception_if_processing_interrupted)
        # Match ComfyUI's native PreviewVideo payload so the canvas can play
        # the muxed audio/video, including on installations without VHS.
        return {"ui": {"images": [{"filename": path.name, "subfolder": subfolder,
                                    "type": "output"}], "animated": [True],
                       "text": [result]}, "result": (result,)}


class PrismDistillLoRALoader:
    CATEGORY = 'Prism/acceleration'
    RETURN_TYPES = ('PRISM_DISTILL_LORAS',)
    FUNCTION = 'load'
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        register_standalone_paths()
        names = folder_paths.get_filename_list('loras')
        return {'required': {'high_noise_lora': (names,), 'low_noise_lora': (names,)}}
    def load(self, high_noise_lora, low_noise_lora):
        import folder_paths
        return ([folder_paths.get_full_path_or_raise('loras', name)
                 for name in (high_noise_lora, low_noise_lora)],)
    @classmethod
    def IS_CHANGED(cls, high_noise_lora, low_noise_lora):
        import folder_paths
        from pathlib import Path
        stats = [Path(folder_paths.get_full_path_or_raise('loras', name)).stat()
                 for name in (high_noise_lora, low_noise_lora)]
        return ':'.join(f'{info.st_size}:{info.st_mtime_ns}' for info in stats)


class PrismAcceleratedSampler:
    CATEGORY = 'Prism/acceleration'
    RETURN_TYPES = ('IMAGE', 'AUDIO', 'FLOAT')
    RETURN_NAMES = ('frames', 'audio', 'fps')
    FUNCTION = 'sample'
    @classmethod
    def IS_CHANGED(cls, **_):
        from .prism.acceleration.runtime import implementation_fingerprint
        return implementation_fingerprint()

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'pipeline': ('PRISM_PIPELINE',),
            'quality': (['Light', 'Standard', 'High', 'Max'], {'default': 'High'}),
            'prompt': ('STRING', {'default': '', 'multiline': True}),
            'audio_prompt': ('STRING', {'default': '', 'multiline': True}),
            'negative_prompt': ('STRING', {'default': GENERATION_DEFAULTS['negative_prompt'], 'multiline': True}),
            'width': ('INT', {'default': 1280, 'min': 16, 'max': 8192, 'step': 16,
                            'tooltip': '720p (1280 × 720) is the upstream reference size. Smaller sizes are experimental and may degrade audio.'}),
            'height': ('INT', {'default': 720, 'min': 16, 'max': 8192, 'step': 16}),
            'num_frames': ('INT', {'default': 205, 'min': 5, 'max': 2049, 'step': 4,
                                 'tooltip': '205 frames at 24 fps is the upstream reference duration. Short clips remain experimental.'}),
            'fps': ('FLOAT', {'default': 24.0, 'min': 1.0, 'max': 120.0}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xFFFFFFFFFFFFFFFF}),
            'vram_gib': ('FLOAT', {'default': 18.0, 'min': 4, 'max': 192, 'step': .5}),
            'ram_gib': ('FLOAT', {'default': 20.0, 'min': 8, 'max': 512, 'step': 1}),
        }, 'optional': {'reference_image': ('IMAGE',), 'distill_loras': ('PRISM_DISTILL_LORAS',)}}
    def sample(self, pipeline, quality, reference_image=None, distill_loras=None, vram_gib=18, ram_gib=20, **settings):
        import comfy.model_management as management
        import folder_paths
        from pathlib import Path
        from comfy.utils import ProgressBar
        from .prism.acceleration.runtime import run
        management.unload_all_models()
        management.soft_empty_cache()
        steps = 20 if quality == 'Max' else 8
        settings['steps'] = steps
        progress = ProgressBar(steps)
        return run(pipeline, reference_image, quality, distill_loras, settings,
            vram_gib=vram_gib, ram_gib=ram_gib,
            output_root=Path(folder_paths.get_output_directory()) / '.prism-acceleration',
            device=management.get_torch_device(), callback=progress.update_absolute,
            interrupt=management.throw_exception_if_processing_interrupted)


NODE_CLASS_MAPPINGS = {
    "PrismVideoDiTLoader": component_loader("video_dit", "diffusion_models"),
    "PrismLowNoiseDiTLoader": component_loader("video_dit_2", "diffusion_models"),
    "PrismAudioDiTLoader": component_loader("audio_dit", "diffusion_models"),
    "PrismBridgeLoader": component_loader("dual_tower_bridge", "diffusion_models"),
    "PrismTextEncoderLoader": component_loader("text_encoder", "text_encoders"),
    "PrismVideoVAELoader": component_loader("video_vae", "vae"),
    "PrismAudioVAELoader": component_loader("audio_vae", "vae"),
    "PrismNativePipeline": PrismNativePipeline,
    "PrismSparseOptions": PrismSparseOptions,
    "PrismNativeSampler": PrismNativeSampler,
    "PrismSaveVideo": PrismSaveVideo,
    'PrismDistillLoRALoader': PrismDistillLoRALoader,
    'PrismAcceleratedSampler': PrismAcceleratedSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PrismVideoDiTLoader": "Prism High Noise Video DiT (UNet)",
    "PrismLowNoiseDiTLoader": "Prism Low Noise Video DiT (UNet)",
    "PrismAudioDiTLoader": "Prism Audio DiT",
    "PrismBridgeLoader": "Prism Audio Video Bridge",
    "PrismTextEncoderLoader": "Prism UMT5 Text Encoder",
    "PrismVideoVAELoader": "Prism Wan Video VAE",
    "PrismAudioVAELoader": "Prism DAC Audio VAE",
    "PrismNativePipeline": "Prism Native Pipeline",
    "PrismSparseOptions": "Prism Native Sparse Options",
    "PrismNativeSampler": "Prism Native Video Audio Sampler",
    "PrismSaveVideo": "Prism Save Video with Audio",
    'PrismDistillLoRALoader': 'Prism FreeVideo Distill LoRA Pair',
    'PrismAcceleratedSampler': 'Prism FreeVideo Accelerated Sampler',
}

# ComfyUI has folder_paths available when importing custom nodes. Keep pure
# package imports possible in conversion/test environments without ComfyUI.
try:
    register_standalone_paths()
except ImportError as error:
    if error.name != "folder_paths":
        raise
