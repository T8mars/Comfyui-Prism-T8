"""Explicit generation validation and every native inference sparse-attention flag."""
import json
import math

OFFICIAL_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)

GENERATION_DEFAULTS = {"mode": "i2va", "prompt": "", "audio_prompt": "",
    "negative_prompt": OFFICIAL_NEGATIVE_PROMPT,
    "width": 848, "height": 480, "num_frames": 49, "fps": 24.0, "steps": 50,
    "cfg": 5.0, "seed": 42, "visual_shift": 9.0, "audio_shift": 7.0,
    "offload": "block", "attention": "sdpa", "int8_backend": "portable",
    "vae_tiling": False, "tile_size": 256, "tile_stride": 192, "frame_policy": "strict"}
SPARSE_DEFAULTS = {
    "enable_bsa": False, "bsa_sparsity": 0.75, "bsa_chunk_3d_shape_q": [4, 4, 4],
    "bsa_chunk_3d_shape_k": [4, 4, 4], "bsa_cdf_threshold": 0.2,
    "enable_bsa_v2a": False, "bsa_v2a_sparsity": 0.875, "bsa_v2a_audio_chunk_size": 64,
    "bsa_v2a_chunk_3d_shape_k": [4, 4, 4], "bsa_v2a_cdf_threshold": None,
    "enable_audio_guidance": False, "enable_audio_concentration_gate": False,
    "enable_timestep_reliability_gate": False, "audio_boost_gamma": 2.0,
    "enable_audio_weighted_pooling": False, "audio_weighted_lambda": 2.0,
    "enable_variance_guidance": False, "variance_boost_gamma": 2.0,
    "enable_taylor_sparse_attn": False, "taylor_alpha_f": 0.5,
    "enable_rectified_sparse_attn": False, "enable_ivpq_dynamic_block": False,
    "enable_penalty_dynamic_block": False, "dynamic_block_lambda_a": 0.5,
    "dynamic_block_tau_128": 0.25, "dynamic_block_lambda_128": 1.0,
    "enable_layer_adaptive_dynamic_block": False, "sparse_high_noise_only": False,
}


def validate_generation(values):
    if not isinstance(values, dict):
        raise ValueError("Generation options must be an object")
    if set(values) - GENERATION_DEFAULTS.keys():
        raise ValueError(f"Unknown generation options: {sorted(set(values) - GENERATION_DEFAULTS.keys())}")
    settings = {**GENERATION_DEFAULTS, **values}
    for key in ("prompt", "audio_prompt", "negative_prompt"):
        if not isinstance(settings[key], str):
            raise ValueError(f"{key} must be text")
    if type(settings["vae_tiling"]) is not bool:
        raise ValueError("vae_tiling must be a boolean")
    for key in ("fps", "cfg", "visual_shift", "audio_shift"):
        if type(settings[key]) not in (float, int):
            raise ValueError(f"{key} must be a number")
    choices = {"mode": ("i2va", "t2va_white_reference"), "offload": ("none", "cpu", "block"),
               "attention": ("sdpa", "auto"), "int8_backend": ("portable", "kitchen"),
               "frame_policy": ("strict", "snap")}
    for key, options in choices.items():
        if settings[key] not in options:
            raise ValueError(f"Invalid {key}: {settings[key]}")
    for key in ("width", "height", "num_frames", "steps", "seed", "tile_size", "tile_stride"):
        if type(settings[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    for key in ("width", "height"):
        if settings[key] < 16 or settings[key] % 16:
            raise ValueError(f"{key} must be positive and divisible by 16")
    if settings["frame_policy"] == "snap":
        frames = settings["num_frames"]
        settings["num_frames"] = max(5, frames - (frames - 1) % 4)
    if settings["num_frames"] < 5 or (settings["num_frames"] - 1) % 4:
        raise ValueError("num_frames must be at least 5 and satisfy (num_frames - 1) % 4 == 0")
    if settings["steps"] < 1 or not 0 <= settings["seed"] < 2 ** 64:
        raise ValueError("steps must be positive and seed must fit uint64")
    for key in ("fps", "visual_shift", "audio_shift"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(settings["cfg"]) or settings["cfg"] < 0:
        raise ValueError("cfg must be finite and nonnegative")
    if not 0 < settings["tile_stride"] < settings["tile_size"]:
        raise ValueError("VAE tile stride must be positive and smaller than tile size")
    if settings["tile_size"] % 8 or settings["tile_stride"] % 8:
        raise ValueError("VAE tile size/stride must be divisible by 8")
    return settings


def validate_sparse(values):
    if isinstance(values, str):
        values = json.loads(values or "{}")
    if not isinstance(values, dict):
        raise ValueError("Sparse options must be a JSON object")
    if set(values) - SPARSE_DEFAULTS.keys():
        raise ValueError(f"Unknown sparse flags: {sorted(set(values) - SPARSE_DEFAULTS.keys())}")
    options = {**SPARSE_DEFAULTS, **values}
    for key, value in options.items():
        if key.startswith("enable_") or key == "sparse_high_noise_only":
            if type(value) is not bool:
                raise ValueError(f"{key} must be a JSON boolean")
        elif "chunk_3d_shape" in key:
            if not isinstance(value, (list, tuple)) or len(value) != 3 or any(type(x) is not int or x < 1 for x in value):
                raise ValueError(f"{key} must contain three positive integers")
        elif value is None:
            if key not in ("bsa_cdf_threshold", "bsa_v2a_cdf_threshold"):
                raise ValueError(f"{key} cannot be null")
        else:
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid numeric sparse option {key}")
    for key in ("bsa_sparsity", "bsa_v2a_sparsity"):
        if not 0 <= options[key] < 1:
            raise ValueError(f"{key} must be in [0,1)")
    for key in ("bsa_cdf_threshold", "bsa_v2a_cdf_threshold", "taylor_alpha_f"):
        if options[key] is not None and not 0 <= options[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if type(options["bsa_v2a_audio_chunk_size"]) is not int or options["bsa_v2a_audio_chunk_size"] < 64 or options["bsa_v2a_audio_chunk_size"] % 64:
        raise ValueError("bsa_v2a_audio_chunk_size must be a positive multiple of 64")
    if options["enable_taylor_sparse_attn"] and options["enable_rectified_sparse_attn"]:
        raise ValueError("Taylor and rectified sparse attention are mutually exclusive")
    if options["enable_ivpq_dynamic_block"] and options["enable_penalty_dynamic_block"]:
        raise ValueError("IVPQ and penalty dynamic block methods are mutually exclusive")
    dynamic = options["enable_ivpq_dynamic_block"] or options["enable_penalty_dynamic_block"]
    features = ("enable_audio_guidance", "enable_audio_weighted_pooling", "enable_variance_guidance",
                "enable_taylor_sparse_attn", "enable_rectified_sparse_attn")
    if dynamic and any(options[key] for key in features):
        raise ValueError("Dynamic block methods cannot be combined with audio/variance guidance or bias correction")
    if options["enable_layer_adaptive_dynamic_block"] and not dynamic:
        raise ValueError("Layer-adaptive mode requires IVPQ or penalty dynamic blocks")
    if (dynamic or any(options[key] for key in features)) and not options["enable_bsa"]:
        raise ValueError("Video sparse features require enable_bsa=true")
    if (options["enable_audio_concentration_gate"] or options["enable_timestep_reliability_gate"]) and not options["enable_audio_guidance"]:
        raise ValueError("Audio guidance gates require enable_audio_guidance=true")
    return options


def configure_sparse(bridge, values):
    options = validate_sparse(values)
    if options["enable_bsa"] or options["enable_bsa_v2a"]:
        try:
            from .native.models.modules import block_sparse_attention
        except (ImportError, RuntimeError) as error:
            raise RuntimeError("Native Prism BSA requires a working Triton CUDA installation. Install it or disable BSA; settings are never silently ignored.") from error
    bridge.set_sparse_high_noise_only(options["sparse_high_noise_only"])
    video = {"sparsity": options["bsa_sparsity"], "cdf_threshold": options["bsa_cdf_threshold"],
             "chunk_3d_shape_q": options["bsa_chunk_3d_shape_q"], "chunk_3d_shape_k": options["bsa_chunk_3d_shape_k"]}
    cross = {"sparsity": options["bsa_v2a_sparsity"], "cdf_threshold": options["bsa_v2a_cdf_threshold"],
             "chunk_size_q": options["bsa_v2a_audio_chunk_size"], "chunk_3d_shape_k": options["bsa_v2a_chunk_3d_shape_k"]}
    numeric = ("audio_boost_gamma", "audio_weighted_lambda", "variance_boost_gamma", "taylor_alpha_f",
               "dynamic_block_lambda_a", "dynamic_block_tau_128", "dynamic_block_lambda_128")
    video.update({key: options[key] for key in numeric})
    bridge.configure_bsa(options["enable_bsa"], video, options["enable_bsa_v2a"], cross)
    bridge.configure_audio_guidance(options["enable_audio_guidance"], options["enable_audio_concentration_gate"],
                                    options["enable_timestep_reliability_gate"], options["enable_audio_weighted_pooling"])
    bridge.configure_variance_guidance(options["enable_variance_guidance"])
    bridge.configure_taylor_sparse_attn(options["enable_taylor_sparse_attn"])
    bridge.configure_rectified_sparse_attn(options["enable_rectified_sparse_attn"])
    bridge.configure_ivpq_dynamic_block(options["enable_ivpq_dynamic_block"])
    bridge.configure_penalty_dynamic_block(options["enable_penalty_dynamic_block"])
    bridge.configure_layer_adaptive_dynamic_block(options["enable_layer_adaptive_dynamic_block"])
    return options
