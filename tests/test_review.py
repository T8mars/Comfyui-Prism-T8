"""Independent review regressions; CPU-only unless explicitly coordinated."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prism.format import Component, TensorReader
from prism.loading import load_component
from prism.quantization import decode_config, quantize
from prism.runtime import crop_reference, dense_attention
from prism.settings import validate_generation, validate_sparse


@pytest.mark.parametrize("options", [
    {"vae_tiling": "false"}, {"fps": True}, {"cfg": True}, {"visual_shift": "5"},
    {"prompt": None}, {"audio_prompt": []}, {"negative_prompt": 17},
])
def test_generation_rejects_wrong_scalar_types(options):
    with pytest.raises(ValueError):
        validate_generation(options)


@pytest.mark.parametrize("key", ["audio_boost_gamma", "audio_weighted_lambda", "variance_boost_gamma",
                                  "taylor_alpha_f", "dynamic_block_lambda_a", "dynamic_block_tau_128",
                                  "dynamic_block_lambda_128"])
def test_sparse_rejects_null_nonoptional_numeric(key):
    with pytest.raises(ValueError):
        validate_sparse({key: None})


def test_sparse_allows_native_optional_cdf():
    options = validate_sparse({"bsa_cdf_threshold": None, "bsa_v2a_cdf_threshold": None})
    assert options["bsa_cdf_threshold"] is None


def test_reference_rejects_nonfinite_before_integer_pixels():
    image = torch.zeros(1, 16, 16, 3)
    image[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        crop_reference(image, 16, 16)


def test_reference_crop_is_centered_and_rgb():
    from PIL import Image
    image = Image.new("RGBA", (48, 16), "red")
    image.paste((0, 255, 0, 255), (16, 0, 32, 16))
    result = crop_reference(image, 16, 16)
    assert result.mode == "RGB" and result.size == (16, 16)
    assert result.getpixel((8, 8)) == (0, 255, 0)


@pytest.mark.parametrize("height,width", [(8192, 16), (16, 8192)])
def test_reference_extreme_aspect_ratio_preserves_color(height, width):
    from PIL import Image
    validate_generation({"height": height, "width": width})
    result = crop_reference(Image.new("RGB", (32, 32), (230, 50, 70)), height, width)
    assert result.size == (width, height)
    assert result.getextrema() == ((230, 230), (50, 50), (70, 70))


def test_reference_tensor_with_grad_can_be_used_for_inference():
    image = torch.full((1, 32, 32, 3), .5, requires_grad=True)
    result = crop_reference(image, 16, 16)
    assert result.getextrema() == ((127, 127),) * 3
    assert image.requires_grad and image.grad is None


def test_quantization_json_requires_object():
    tensor = torch.tensor(list(b"[]"), dtype=torch.uint8)
    with pytest.raises(ValueError):
        decode_config(tensor)


def test_loader_rejects_integer_bias_before_cast(tmp_path, monkeypatch):
    import prism.loading as loading
    q, scale, config = quantize(torch.randn(16, 16))
    path = tmp_path / "bad.safetensors"
    save_file({"layer.weight": q, "layer.weight_scale": scale, "layer.comfy_quant": config,
               "layer.bias": torch.zeros(16, dtype=torch.int32)}, path)
    monkeypatch.setattr(loading, "make_module", lambda _: torch.nn.Sequential())
    def factory(_):
        model = torch.nn.Module()
        model.layer = torch.nn.Linear(16, 16)
        return model
    monkeypatch.setattr(loading, "make_module", factory)
    with pytest.raises(ValueError, match="bias"):
        load_component(Component(path, "test", {}, {}))


def test_loader_interrupts_between_tensors_and_closes_reader(tmp_path, monkeypatch):
    import prism.loading as loading
    path = tmp_path / "weights.safetensors"
    save_file({"weight": torch.ones(16, 16), "bias": torch.zeros(16)}, path)
    monkeypatch.setattr(loading, "make_module", lambda _: torch.nn.Linear(16, 16))
    closed = []
    original = TensorReader.close
    def close(reader):
        closed.append(True)
        original(reader)
    monkeypatch.setattr(TensorReader, "close", close)
    calls = []
    def interrupt():
        calls.append(True)
        if len(calls) == 2:
            raise InterruptedError("review cancellation")
    with pytest.raises(InterruptedError, match="review cancellation"):
        load_component(Component(path, "test", {}, {}), interrupt=interrupt)
    assert len(calls) == 2 and closed
    path.unlink()


def test_malformed_safetensors_layout_rejected(tmp_path):
    path = tmp_path / "truncated.safetensors"
    save_file({"x": torch.ones(16)}, path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(Exception):
        TensorReader(path)


def test_dense_attention_restores_on_exception():
    from prism.native.models.modules import wan_video_dit, interactionv2
    first, second = wan_video_dit.flash_attention, interactionv2.flash_attention
    with pytest.raises(InterruptedError):
        with dense_attention("sdpa"):
            assert wan_video_dit.flash_attention is not first
            raise InterruptedError()
    assert wan_video_dit.flash_attention is first
    assert interactionv2.flash_attention is second


@pytest.mark.parametrize("weights,match", [
    ({"weight": torch.ones(16, 16)}, "Incomplete"),
    ({"weight": torch.ones(15, 16), "bias": torch.zeros(16)}, "Shape mismatch"),
    ({"weight": torch.full((16, 16), float("nan")), "bias": torch.zeros(16)}, "Non-finite"),
])
def test_loader_rejects_incomplete_shapes_and_nonfinite(tmp_path, monkeypatch, weights, match):
    import prism.loading as loading
    path = tmp_path / "invalid.safetensors"
    save_file(weights, path)
    monkeypatch.setattr(loading, "make_module", lambda _: torch.nn.Linear(16, 16))
    with pytest.raises(ValueError, match=match):
        load_component(Component(path, "test", {}, {}))


def test_native_scheduler_matches_pinned_source():
    import runpy
    from prism.native.diffusion.schedulers.flow_match_pair import FlowMatchPairScheduler
    root = Path(__file__).resolve().parents[1]
    source = root / ".research/Prism/hymm/diffusion/schedulers/flow_match_pair.py"
    if not source.exists():
        pytest.skip("Pinned upstream checkout is a local review dependency")
    official = runpy.run_path(str(source))["FlowMatchPairScheduler"]
    native, reference = FlowMatchPairScheduler(), official()
    for scheduler in (native, reference):
        scheduler.set_timesteps(17)
        scheduler.set_pair_postprocess_by_name("dual_sigma_shift", visual_shift=9., audio_shift=7.)
    assert torch.equal(native.pair_timesteps, reference.pair_timesteps)
    assert torch.equal(native.pair_sigmas, reference.pair_sigmas)
    sample = torch.linspace(-1., 1., 32)
    prediction = torch.sin(sample)
    for column in (0, 1):
        for i in range(17):
            first = native.pair_timesteps[i, column]
            second = native.pair_timesteps[i + 1, column] if i < 16 else None
            assert torch.equal(native.step_from_to(prediction, first, second, sample),
                               reference.step_from_to(prediction, first, second, sample))


def test_offload_restores_hooks_after_forward_exception():
    from prism.offload import ManagedTransformer
    class Broken(torch.nn.Linear):
        def forward(self, *args):
            raise InterruptedError("block failure")
    class Bridge(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion_blocks = torch.nn.ModuleList([Broken(16, 16)])
            self.remaining_video_blocks = torch.nn.ModuleList()
            self.video_dit_2 = None
    bridge = Bridge()
    managed = ManagedTransformer(bridge, block_offload=True)
    block = bridge.fusion_blocks[0]
    original = block.weight.detach()
    pointer = original.data_ptr()
    with pytest.raises(InterruptedError, match="block failure"):
        block(torch.zeros(1, 16))
    assert block.weight.data_ptr() == pointer
    assert torch.equal(block.weight, original)
    managed.close()
    assert not managed.handles and not managed.cpu_state
    assert not block._forward_hooks and not block._forward_pre_hooks


def test_component_cache_changes_after_atomic_replace(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import types
    from prism.format import StreamingWriter
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("review_prism_plugin", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, plugin)
    spec.loader.exec_module(plugin)
    path = tmp_path / "component.safetensors"
    metadata = {"prism.format_version": "1", "prism.component": "video_dit", "prism.config": "{}"}
    with StreamingWriter(path, metadata) as writer:
        writer.add("test", torch.ones(1))
        writer.finish()
    folder_paths = types.SimpleNamespace(get_full_path_or_raise=lambda *_: str(path))
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    loader = plugin.NODE_CLASS_MAPPINGS["PrismVideoDiTLoader"]
    before = loader.IS_CHANGED(path.name)
    assert loader().load(path.name)[0].kind == "video_dit"
    with StreamingWriter(path, metadata, overwrite=True) as writer:
        writer.add("test", torch.ones(2))
        writer.finish()
    assert loader.IS_CHANGED(path.name) != before


def test_vendor_generation_is_reproducible(tmp_path):
    import runpy
    root = Path(__file__).resolve().parents[1]
    source = root / ".research/Prism"
    if not (source / "hymm").exists():
        pytest.skip("Pinned upstream checkout is a local review dependency")
    namespace = runpy.run_path(str(root / "scripts/vendor_native.py"))
    main = namespace["main"]
    main.__globals__["ROOT"] = tmp_path
    main.__globals__["SOURCE"] = source
    main()
    generated = [p for p in (tmp_path / "prism/native").rglob("*.py") if p.name != "__init__.py"]
    assert len(generated) == 16
    for path in generated:
        assert path.read_bytes() == (root / path.relative_to(tmp_path)).read_bytes()
    assert (tmp_path / "LICENSE").read_bytes() == (root / "LICENSE").read_bytes()


def test_mux_interrupt_removes_partial_output(tmp_path):
    import shutil
    from prism.media import save_video
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    path = tmp_path / "cancel.mp4"
    calls = []
    def interrupt():
        calls.append(True)
        if len(calls) >= 2:
            raise InterruptedError("cancel encoding")
    with pytest.raises(InterruptedError, match="cancel encoding"):
        save_video(torch.zeros(1, 16, 16, 3),
                   {"waveform": torch.zeros(1, 1, 2000), "sample_rate": 48000},
                   24., path, interrupt=interrupt)
    assert len(calls) >= 2 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), True, "24"])
def test_mux_rejects_invalid_frame_rate(tmp_path, fps):
    from prism.media import save_video
    with pytest.raises(ValueError, match="frame rate"):
        save_video(torch.zeros(1, 16, 16, 3),
                   {"waveform": torch.zeros(1, 1, 2000), "sample_rate": 48000},
                   fps, tmp_path / "invalid.mp4")
    assert not list(tmp_path.iterdir())
