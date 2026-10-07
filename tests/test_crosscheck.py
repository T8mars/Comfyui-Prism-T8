"""Regressions from the independent 20-round cross-check."""
import pytest
import torch

from prism.loading import set_tensor
from prism.quantization import quantize


@pytest.mark.parametrize("dtype,value", [
    (torch.float32, torch.finfo(torch.float32).max),
    (torch.float64, 1e300),
])
def test_loader_rejects_finite_source_overflow_after_cast(dtype, value):
    module = torch.nn.Linear(16, 16, bias=False)
    original = module.weight
    source = torch.full((16, 16), value, dtype=dtype)
    assert torch.isfinite(source).all()
    with pytest.raises(ValueError, match="after dtype conversion"):
        set_tensor(module, "weight", source, torch.bfloat16)
    assert module.weight is original
    assert torch.isfinite(module.weight).all()


def test_fp32_timestep_parameter_rejects_float64_overflow():
    module = torch.nn.Module()
    module.time_embedding = torch.nn.Sequential(torch.nn.Linear(16, 16))
    source = torch.full((16, 16), 1e300, dtype=torch.float64)
    with pytest.raises(ValueError, match="after dtype conversion"):
        set_tensor(module, "time_embedding.0.weight", source, torch.bfloat16)
    assert torch.isfinite(module.time_embedding[0].weight).all()


@pytest.mark.parametrize("value", [1e300, float("nan"), float("inf")])
def test_quantization_rejects_nonfinite_fp32_rotation(value):
    with pytest.raises(ValueError, match="Non-finite ConvRot"):
        quantize(torch.full((16, 16), value, dtype=torch.float64))


@pytest.mark.parametrize("rows", [0, -1, True, 1.5])
def test_quantization_rejects_invalid_row_chunk(rows):
    with pytest.raises(ValueError, match="row chunk size"):
        quantize(torch.ones(16, 16), rows=rows)


def test_conversion_overflow_never_installs_component_or_manifest(tmp_path):
    import json
    from diffusers import AutoencoderKLWan
    from safetensors.torch import save_file
    from prism.conversion import convert_bundle

    base, output = tmp_path / "base", tmp_path / "converted"
    vae = AutoencoderKLWan(base_dim=4, z_dim=16, num_res_blocks=1)
    configs = {
        "video_dit": {"num_layers": 2},
        "audio_dit": {"num_layers": 1, "vae_type": "dac"},
        "dual_tower_bridge": {}, "text_encoder": {}, "audio_vae": {},
        "video_vae": dict(vae.config),
    }
    for kind, config in configs.items():
        folder = base / kind
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (base / "tokenizer").mkdir()
    (base / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
    tensors = {key: value.detach().clone().contiguous() for key, value in vae.state_dict().items()}
    tensors[sorted(tensors)[0]].fill_(torch.finfo(torch.float32).max)
    assert all(torch.isfinite(value).all() for value in tensors.values())
    save_file(tensors, base / "video_vae/model.safetensors")
    with pytest.raises(ValueError, match="after BF16 conversion"):
        convert_bundle(base, None, output, components=["video_vae"])
    assert not list(output.iterdir())


@pytest.mark.parametrize("samples,rate", [(0, 48000), (2000, True), (2000, 48000.5), (2000, "48000")])
def test_mux_rejects_empty_audio_and_invalid_sample_rate(tmp_path, samples, rate):
    from prism.media import save_video
    with pytest.raises(ValueError, match="Invalid audio"):
        save_video(torch.zeros(5, 16, 16, 3),
                   {"waveform": torch.zeros(1, 1, samples), "sample_rate": rate},
                   24., tmp_path / "invalid-audio.mp4")
    assert not list(tmp_path.iterdir())


def _package_fixture(tmp_path, monkeypatch, include_reference=True):
    import json
    import scripts.package_workflows as packaging
    folder = tmp_path / "examples"
    folder.mkdir()
    for index in range(1, 10):
        (folder / f"0{index}_canvas.json").write_text(json.dumps(
            {"version": 0.4, "nodes": [{"id": 1}], "links": []}), encoding="utf-8")
    (folder / "README.md").write_text("Canvas instructions", encoding="utf-8")
    if include_reference:
        (folder / "prism_official_case5.png").write_bytes(b"reference fixture")
    target = folder / "Prism-canvas-workflows.zip"
    target.write_bytes(b"previous successful archive")
    monkeypatch.setattr(packaging, "ROOT", tmp_path)
    return packaging, folder, target


def test_missing_package_input_preserves_existing_zip(tmp_path, monkeypatch):
    packaging, folder, target = _package_fixture(tmp_path, monkeypatch, include_reference=False)
    previous = target.read_bytes()
    with pytest.raises(FileNotFoundError):
        packaging.main()
    assert target.read_bytes() == previous
    assert not list(folder.glob("*.part"))


def test_interrupted_package_write_preserves_existing_zip(tmp_path, monkeypatch):
    import zipfile
    packaging, folder, target = _package_fixture(tmp_path, monkeypatch)
    previous = target.read_bytes()
    original = zipfile.ZipFile.write
    def fail(archive, path, *args, **kwargs):
        if str(path).endswith("README.md"):
            raise OSError("injected package write failure")
        return original(archive, path, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "write", fail)
    with pytest.raises(OSError, match="injected package"):
        packaging.main()
    assert target.read_bytes() == previous
    assert not list(folder.glob("*.part"))


def test_successful_package_contains_exact_inputs_and_passes_crc(tmp_path, monkeypatch):
    import zipfile
    packaging, folder, target = _package_fixture(tmp_path, monkeypatch)
    packaging.main()
    with zipfile.ZipFile(target) as archive:
        expected = {f"0{index}_canvas.json" for index in range(1, 10)} | {"README.md", "prism_official_case5.png"}
        assert set(archive.namelist()) == expected
        assert archive.testzip() is None
        assert all(archive.read(name) == (folder / name).read_bytes() for name in expected)
    assert not list(folder.glob("*.part"))


@pytest.mark.parametrize("sparse", [[], False, 0, {"enable_bsaa": True}, {"enable_bsa": "false"}])
def test_runtime_rejects_invalid_sparse_before_loading_or_gpu(tmp_path, monkeypatch, sparse):
    from prism.format import COMPONENTS, Component
    import prism.runtime as runtime
    parts = {kind: Component(tmp_path, kind, {}, {"prism.bundle_id": "crosscheck"}) for kind in COMPONENTS}
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("model loading")
        raise AssertionError("Invalid sparse options reached model loading")
    monkeypatch.setattr(runtime, "load_component", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(ValueError):
        runtime.run(parts, None, {"mode": "t2va_white_reference"}, sparse=sparse)
    assert not calls


@pytest.mark.parametrize("sparse", [None, {}])
def test_runtime_accepts_default_sparse_and_reaches_device_validation(tmp_path, monkeypatch, sparse):
    from prism.format import COMPONENTS, Component
    import prism.runtime as runtime
    parts = {kind: Component(tmp_path, kind, {}, {"prism.bundle_id": "crosscheck"}) for kind in COMPONENTS}
    monkeypatch.setattr(runtime, "load_component", lambda *a, **k: pytest.fail("CPU rejected before model load"))
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        runtime.run(parts, None, {"mode": "t2va_white_reference"}, sparse=sparse, device="cpu")


def test_native_kitchen_keeps_text_and_audio_activations_floating_point(tmp_path, monkeypatch):
    from prism.format import COMPONENTS, Component
    import prism.runtime as runtime
    # Exercise native runtime routing without performing GPU sampling.
    kinds = ["text_encoder", "audio_dit", "dual_tower_bridge", "video_dit"] + [k for k in COMPONENTS if k not in ("text_encoder", "audio_dit", "dual_tower_bridge", "video_dit")]
    parts = {kind: Component(tmp_path, kind, {}, {"prism.bundle_id": "crosscheck"}) for kind in kinds}
    calls = []
    def load(component, **options):
        calls.append((component.kind, options["backend"]))
        if component.kind == "video_dit":
            raise RuntimeError("stop before diffusion loading")
        return torch.nn.Identity()
    monkeypatch.setattr(runtime, "load_component", load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="stop before diffusion loading"):
        runtime.run(parts, None, {"mode": "t2va_white_reference", "int8_backend": "kitchen"}, device="cuda")
    assert calls == [("text_encoder", "portable"), ("audio_dit", "portable"),
                     ("dual_tower_bridge", "portable"), ("video_dit", "kitchen")]


def test_native_inference_changes_invalidate_cached_audio(tmp_path, monkeypatch):
    import prism.runtime as runtime
    from pathlib import Path
    import shutil
    copied = tmp_path / 'prism'
    original = runtime.implementation_fingerprint()
    shutil.copytree(Path(runtime.__file__).parent, copied,
                    ignore=shutil.ignore_patterns('__pycache__', 'acceleration'))
    monkeypatch.setattr(runtime, '__file__', str(copied / 'runtime.py'))
    before = runtime.implementation_fingerprint()
    assert before == original
    (copied / 'quantization.py').write_text('changed audio arithmetic', encoding='utf-8')
    assert before != runtime.implementation_fingerprint()
