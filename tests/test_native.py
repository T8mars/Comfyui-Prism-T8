"""A real native miniature architecture, never evidence of full-weight quality."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prism.conversion import convert_bundle
from prism.format import Component
from prism.loading import load_component
from prism.quantization import ConvRotLinear
from prism.runtime import run


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    from diffusers import AutoencoderKLWan
    from transformers import UMT5Config, UMT5EncoderModel
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import T5TokenizerFast
    from prism.native.models.modules.wan_video_dit import WanModel
    from prism.native.models.modules.wan_audio_dit import WanAudioModel
    from prism.native.models.modules.interactionv2 import DualTowerConditionalBridge
    from prism.native.models.modules.dac_vae import DAC
    from prism.native.models.modules.mova import MOVABridge

    torch.manual_seed(4)
    root = tmp_path_factory.mktemp("native")
    base = root / "base"
    common = dict(dim=48, ffn_dim=96, text_dim=32, freq_dim=16, eps=1e-6,
                  num_heads=1, has_image_input=False, require_clip_embedding=False)
    video = WanModel(**common, in_dim=36, out_dim=16, num_layers=2, patch_size=(1, 2, 2))
    low = WanModel(**common, in_dim=36, out_dim=16, num_layers=2, patch_size=(1, 2, 2))
    audio = WanAudioModel(**common, in_dim=4, out_dim=4, num_layers=1, patch_size=(1,), vae_type="dac")
    bridge = DualTowerConditionalBridge(visual_layers=2, audio_layers=1, visual_hidden_dim=48,
        audio_hidden_dim=48, head_dim=48, interaction_strategy="full", apply_cross_rope=True, audio_fps=10.)
    video_vae = AutoencoderKLWan(base_dim=4, z_dim=16, num_res_blocks=1)
    audio_vae = DAC(encoder_dim=2, decoder_dim=16, encoder_rates=[2, 2, 2], decoder_rates=[2, 2, 2],
                    latent_dim=4, sample_rate=80, continuous=True, use_weight_norm=False)
    text = UMT5EncoderModel(UMT5Config(vocab_size=16, d_model=32, d_ff=64, d_kv=8, num_heads=4, num_layers=1))
    modules = {"video_dit": video, "video_dit_2": low, "audio_dit": audio,
               "dual_tower_bridge": bridge, "video_vae": video_vae, "audio_vae": audio_vae, "text_encoder": text}
    for kind, module in modules.items():
        folder = base / kind
        folder.mkdir(parents=True)
        config = module.config.to_dict() if hasattr(module.config, "to_dict") else dict(module.config)
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if kind in ("video_vae", "audio_vae", "text_encoder"):
            # Clone shared UMT5 parameters so save_file can store both names.
            save_file({k: v.detach().clone().contiguous() for k, v in module.state_dict().items()}, str(folder / "model.safetensors"))
    (base / "model_index.json").write_text(json.dumps({"boundary_ratio": 0.9}), encoding="utf-8")
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "<pad>": 1, "</s>": 2, "test": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = T5TokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", eos_token="</s>", extra_ids=0)
    fast.save_pretrained(base / "tokenizer")
    fused = MOVABridge(video, low, audio, bridge)
    preview = root / "preview.safetensors"
    save_file({k: v.detach().clone().contiguous() for k, v in fused.state_dict().items()}, str(preview))
    report = convert_bundle(base, preview, root / "out", device="cpu")
    parts = {kind: Component.inspect(root / "out" / entry["file"], kind) for kind, entry in report["components"].items()}
    return parts


def test_native_roundtrip_quantized_components(tiny_bundle):
    for kind, component in tiny_bundle.items():
        module = load_component(component, dtype=torch.float32 if kind == "audio_vae" else torch.bfloat16)
        assert all(not tensor.is_meta for tensor in module.parameters())
        if kind not in ("video_vae", "audio_vae"):
            assert any(isinstance(child, ConvRotLinear) for child in module.modules())
        if kind in ("video_dit", "video_dit_2", "audio_dit"):
            assert module.time_embedding[0].weight.dtype == torch.float32


def test_header_validation_roundtrip(tiny_bundle):
    from scripts.validate_file_headers import validate_headers
    folder = next(iter(tiny_bundle.values())).path.parent
    report = validate_headers(folder)
    assert report["complete"] is True
    assert set(report["components"]) == set(tiny_bundle)
    assert all(part["native_shapes_valid"] for part in report["components"].values())


def test_conversion_resumes_after_later_component_failure(tiny_bundle, tmp_path, monkeypatch):
    import prism.conversion as conversion
    root = next(iter(tiny_bundle.values())).path.parent.parent
    original = conversion.StreamingWriter
    def fail_second(path, *args, **kwargs):
        if "video_dit_2" in Path(path).name:
            raise RuntimeError("injected second-component failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(conversion, "StreamingWriter", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        convert_bundle(root / "base", root / "preview.safetensors", tmp_path)
    manifest = json.loads((tmp_path / "prism_alpha_conversion.json").read_text())
    assert set(manifest["components"]) == {"video_dit"}
    assert manifest["complete"] is False
    first = tmp_path / manifest["components"]["video_dit"]["file"]
    first_mtime = first.stat().st_mtime_ns
    monkeypatch.setattr(conversion, "StreamingWriter", original)
    remaining = set(tiny_bundle) - manifest["components"].keys()
    result = convert_bundle(root / "base", root / "preview.safetensors", tmp_path, components=remaining)
    assert result["complete"] is True
    assert first.stat().st_mtime_ns == first_mtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native inference requires CUDA")
def test_rejects_nonfinite_vae_output_before_pil_conversion(tiny_bundle, monkeypatch):
    from diffusers import AutoencoderKLWan
    original = AutoencoderKLWan.decode
    def broken(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        output.sample.fill_(float("nan"))
        return output
    monkeypatch.setattr(AutoencoderKLWan, "decode", broken)
    with pytest.raises(RuntimeError, match="before image conversion"):
        run(tiny_bundle, torch.full((1, 16, 16, 3), .5),
            dict(prompt="test", width=16, height=16, num_frames=5, steps=1, cfg=1., offload="block"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native inference requires CUDA")
def test_frozen_umt5_offload_restores_tied_parameters_and_quantized_buffers(tiny_bundle):
    from prism.offload import FrozenOffloadModule
    native = load_component(tiny_bundle["text_encoder"])
    assert native.shared.weight is native.encoder.embed_tokens.weight
    pointers = {name: value.data_ptr() for name, value in native.named_parameters()}
    buffer_pointers = {name: value.data_ptr() for name, value in native.named_buffers()}
    buffer_dtypes = {name: value.dtype for name, value in native.named_buffers()}
    wrapped = FrozenOffloadModule(native)
    previous = None
    try:
        with torch.inference_mode():
            for _ in range(3):
                wrapped.to("cuda")
                assert wrapped.shared.weight is wrapped.encoder.embed_tokens.weight
                tokens = torch.tensor([[3, 3, 2, 1]], device="cuda")
                output = wrapped(input_ids=tokens, attention_mask=tokens != 1).last_hidden_state.cpu()
                assert torch.isfinite(output).all()
                if previous is not None:
                    assert torch.equal(output, previous)
                previous = output
                wrapped.to("cpu")
                assert wrapped.shared.weight is wrapped.encoder.embed_tokens.weight
                assert {name: value.data_ptr() for name, value in native.named_parameters()} == pointers
                assert {name: value.data_ptr() for name, value in native.named_buffers()} == buffer_pointers
                assert {name: value.dtype for name, value in native.named_buffers()} == buffer_dtypes
    finally:
        wrapped.cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native inference requires CUDA")
def test_block_offload_restores_cpu_parameters_after_each_transfer(tiny_bundle):
    from prism.native.models.modules.mova import MOVABridge
    from prism.offload import ManagedTransformer
    modules = {kind: load_component(tiny_bundle[kind]) for kind in
               ("video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge")}
    bridge = MOVABridge(**modules)
    managed = ManagedTransformer(bridge, block_offload=True)
    managed.to("cuda")
    block = bridge.fusion_blocks[0]
    try:
        for _ in range(2):
            managed._pre(block, ())
            assert all(p.device.type == "cuda" for p in block.parameters())
            managed._post(block, (), None)
            assert all(p.device.type == "cpu" for p in block.parameters())
            assert all(b.device.type == "cpu" for b in block.buffers())
            assert all(t.device.type == "cpu" for _, parameters, _ in managed.cpu_state[id(block)]
                       for t in parameters.values() if t is not None)
    finally:
        managed.close()


def _offload_expert_case(tiny_bundle):
    from prism.native.models.modules.mova import MOVABridge, assemble_visual_freqs, assemble_audio_freqs
    modules = {kind: load_component(tiny_bundle[kind]) for kind in
               ("video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge")}
    bridge = MOVABridge(**modules).eval()
    grid = (1, 2, 2)
    visual_freqs = assemble_visual_freqs(bridge.video_dit.freqs, *grid, torch.device("cuda"))
    audio_freqs = assemble_audio_freqs(bridge.audio_dit.freqs, 4, torch.device("cuda"))
    visual_rope, audio_rope = bridge.dual_tower_bridge.build_aligned_freqs(
        24., grid, 4, device=torch.device("cuda"), dtype=torch.bfloat16)
    torch.manual_seed(19)
    arguments = [torch.randn(1, 4, 48, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
    arguments += [torch.randn(1, 6, 48, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    arguments += [visual_freqs, audio_freqs, visual_rope, audio_rope, 1., 1., grid]
    return bridge, arguments


def _block_storage(block):
    return {("parameter", name): (value.data_ptr(), value.dtype, value.device.type)
            for name, value in block.named_parameters()} | {
        ("buffer", name): (value.data_ptr(), value.dtype, value.device.type)
        for name, value in block.named_buffers()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native inference requires CUDA")
def test_low_expert_offload_skips_primary_for_keyword_and_positional_calls(tiny_bundle):
    from prism.offload import ManagedTransformer
    from prism.runtime import dense_attention
    bridge, arguments = _offload_expert_case(tiny_bundle)
    fused, override = bridge.fusion_blocks[0], bridge.video_dit_2.blocks[0]
    with torch.inference_mode(), dense_attention("sdpa"), torch.autocast("cuda", dtype=torch.bfloat16):
        bridge.to("cuda")
        expected_high = tuple(value.cpu() for value in fused(*arguments))
        expected_low = tuple(value.cpu() for value in fused(*arguments, override_video_block=override))
        bridge.to("cpu")
        # The old ordinary .to(cpu) baseline above is intentionally independent
        # from the snapshots created here; output equality checks the equations.
        managed = ManagedTransformer(bridge, block_offload=True)
        fused_original, override_original = _block_storage(fused), _block_storage(override)
        managed.to("cuda")
        observations = []
        primary_hook = fused.video_block.register_forward_pre_hook(
            lambda module, args: observations.append(("primary", next(module.parameters()).device.type)))
        override_hook = override.register_forward_pre_hook(
            lambda module, args: observations.append(("override", next(fused.video_block.parameters()).device.type)))
        try:
            for call in ("high", "low_keyword", "low_positional", "high"):
                observations.clear()
                if call == "low_keyword":
                    output = fused(*arguments, override_video_block=override)
                elif call == "low_positional":
                    output = fused(*arguments, override)
                else:
                    output = fused(*arguments)
                expected = expected_high if call == "high" else expected_low
                assert all(torch.equal(actual.cpu(), value) for actual, value in zip(output, expected))
                assert observations == ([("primary", "cuda")] if call == "high" else [("override", "cpu")])
                assert _block_storage(fused) == fused_original
                assert _block_storage(override) == override_original
        finally:
            primary_hook.remove()
            override_hook.remove()
            managed.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native inference requires CUDA")
def test_low_expert_interruption_restores_already_uploaded_active_children(tiny_bundle):
    from prism.offload import ManagedTransformer
    from prism.runtime import dense_attention
    bridge, arguments = _offload_expert_case(tiny_bundle)
    fused, override = bridge.fusion_blocks[0], bridge.video_dit_2.blocks[0]
    calls = []
    def interrupt():
        calls.append(True)
        if len(calls) == 2:
            assert next(fused.video_block.parameters()).device.type == "cpu"
            assert next(fused.audio_block.parameters()).device.type == "cuda"
            raise RuntimeError("cancel during override prehook")
    managed = ManagedTransformer(bridge, block_offload=True, interrupt=interrupt)
    fused_original, override_original = _block_storage(fused), _block_storage(override)
    managed.to("cuda")
    try:
        with torch.inference_mode(), dense_attention("sdpa"), torch.autocast("cuda", dtype=torch.bfloat16):
            with pytest.raises(RuntimeError, match="cancel during override"):
                fused(*arguments, override_video_block=override)
        assert len(calls) == 2
        assert _block_storage(fused) == fused_original
        assert _block_storage(override) == override_original
    finally:
        managed.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native inference requires CUDA")
@pytest.mark.parametrize("offload", ["cpu", "block"])
def test_native_gpu_joint_generation(tiny_bundle, offload):
    settings = dict(prompt="test", width=16, height=16, num_frames=5, steps=2, cfg=1.,
                    fps=24., offload=offload, attention="sdpa", mode="i2va", visual_shift=5., audio_shift=5.)
    calls = []
    frames, audio, fps = run(tiny_bundle, torch.full((1, 16, 16, 3), 0.5), settings,
                              callback=lambda step, total: calls.append((step, total)))
    assert frames.shape == (5, 16, 16, 3)
    assert audio["waveform"].shape == (1, 1, 16)
    assert audio["sample_rate"] == 80 and fps == 24.
    assert torch.isfinite(frames).all() and torch.isfinite(audio["waveform"]).all()
    assert calls == [(0, 2), (1, 2)]
