"""Real pixel and ffmpeg regressions from the new 0.1.3 independent audit."""
import json
import shutil
import subprocess

import numpy as np
import pytest
import torch

from prism.media import save_video
from prism.runtime import crop_reference

# Reuse the architecture fixture, while every audit run builds fresh weights.
from test_native import tiny_bundle


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_reference_floating_dtypes_preserve_rgba_center_and_gradient(dtype):
    image = torch.empty(1, 16, 48, 4, dtype=dtype)
    image[:] = torch.tensor([1., 0., 0., .25], dtype=dtype)
    image[:, :, 16:32] = torch.tensor([0., 1., .5, 0.], dtype=dtype)
    image.requires_grad_(True)
    before = image.detach().clone()
    result = crop_reference(image, 16, 16)
    assert result.mode == "RGB" and result.size == (16, 16)
    assert result.getextrema() == ((0, 0), (255, 255), (127, 127))
    assert torch.equal(image.detach(), before)
    assert image.requires_grad and image.grad is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 2])
def test_mux_floating_dtypes_decode_expected_pixels_pcm_and_frames(tmp_path, dtype, channels):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    count, height, width, rate, fps = 9, 16, 16, 16000, 10.
    # A stable representable colour allows comparison across all four dtypes.
    frames = torch.tensor([.75, .25, .5], dtype=dtype).expand(count, height, width, 3).clone()
    t = torch.arange(int(count / fps * rate), dtype=torch.float32) / rate
    expected = torch.stack([.25 * torch.sin(2 * torch.pi * (440 + c * 220) * t)
                            for c in range(channels)])
    waveform = expected.to(dtype).unsqueeze(0).requires_grad_(True)
    target = tmp_path / "float-dtype.mp4"
    save_video(frames, {"waveform": waveform, "sample_rate": rate}, fps, target)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
        "-show_streams", "-show_format", "-of", "json", str(target)]))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert int(video["nb_read_frames"]) == count
    assert (video["width"], video["height"]) == (width, height)
    assert int(audio["channels"]) == channels and int(audio["sample_rate"]) == rate
    pixels = np.frombuffer(subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(target),
        "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]), dtype=np.uint8)
    assert pixels.size == count * height * width * 3
    mean = pixels.reshape(count, height, width, 3).mean(axis=(0, 1, 2))
    np.testing.assert_allclose(mean, np.array([.75, .25, .5]) * 255., atol=6.)
    pcm = np.frombuffer(subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(target),
        "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]), dtype="<f4")
    pcm = pcm.reshape(-1, channels)[:expected.shape[-1]].T
    assert pcm.shape == tuple(expected.shape)
    assert np.isfinite(pcm).all()
    for channel in range(channels):
        assert np.corrcoef(pcm[channel], expected[channel].numpy())[0, 1] > .97
        assert .14 < float(np.sqrt(np.mean(pcm[channel] ** 2))) < .21
    assert waveform.grad is None
    assert not list(tmp_path.glob(".prism_mux_*"))


def test_header_rejects_changed_rotation_group_even_if_it_divides_weight(tiny_bundle, tmp_path):
    from safetensors.torch import save_file
    from prism.format import TensorReader
    from scripts.validate_file_headers import validate_headers
    folder = next(iter(tiny_bundle.values())).path.parent
    output = tmp_path / "bundle"
    shutil.copytree(folder, output)
    component = tiny_bundle["text_encoder"]
    path = output / component.path.name
    with TensorReader(path, copy=True) as reader:
        tensors = {key: reader.get_tensor(key).clone() for key in reader.keys()}
    marker = next(key for key in tensors if key.endswith(".comfy_quant")
                  and json.loads(bytes(tensors[key].tolist()))["convrot_groupsize"] == 64)
    config = json.loads(bytes(tensors[marker].tolist()))
    config["convrot_groupsize"] = 16
    tensors[marker] = torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)
    save_file(tensors, path, metadata=component.metadata)
    with pytest.raises(ValueError, match="group"):
        validate_headers(output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA VAE attention boundary")
@pytest.mark.parametrize("dim", [16, 384])
@pytest.mark.parametrize("height,width", [(1, 1), (1, 2), (2, 2)])
def test_wan_vae_attention_single_token_and_normal_paths_preserve_math(dim, height, width):
    import copy
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock
    from prism.vae import PrismWanAttentionBlock
    torch.manual_seed(327)
    original = WanAttentionBlock(dim).to("cuda", dtype=torch.bfloat16).eval()
    adapted = copy.deepcopy(original)
    adapted.__class__ = PrismWanAttentionBlock
    old_state = original.state_dict()
    assert set(adapted.state_dict()) == set(old_state)
    assert all(torch.equal(v, old_state[k]) for k, v in adapted.state_dict().items())
    value = torch.randn(1, dim, 2, height, width, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        if height * width == 1:
            # The fused default kernel fails on original singleton strideM=1.
            # MATH provides an independent implementation of that same formula.
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                expected = original(value)
        else:
            expected = original(value)
        actual = adapted(value)
    assert actual.dtype == value.dtype and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Miniature native INT8 inference requires CUDA")
def test_native_int8_small_tiled_vae_handles_single_pixel_corner(tiny_bundle):
    from prism.loading import load_component
    from prism.runtime import run
    from prism.vae import PrismWanAttentionBlock
    native = load_component(tiny_bundle["video_vae"])
    assert any(isinstance(child, PrismWanAttentionBlock) for child in native.modules())
    callbacks = []
    frames, audio, fps = run(tiny_bundle, torch.full((1, 24, 64, 4), .5, dtype=torch.bfloat16),
        dict(prompt="test test", width=32, height=16, num_frames=5, steps=2, seed=107,
             cfg=1., offload="block", vae_tiling=True, tile_size=16, tile_stride=8),
        callback=lambda step, total: callbacks.append((step, total)))
    assert frames.shape == (5, 16, 32, 3)
    assert audio["waveform"].shape == (1, 1, 16)
    assert torch.isfinite(frames).all() and torch.isfinite(audio["waveform"]).all()
    assert fps == 24. and callbacks == [(0, 2), (1, 2)]
