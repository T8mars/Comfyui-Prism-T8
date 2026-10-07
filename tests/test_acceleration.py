import math
import json

import pytest
import torch

from prism.acceleration.rotation import activation_mult, convert_weight, tables
from prism.quantization import hadamard


def sylvester(size):
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return matrix / math.sqrt(size)


@pytest.mark.parametrize("group", [16, 64, 256])
def test_convrot_exact_signed_permutation(group):
    permutation, signs = tables(group)
    d = torch.diag(signs.float())
    perm = torch.eye(group)[:, permutation]
    assert torch.equal(hadamard(group), d @ sylvester(group) @ perm @ d)
    gen = torch.Generator().manual_seed(42)
    w = torch.randint(-127, 128, (7, group * 2), dtype=torch.int8, generator=gen)
    converted = convert_weight(w, group)
    original = w.float().reshape(-1, group) @ hadamard(group)
    recovered = converted.float().reshape(-1, group) @ sylvester(group) * signs
    assert torch.equal(original, recovered)
    x = torch.randn(3, group * 2, generator=gen)
    scale = torch.rand(7, 1, generator=gen) + .01
    old_x = (x.reshape(-1, group) @ hadamard(group)).reshape(x.shape)
    new_x = ((x * activation_mult(group * 2, group)).reshape(-1, group) @ sylvester(group)).reshape(x.shape)
    old = old_x @ (w.float() * scale).T
    new = new_x @ (converted.float() * scale).T
    torch.testing.assert_close(old, new, rtol=2e-5, atol=.002)


def test_convrot_signed_permutation_refuses_overflow():
    with pytest.raises(ValueError, match="-128"):
        convert_weight(torch.full((2, 16), -128, dtype=torch.int8), 16)


def test_exported_qlinear_preserves_bias_and_reconstructed_weights(tmp_path):
    from safetensors.torch import save_file
    from prism.format import TensorReader
    from prism.quantization import quantize
    from prism.acceleration.cache import tensors
    torch.manual_seed(9)
    weight, scale, marker = quantize(torch.randn(7, 256), 256)
    source = 'blocks.0.self_attn.q'
    bias = torch.randn(7).bfloat16()
    values = {source + '.weight': weight, source + '.weight_scale': scale,
              source + '.comfy_quant': marker, source + '.bias': bias}
    path = tmp_path / 'input.safetensors'
    save_file(values, str(path))
    owner = '00.video.self_attn.q'
    with TensorReader(path) as reader:
        output, metadata = tensors(reader, {k: k.replace('blocks.0.self_attn.q', owner) for k in reader.keys()})
    assert torch.equal(output[owner + '.bias'], bias)
    assert metadata[owner]['bias'] and metadata[owner]['rot_block'] == 256
    assert output[owner + '.w_scale'].shape == (7,)
    recovered = (output[owner + '.qweight'].float() @ sylvester(256)) * output[owner + '.act_mult']
    expected = weight.float() @ hadamard(256)
    assert torch.equal(recovered, expected)


def test_acceleration_rejects_bf16_before_preparation(tmp_path):
    from prism.format import COMPONENTS, Component
    from prism.acceleration.cache import prepare
    parts = {kind: Component(tmp_path, kind, {}, {'prism.bundle_id': 'test', 'prism.precision': 'bf16'})
             for kind in COMPONENTS}
    with pytest.raises(ValueError, match='INT8 ConvRot'):
        prepare(parts, [], tmp_path / 'cache')
    assert not (tmp_path / 'cache').exists()


@pytest.mark.parametrize('core', ['video_dit', 'video_dit_2', 'audio_dit', 'dual_tower_bridge'])
def test_mixed_source_text_does_not_allow_bf16_diffusion(tmp_path, core):
    from prism.format import COMPONENTS, Component
    from prism.acceleration.cache import prepare
    parts = {kind: Component(tmp_path, kind, {}, {'prism.bundle_id': 'test',
             'prism.precision': 'bf16' if kind in ('text_encoder', core) else 'int8_convrot'})
             for kind in COMPONENTS}
    with pytest.raises(ValueError, match='INT8 ConvRot ' + core):
        prepare(parts, [], tmp_path / 'cache')
    assert not (tmp_path / 'cache').exists()


def test_prepared_int8_cache_accepts_source_bf16_text(tmp_path):
    from prism.format import COMPONENTS, Component
    from prism.acceleration.cache import fingerprint, prepare
    source = tmp_path / 'component.safetensors'
    source.write_bytes(b'cache identity test fixture')
    parts = {kind: Component(source, kind, {}, {'prism.bundle_id': 'test',
             'prism.precision': 'bf16' if kind in ('text_encoder', 'video_vae', 'audio_vae') else 'int8_convrot'})
             for kind in COMPONENTS}
    target = tmp_path / 'cache' / fingerprint(parts, [])
    target.mkdir(parents=True)
    (target / 'manifest.json').write_text(json.dumps({'files': {}}), encoding='utf-8')
    assert prepare(parts, [], tmp_path / 'cache') == target


def test_embedded_scheduler_adapter_produces_paired_eight_step_schedule(tmp_path):
    from prism.acceleration.vendor.prism_runtime import scheduler
    config = tmp_path / 'configs/scheduler/scheduler_config.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'num_train_timesteps': 1000, 'extra_one_step': True, 'sigma_min': 0.0}), encoding='utf-8')
    actual = scheduler(tmp_path)
    actual.set_timesteps(8, shift=5., device='cpu')
    actual.set_pair_postprocess_by_name('dual_sigma_shift', visual_shift=5., audio_shift=7.)
    pairs = actual.get_pairs()
    assert pairs.shape == (8, 2)
    assert torch.equal(pairs[0], torch.tensor([1000., 1000.]))
    assert torch.all(pairs[:-1] > pairs[1:]) and torch.all(pairs[-1] > 0)


@pytest.mark.parametrize('changed_file', ['acceleration/worker.py', 'quantization.py', 'acceleration/kernels.py'])
def test_inference_fix_invalidates_previous_accelerated_output(tmp_path, monkeypatch, changed_file):
    import shutil
    from pathlib import Path
    import prism.runtime as shared_runtime
    import prism.acceleration.runtime as runtime
    copied = tmp_path / 'prism'
    shutil.copytree(Path(shared_runtime.__file__).parent, copied,
                    ignore=shutil.ignore_patterns('__pycache__'))
    monkeypatch.setattr(shared_runtime, '__file__', str(copied / 'runtime.py'))
    monkeypatch.setattr(runtime, '__file__', str(copied / 'acceleration/runtime.py'))
    previous = runtime.implementation_fingerprint()
    assert runtime.implementation_fingerprint() == previous
    (copied / changed_file).write_text('fixed text conditioning', encoding='utf-8')
    assert runtime.implementation_fingerprint() != previous
