"""Isolated FreeVideo inference worker; invoked by the canvas sampler."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault('PRISM_SAGE_F16ACC', '0')
os.environ.setdefault('PRISM_SAGE_BSA', 'auto')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import psutil
import torch
from PIL import Image
from safetensors.torch import save_file

from ..format import Component, load_tokenizer
from ..loading import load_component
from ..runtime import crop_reference
from . import FREEVIDEO_COMMIT
from .cache import prepare
from .memory import check as check_memory
from .kernels import initialize as initialize_kernels


def emit(**event):
    print(json.dumps(event, ensure_ascii=False), flush=True)


def tier(name):
    data = json.loads((Path(__file__).parent / 'vendor/prism_tiers.json').read_text(encoding='utf-8'))
    return next(row for row in data['tiers'] if row['id'] == name and name != 'original')


@torch.no_grad()
def run(request, output):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('FreeVideo acceleration requires CUDA and Triton')
    from .vendor import prism_policy, prism_runtime
    from .vendor.prism_model import fast_vae, sampling, wan_video_dit
    device = torch.device(request.get('device', 'cuda:0'))
    recipe = tier(request['quality'])
    parts = {k: Component.inspect(v, k) for k, v in request['parts'].items()}
    settings = request['settings']
    process = psutil.Process()
    peaks = {'rss_bytes': 0, 'system_used_bytes': 0,
             'cuda_allocated_bytes': 0, 'cuda_reserved_bytes': 0}
    def guard(*_):
        memory = check_memory(process, request['ram_gib'], emit)
        rss = memory['rss_bytes']
        peaks['rss_bytes'] = max(peaks['rss_bytes'], rss)
        peaks['system_used_bytes'] = max(peaks['system_used_bytes'], memory['total_bytes'] - memory['available_bytes'])
        peaks['ram_guard_bytes'] = max(peaks.get('ram_guard_bytes', 0), memory['guard_bytes'])
        if torch.cuda.is_initialized():
            # Upstream resets CUDA peaks per video step. Keep the maxima here
            # across text encoding, every step, and VAE decoding.
            peaks['cuda_allocated_bytes'] = max(peaks['cuda_allocated_bytes'], torch.cuda.max_memory_allocated(device))
            peaks['cuda_reserved_bytes'] = max(peaks['cuda_reserved_bytes'], torch.cuda.max_memory_reserved(device))
    guard()
    cache = prepare(parts, request['loras'] if recipe['distilled'] else [], request['cache_root'],
        progress=lambda name: (guard(), emit(event='prepare', file=name)))
    manifest = prism_runtime.read_manifest(cache)
    free, total = torch.cuda.mem_get_info(device)
    policy = prism_policy.choose(manifest, vram_total=total, vram_free=free,
        ram_available=psutil.virtual_memory().available, width=settings['width'], height=settings['height'],
        frames=settings['num_frames'], vram_budget=int(request['vram_gib'] * 2**30),
        ram_budget=int(request['ram_gib'] * 2**30), desktop=True,
        audio_teacher=bool(recipe.get('audio_teacher')), teacher_start=(recipe.get('audio_teacher') or {}).get('start_layer', 0),
        attention='sage', force_lean=True)
    if not policy['feasible']:
        raise RuntimeError('Insufficient memory for acceleration: ' + '; '.join(policy['notes']))
    emit(event='policy', policy=policy)
    limit = policy.get('allocator_limit_bytes')
    if limit:
        torch.cuda.set_per_process_memory_fraction(min(.95, limit / total), device)
    kernel_setup = initialize_kernels(policy)
    emit(event='encode_text')
    tick = time.perf_counter()
    tokenizer = load_tokenizer(parts['text_encoder'])
    # The published FreeVideo bundle uses BF16 T5; its optional INT8 encoder
    # keeps floating-point activations (W8A16). Both standalone inputs work here.
    # Kitchen's W8A8 path additionally quantizes them and changes conditioning.
    # For the INT8 option, each linear is dequantized only for its floating-point
    # matmul, without materializing a BF16 model copy.
    text = load_component(parts['text_encoder'], backend='portable', interrupt=guard).to(device)
    embeds = []
    for prompt in (settings['prompt'], settings['negative_prompt'], settings['audio_prompt'] or settings['prompt']):
        guard()
        embeds.append(sampling.t5_prompt_embeds(tokenizer, text, prompt, device)[0])
    # Drop the GPU model directly. Moving a multi-GB encoder back to CPU just
    # before deleting it creates an unnecessary host allocation on Windows.
    del text, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    emit(event='encode_image')
    vae = load_component(parts['video_vae'], interrupt=guard).to(device)
    image = None
    if request.get('reference'):
        image = sampling.image_tensor(crop_reference(Image.open(request['reference']), settings['height'], settings['width']))
    elif recipe['distilled']:
        raise ValueError('Light/Standard/High require a reference image')
    condition, _ = sampling.image_condition(vae, parts['video_vae'].config, image, settings['num_frames'],
        settings['height'], settings['width'], device,
        first_frame_encoder=lambda frame: fast_vae.encode_i2v_condition(vae, frame, settings['num_frames'], mode='template'))
    fast_vae.clear_caches()
    del vae, image
    gc.collect()
    torch.cuda.empty_cache()
    encoding_seconds = time.perf_counter() - tick
    save_file({'condition': condition.cpu().contiguous(), 'video_prompt': embeds[0].cpu().contiguous(),
        'negative_prompt': embeds[1].cpu().contiguous(), 'audio_prompt': embeds[2].cpu().contiguous()},
        str(output / 'conditioning.safetensors'))
    policy.update(sparsity=recipe['sparsity'], cdf=recipe['cdf'])
    model = prism_runtime.PrismModel(cache, device, policy=policy,
        progress=lambda expert, done, count: (guard(), emit(event='load', expert=expert, done=done, total=count)))
    scheduler = prism_runtime.scheduler(cache)
    rate = parts['audio_vae'].config['sample_rate']
    audio_samples = int(rate * settings['num_frames'] / settings['fps'])
    hop = 1
    for factor in parts['audio_vae'].config['encoder_rates']:
        hop *= factor
    sampling_seconds = 0
    def step_done(step, seconds):
        guard()
        emit(event='step', done=step + 1, total=recipe['steps'], seconds=seconds,
             rss_bytes=process.memory_info().rss, available_bytes=psutil.virtual_memory().available)
    try:
        model.sampling_started = True
        tick = time.perf_counter()
        video_latents, audio_latents, steps = sampling.sample(scheduler=scheduler, phase=model.activate,
            condition=condition, prompt_embeds=embeds[0], negative_embeds=embeds[1], audio_prompt_embeds=embeds[2],
            seed=settings['seed'], frames=settings['num_frames'], height=settings['height'], width=settings['width'],
            video_fps=settings['fps'], audio_latent_dim=parts['audio_dit'].config['in_dim'],
            audio_samples=audio_samples, audio_hop=hop,
            boundary_ratio=float(parts['dual_tower_bridge'].metadata['prism.boundary_ratio']),
            steps=recipe['steps'], video_shift=recipe['shift'], audio_shift=recipe['audio_shift'],
            cfg_scale=recipe['cfg_scale'], cfg_steps=recipe['steps'] if recipe['cfg_steps'] == 'all' else recipe['cfg_steps'],
            audio_cfg=recipe['audio_cfg'], device=device, lean=policy['lean'], chunk=policy['chunk'],
            distilled=recipe['distilled'], audio_teacher=recipe.get('audio_teacher'),
            kv_placement=policy['kv_placement'] or 'host', vram_limit=int(request['vram_gib'] * 2**30),
            kv_disk_layers=policy['kv_disk_layers'], kv_spill_path=output / 'audio-kv.spill',
            kv_width=parts['audio_dit'].config['dim'], kv_layers=parts['audio_dit'].config['num_layers'],
            layer_callback=lambda step, branch, layer, count: (guard(), emit(event='layer', step=step, branch=branch, layer=layer, total=count)),
            audio_callback=guard,
            step_callback=step_done)
        sampling_seconds = time.perf_counter() - tick
        if not torch.isfinite(video_latents).all() or not torch.isfinite(audio_latents).all():
            raise RuntimeError('Non-finite accelerated latents')
        video_latents, audio_latents = video_latents.cpu(), audio_latents.cpu()
    finally:
        model.close()
        sampling.release_audio_graphs()
        sampling.release_park_buffers()
        del model, condition, embeds
        gc.collect()
        torch.cuda.empty_cache()
    emit(event='decode')
    tick = time.perf_counter()
    vae = load_component(parts['video_vae'], interrupt=guard).to(device)
    latent = video_latents.to(device)
    mean, std = sampling.latent_stats(parts['video_vae'].config, parts['video_vae'].config['z_dim'], device, latent.dtype)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        decoded = fast_vae.fast_decode(vae, latent * std + mean, low_vram=policy['vae_decode'] == 'fast_low_vram')
    for start in range(0, decoded.shape[2], 8):
        if not torch.isfinite(decoded[:, :, start:start + 8]).all():
            raise RuntimeError('Non-finite accelerated VAE output')
    frames = sampling.frames_uint8(decoded)
    if frames.shape[0] != settings['num_frames']:
        raise RuntimeError('Incorrect accelerated frame count')
    del decoded, latent
    fast_vae.clear_caches()
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    audio_vae = load_component(parts['audio_vae'], dtype=torch.float32, interrupt=guard).to(device)
    waveform = audio_vae.decode(audio_latents.to(device, dtype=torch.float32)).float().cpu()[..., :audio_samples]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[-1] != audio_samples or not torch.isfinite(waveform).all():
        raise RuntimeError('Invalid accelerated audio')
    save_file({'frames': frames.contiguous(), 'waveform': waveform.contiguous(),
               'audio_latents': audio_latents.contiguous()}, str(output / 'result.safetensors'))
    guard()
    from .vendor.prism_model import ivpq_fast
    receipt = dict(status='complete', engine='FreeVideo', commit=FREEVIDEO_COMMIT,
        quality=request['quality'], recipe=recipe, settings=settings, sample_rate=rate, policy=policy,
        cache=str(cache), text_encoding=dict(weights=parts['text_encoder'].metadata['prism.precision'],
            component=str(parts['text_encoder'].path), activations='bfloat16', backend='portable'),
        encoding_seconds=encoding_seconds, sampling_seconds=sampling_seconds,
        decoding_seconds=time.perf_counter() - tick, total_seconds=time.perf_counter() - started,
        attention_stats=dict(ivpq_fast.STATS), attention_mode=ivpq_fast.sage_mode(),
        kernel_setup=kernel_setup,
        steps=steps, peaks=dict(peaks))
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    emit(event='complete', receipt=str(output / 'receipt.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    parser.add_argument('output')
    args = parser.parse_args()
    run(json.loads(Path(args.request).read_text(encoding='utf-8')), args.output)


if __name__ == '__main__':
    main()
