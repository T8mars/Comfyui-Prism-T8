"""Canvas-facing worker supervision and cancellation."""
import json
import hashlib
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid

import torch
from safetensors.torch import load_file

from ..runtime import check_bundle, crop_reference
from ..settings import validate_generation


def implementation_fingerprint():
    """Invalidate ComfyUI's output cache after an inference implementation fix."""
    from ..runtime import implementation_fingerprint as shared_fingerprint
    folder = Path(__file__).parent
    digest = hashlib.sha256()
    # Standalone loading, T5 quantization, reference cropping and DAC are shared
    # with the native path. Their fixes must invalidate accelerated media too.
    digest.update(shared_fingerprint().encode('ascii'))
    for name in ('worker.py', 'runtime.py', 'cache.py', 'rotation.py', 'memory.py', 'kernels.py',
                 'vendor/SOURCE.json', 'vendor/prism_tiers.json'):
        digest.update(name.encode('utf-8'))
        digest.update((folder / name).read_bytes())
    return digest.hexdigest()


def run(parts, reference, quality, loras, settings, vram_gib=18, ram_gib=20,
        output_root=None, device='cuda:0', callback=None, interrupt=None):
    parts = check_bundle(parts)
    quality = quality.lower()
    if quality not in ('light', 'standard', 'high', 'max'):
        raise ValueError('Unknown FreeVideo quality profile')
    if quality != 'max' and reference is None:
        raise ValueError('Light/Standard/High require a reference IMAGE')
    if quality != 'max' and (not loras or len(loras) != 2):
        raise ValueError('Connect Prism Distill LoRA Pair: 260412 HIGH and LOW rank-256')
    for value, name in ((vram_gib, 'vram_gib'), (ram_gib, 'ram_gib')):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    settings = validate_generation(settings)
    root = Path(__file__).resolve().parents[2]
    tiers = json.loads((Path(__file__).parent / 'vendor/prism_tiers.json').read_text(encoding='utf-8'))
    recipe = next(row for row in tiers['tiers'] if row['id'] == quality)
    settings = {k: settings[k] for k in ('prompt', 'audio_prompt', 'negative_prompt', 'width', 'height',
                                        'num_frames', 'fps', 'seed')}
    settings.update(mode='i2va' if reference is not None else 't2va', steps=recipe['steps'],
                    cfg=recipe['cfg_scale'], visual_shift=recipe['shift'], audio_shift=recipe['audio_shift'])
    output = Path(output_root or root / 'outputs/acceleration') / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=True)
    reference_path = None
    if reference is not None:
        reference_path = output / 'reference.png'
        crop_reference(reference, settings['height'], settings['width']).save(reference_path)
    request = dict(parts={k: str(c.path) for k, c in parts.items()}, quality=quality,
        loras=[str(Path(p).resolve()) for p in (loras or [])], settings=settings,
        reference=str(reference_path) if reference_path else None, vram_gib=vram_gib, ram_gib=ram_gib,
        cache_root=str(root / 'models/.prism-acceleration-cache'), device=str(device))
    request_path = output / 'request.json'
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')
    events = queue.Queue()
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    environment = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
    process = subprocess.Popen([sys.executable, '-u', '-m', 'prism.acceleration.worker', str(request_path), str(output)],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
        creationflags=creationflags, env=environment)
    def read():
        for line in process.stdout:
            events.put(line)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    tail = []
    try:
        with (output / 'worker.log').open('w', encoding='utf-8') as log:
            while process.poll() is None or reader.is_alive() or not events.empty():
                if interrupt:
                    interrupt()
                try:
                    line = events.get(timeout=.25)
                except queue.Empty:
                    continue
                log.write(line)
                log.flush()
                tail = (tail + [line.rstrip()])[-25:]
                print(line.rstrip(), flush=True)
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if callback and event.get('event') == 'step':
                    callback(event['done'], event['total'])
        process.wait()
        if process.returncode:
            raise RuntimeError('FreeVideo worker failed; log: ' + str(output / 'worker.log') + '\n' + '\n'.join(tail))
        receipt = json.loads((output / 'receipt.json').read_text(encoding='utf-8'))
        result = load_file(str(output / 'result.safetensors'))
        frames = result['frames'].float() / 255.0
        waveform = result['waveform']
        if not torch.isfinite(waveform).all() or not torch.isfinite(frames).all():
            raise RuntimeError('Non-finite accelerated output')
        return frames, {'waveform': waveform, 'sample_rate': receipt['sample_rate']}, settings['fps']
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
