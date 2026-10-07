"""Torch-free proposals for spending measured spare memory on compute.

These are trial bounds, not predictions of speed or certified peak memory.
No device name/nominal-capacity tier enables a candidate on another machine.
"""
import math

GiB = 2**30
CHOICES = {'head_parallelism': (1, 2, 4), 'head_chunk': (2, 4, 8, 16),
           'window_batch': (1, 2, 4, 8, 16, 32),
           'ff_chunk': (512, 1024, 2048, 4096, 8192, 16384),
           'projection_chunk': (512, 1024, 2048, 4096, 8192)}


def valid_parallelism(engine):
    count = engine.get('head_parallelism', 1)
    return (type(count) is int and count in CHOICES['head_parallelism'] and
            (count == 1 or (engine.get('resident_blocks') == 50 and
                           not engine.get('attention_cpu_outputs') and
                           not engine.get('grouped_attention_outputs') and
                           type(engine.get('head_chunk')) is int and engine['head_chunk'] > 0)))


def spare_compute_trials(row, engine, peak_bytes, headroom_bytes):
    """Keep consumer defaults; nominate larger work only from a full baseline.

    The caller authenticates matching complete geometry/configuration and clamps
    headroom against live memory. Each proposal still needs local full-request
    timing, memory and raw output validation before it can become a default.
    """
    if (not valid_parallelism(engine) or engine.get('resident_blocks') != 50 or
            engine.get('attention_cpu_outputs') or
            engine.get('grouped_attention_outputs') or
            type(engine.get('head_chunk')) is not int or engine['head_chunk'] <= 0 or
            any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in (peak_bytes, headroom_bytes))):
        return []
    measured = row.get('engine', {})
    weights = measured.get('config', {}).get('resident_weight_bytes')
    if type(weights) not in (int, float) or not math.isfinite(weights) or not 0 < weights < peak_bytes:
        # A byte-exact resident-weight measurement avoids treating a nominal
        # block-size guess as extra activation memory on another model.
        return []
    lanes = engine.get('head_parallelism', 1)
    workspace = max(GiB // 2, (peak_bytes - weights) / lanes)
    affordable = [n for n in CHOICES['head_parallelism'] if n > lanes and
                  (n - lanes) * workspace <= headroom_bytes]
    result = []
    if affordable:
        result.append(({'head_parallelism': max(affordable)},
                       dict(estimated_extra_bytes=int((max(affordable) - lanes) * workspace),
                            workspace_per_lane_bytes=int(workspace),
                            reason='Overlap unchanged head groups; extra live workspace uses measured spare VRAM')))
    canvas = row.get('geometry', {})
    rows = canvas.get('video_tokens')
    if type(rows) is not int or rows <= 0:
        return result
    # Bound rows using all known conditioning. Missing token metadata disables
    # row-based proposals, rather than guessing that reference inputs are free.
    shape = measured.get('conditioning_shape')
    if not isinstance(shape, (tuple, list)) or not shape or type(shape[0]) is not int or shape[0] <= 0:
        return result
    rows += shape[0]
    for field in ('reference_video_tokens', 'reference_audio_tokens'):
        count = measured.get('conditioning_info', {}).get(field, canvas.get(field, 0))
        if type(count) is not int or count < 0:
            return result
        rows += count
    # Deliberately bounded hypotheses, not a full-run peak extrapolated from
    # one-step probes. GEMM row shapes may change rounding, so they remain
    # arithmetic trials even when their temporary buffers appear affordable.
    for name, per_item in (('ff_chunk', 128 * 1024), ('projection_chunk', 64 * 1024),
                           ('window_batch', 4 * rows * max(1, engine['head_chunk']) * 128 * 2)):
        if name == 'ff_chunk' and engine.get('linear_compute') != 'native-fp8':
            continue
        if name == 'window_batch' and (engine.get('window_varlen') or
                engine['attention'].split('/')[-1] in ('fa2', 'fa4')):
            continue
        current = engine.get(name, 0)
        upper = rows if name != 'window_batch' else canvas.get('latent_frames', 1)
        values = [n for n in CHOICES[name] if current < n <= upper and
                  (n - current) * per_item <= headroom_bytes]
        if values:
            value = max(values)
            result.append(({name: value}, dict(estimated_extra_bytes=(value-current)*per_item,
                reason='Test fewer serial chunks within measured spare VRAM; output equivalence is unverified')))
    return result
