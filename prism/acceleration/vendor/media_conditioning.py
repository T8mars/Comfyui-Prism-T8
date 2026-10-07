"""Explicit adapter for native ComfyUI H3 conditioning and reference latents.

No generic CLIP embeddings or MODEL patches can masquerade as H3 inputs. The
native model's layout metadata is retained and checked before CUDA loading.
"""
import math
import torch
from .conditioning import to_cache
from .keyframes import validate_conditioning


def _tensor(value, shape, name):
    if not isinstance(value, torch.Tensor) or value.ndim != len(shape):
        raise ValueError('Invalid ' + name + ' tensor')
    if any(expected is not None and value.shape[i] != expected for i, expected in enumerate(shape)):
        raise ValueError('Invalid ' + name + ' shape: ' + str(tuple(value.shape)))
    if not value.is_floating_point() or not value.numel() or not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' must contain finite floating point values')
    return value.detach().to(device='cpu', dtype=torch.float32).contiguous()


def describe(value, width, height, frames):
    task = value.get('task', 'fl2va' if value.get('keyframe_anchors') else 't2va')
    prompt, tags = value['prompt_embeds'], value['text_token_tags']
    refs = value.get('references', [])
    if not task.startswith('ref2va'):
        if refs:
            raise ValueError('Reference latents require task=ref2va')
        conditions = (value['keyframe_anchors'], value['condition_latents']) if value.get('keyframe_anchors') else None
        validate_conditioning(prompt, tags, conditions, task, width, height)
        count = len(value.get('keyframe_anchors', [])) * (height // 32) * (width // 32)
        return {'task': task, 'text_tokens': len(tags), 'reference_video_tokens': count, 'reference_audio_tokens': 0,
                'width': width, 'height': height, 'frames': frames}
    if value.get('keyframe_anchors') or not isinstance(refs, list) or not 1 <= len(refs) <= 32:
        raise ValueError('Ref2VA needs reference latents and no keyframe anchors')
    # Reuse the shared prompt/token validation without pretending references are anchors.
    to_cache([[prompt[None], {'minimax_token_tags': tags}]], task='ref2va')
    video_rows = audio_rows = 0
    for ref in refs:
        kind = ref.get('kind')
        if kind not in ('image', 'video', 'audio'):
            raise ValueError('Unknown reference modality')
        if kind in ('image', 'video'):
            latent = _tensor(ref.get('latent'), (1, 24, 1 if kind == 'image' else None, None, None), 'reference video')
            if latent.shape[2] < 1 or any(n < 2 or n % 2 for n in latent.shape[-2:]):
                raise ValueError('Reference latent dimensions must fit H3 patches')
            video_rows += math.prod(latent.shape[2:]) // 4
        if kind == 'audio' or ref.get('audio_latent') is not None:
            audio = _tensor(ref.get('audio_latent'), (None, 32), 'reference audio rows')
            if audio.shape[0] % 2:
                raise ValueError('Reference audio must contain both channels')
            audio_rows += audio.shape[0]
    if video_rows and not bool((tags == 0).any()):
        raise ValueError('Visual references need jointly encoded vision tokens')
    task = 'ref2va_av' if video_rows and audio_rows else 'ref2va_audio' if audio_rows else 'ref2va'
    value['task'] = task
    return {'task': task, 'text_tokens': len(tags), 'reference_video_tokens': video_rows,
            'reference_audio_tokens': audio_rows, 'width': width, 'height': height, 'frames': frames}


def from_comfy(conditioning, width, height, frames):
    if len(conditioning) != 1:
        raise ValueError('One H3 conditioning sequence is required')
    extra = conditioning[0][1]
    # These modify sampling semantics; ignoring them would silently change a workflow.
    allowed = {'minimax_token_tags', 'minimax_keyframes', 'minimax_refs', 'pooled_output'}
    unknown = set(extra) - allowed
    if unknown:
        raise ValueError('Unsupported conditioning modifiers: ' + ', '.join(sorted(unknown)))
    keyframes, refs = extra.get('minimax_keyframes', []), extra.get('minimax_refs', [])
    if keyframes and refs:
        raise ValueError('Combined keyframe/reference conditioning is not an official VDN layout')
    anchors = []
    for item in keyframes:
        index = item.get('resolved_frame_index')
        if index not in (0, frames - 1):
            raise ValueError('Only first and last keyframe anchors are supported')
        anchors.append('first' if index == 0 else 'last')
    task = ('ref2va' if refs else 'fl2va' if len(anchors) == 2 else
            'i2va' if anchors == ['first'] else 'l2va' if anchors == ['last'] else 't2va')
    value = to_cache(conditioning, task=task)
    value.update(task=task, width=width, height=height)
    if anchors:
        value.update(keyframe_anchors=anchors, condition_latents=[
            _tensor(row['latent'], (1, 24, 1, height // 16, width // 16), 'keyframe') for row in keyframes])
    if refs:
        value['references'] = []
        for row in refs:
            kind = row['kind']
            ref = {'kind': 'video' if kind == 'video_audio' else kind}
            if kind in ('image', 'video', 'video_audio'):
                ref['latent'] = _tensor(row.get('latent'), (1, 24, None, None, None), 'reference video')
            if kind in ('audio', 'video_audio'):
                audio = _tensor(row.get('audio_latent'), (1, 32, 2, None), 'native reference audio')
                ref['audio_latent'] = audio[0].permute(1, 2, 0).reshape(-1, 32).contiguous()
            value['references'].append(ref)
    return value, describe(value, width, height, frames)
