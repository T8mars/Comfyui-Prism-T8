"""Generate complete canvas workflows for all four FreeVideo profiles."""
import json
from pathlib import Path
from create_example import Canvas, LOADERS, PROMPT
from prism.settings import OFFICIAL_NEGATIVE_PROMPT

ROOT = Path(__file__).resolve().parents[1]


def workflow(quality):
    c = Canvas()
    parts = {}
    for index, (kind, loader) in enumerate(LOADERS.items()):
        precision = 'bf16' if kind.endswith('vae') else 'int8_convrot'
        parts[kind] = c.node(loader, kind.replace('_', ' ').title(), (50, 60 + index * 104), (410, 80),
            outputs=[(kind, 'PRISM_' + kind.upper())],
            widgets=[('model_name', f'prism_alpha_{kind}_{precision}.safetensors')])
    pipeline = c.node('PrismNativePipeline', 'Standalone Prism Components', (1010, 60), (460, 195),
        sockets=[(kind, 'PRISM_' + kind.upper()) for kind in LOADERS], outputs=[('PRISM_PIPELINE', 'PRISM_PIPELINE')])
    for kind, part in parts.items():
        c.connect(part, 0, pipeline, kind)
    image = c.node('LoadImage', 'Reference Image', (540, 60), (400, 360),
        outputs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')], widgets=[('image', 'prism_official_case5.png'), ('upload', 'image')])
    lora = None
    if quality != 'Max':
        lora = c.node('PrismDistillLoRALoader', '260412 Distill LoRA Pair', (540, 470), (400, 130),
            outputs=[('PRISM_DISTILL_LORAS', 'PRISM_DISTILL_LORAS')],
            widgets=[(name, f'Wan_2_2_I2V_A14B_{expert}_lightx2v_4step_lora_260412_rank_256_fp16.safetensors')
                     for name, expert in [('high_noise_lora', 'HIGH'), ('low_noise_lora', 'LOW')]])
    sampler = c.node('PrismAcceleratedSampler', 'FreeVideo ' + quality, (1010, 300), (460, 740),
        sockets=[('pipeline', 'PRISM_PIPELINE'), ('reference_image', 'IMAGE'), ('distill_loras', 'PRISM_DISTILL_LORAS')],
        outputs=[('frames', 'IMAGE'), ('audio', 'AUDIO'), ('fps', 'FLOAT')],
        widgets=[('quality', quality), ('prompt', PROMPT), ('audio_prompt', ''), ('negative_prompt', OFFICIAL_NEGATIVE_PROMPT),
                 ('width', 1280), ('height', 720), ('num_frames', 205), ('fps', 24.), ('seed', 42),
                 ('control_after_generate', 'fixed'), ('vram_gib', 18.), ('ram_gib', 20.)])
    c.connect(pipeline, 0, sampler, 'pipeline')
    c.connect(image, 0, sampler, 'reference_image')
    if lora:
        c.connect(lora, 0, sampler, 'distill_loras')
    prefix = 'Prism/FreeVideo_' + quality
    frames = c.node('SaveImage', 'Save Frames + Canvas', (1550, 60), (420, 300),
        sockets=[('images', 'IMAGE')], widgets=[('filename_prefix', prefix + '/frame')])
    audio = c.node('SaveAudio', 'Save Audio', (1550, 410), (420, 200),
        sockets=[('audio', 'AUDIO')], outputs=[('audio', 'AUDIO')], widgets=[('filename_prefix', prefix + '/audio')])
    video = c.node('PrismSaveVideo', 'Save Video + Audio', (1550, 670), (420, 380),
        sockets=[('frames', 'IMAGE'), ('audio', 'AUDIO'), ('fps', 'FLOAT')], outputs=[('video_path', 'STRING')],
        widgets=[('fps', 24.), ('filename_prefix', prefix + '/video')])
    video['inputs'][2]['widget'] = {'name': 'fps'}
    for slot, target, name in [(0, frames, 'images'), (1, audio, 'audio'), (0, video, 'frames'), (1, video, 'audio'), (2, video, 'fps')]:
        c.connect(sampler, slot, target, name)
    data = c.serialize('FreeVideo_' + quality)
    data['groups'][1]['title'] = '2. Reference / distill LoRA'
    return data


def main():
    for index, quality in enumerate(('Light', 'Standard', 'High', 'Max'), 6):
        path = ROOT / 'examples' / f'{index:02d}_freevideo_{quality.lower()}.json'
        path.write_text(json.dumps(workflow(quality), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(path)


if __name__ == '__main__':
    main()
