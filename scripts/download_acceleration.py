"""Fetch the two independent, checksum-verified LightX2V 260412 rank-256 LoRAs."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'Kijai/WanVideo_comfy'
REVISION = '8260d429d19fd7a72304cad059160b95d843913f'
ROOT = Path(__file__).resolve().parents[1] / 'models/loras'
FILES = {
    'HIGH': '6beb9e6fbd0e72d01763a537b32a0687816e88611a0252f0c438bcec88952afc',
    'LOW': '3723c455cbfc4f39a028fd8b93689e0357fe78c39fc8f1ae8de9fc98825cd167',
}


def fetch(expert, digest, revision):
    name = f'Wan_2_2_I2V_A14B_{expert}_lightx2v_4step_lora_260412_rank_256_fp16.safetensors'
    path = ROOT / name
    candidate = path
    if not path.exists():
        temporary = path.with_suffix('.part')
        url = f'https://huggingface.co/{REPO}/resolve/{revision}/LoRAs/Wan22_Lightx2v/{name}'
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
            while block := response.read(8 * 2**20):
                output.write(block)
        candidate = temporary
    checksum = hashlib.sha256()
    with candidate.open('rb') as stream:
        while block := stream.read(8 * 2**20):
            checksum.update(block)
    actual = checksum.hexdigest()
    if actual != digest:
        raise ValueError(f'Checksum mismatch: {path}')
    if candidate != path:
        candidate.replace(path)
    print(json.dumps({'expert': expert, 'path': str(path), 'sha256': actual}), flush=True)
    return {'file': name, 'sha256': actual}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    revision = REVISION
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda pair: fetch(*pair, revision), FILES.items()))
    (ROOT / 'lightx2v-source.json').write_text(json.dumps({'repository': REPO, 'revision': revision,
        'license': 'Apache-2.0', 'files': rows}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
