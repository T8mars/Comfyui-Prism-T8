"""Install a pinned ComfyUI application into a newly selected workspace."""
import json
from pathlib import Path

from . import network
from .monitoring import save

RECEIPT = '.freevideo-comfy-source.json'
SOURCE_DISK_BYTES = 2**30


def specification():
    # The same revision already supplies the native H3 encoder. Fetch the full
    # application separately; never expand or modify the encoder's sparse tree.
    spec = json.loads((Path(__file__).parent / 'dependencies.json').read_text(encoding='utf-8'))['encoder']
    return dict(url=spec['comfy_url'], commit=spec['comfy_commit'])


def validate_target(target):
    target = Path(target)
    if not target.exists() and not target.is_symlink():
        return
    try:
        owned = json.loads((target / RECEIPT).read_text(encoding='utf-8')) == specification()
    except (OSError, ValueError):
        owned = False
    if not owned or target.is_symlink():
        raise ValueError('This folder already contains ComfyUI or other files. Choose Use existing ComfyUI or another installation folder: ' + str(target))


def new_layout(directory):
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError('Choose a folder for the new ComfyUI installation')
    parent = Path(directory).expanduser().resolve()
    if parent.exists() and not parent.is_dir():
        raise ValueError('The installation folder is a file: ' + str(parent))
    target = parent / 'ComfyUI'
    validate_target(target)
    return dict(root=str(target), python=None, portable=False, separate=True,
                new_comfy=True, source_spec=specification())


def download(target, *, plan, env, run):
    """Publish a complete checkout with its receipt; retain interrupted stages."""
    target = Path(target)
    validate_target(target)
    spec = specification()
    if target.exists():
        return network.clone(target, spec['url'], spec['commit'], network=plan, env=env, run=run)
    stage = target.with_name(target.name + '.freevideo-download')
    # A finished checkout survives cancellation before publication. Git fetch's
    # own incomplete attempts are retained beside this directory by network.clone.
    network.clone(stage, spec['url'], spec['commit'], network=plan, env=env, run=run)
    if not all((stage / name).is_file() for name in ('main.py', 'folder_paths.py', 'requirements.txt', 'LICENSE')) or not (stage / 'comfy_api/latest').is_dir():
        raise ValueError('The downloaded ComfyUI application is incomplete; files retained at ' + str(stage))
    save(stage / RECEIPT, spec)
    if target.exists() or target.is_symlink():
        raise ValueError('The ComfyUI destination changed during installation; the downloaded files are retained: ' + str(stage))
    stage.rename(target)
    return target
