"""Move stopped provider downloads between mirrors without loading them in RAM."""
from pathlib import Path
import re
import stat
from . import network
from .file_ops import publish
from .download_cache import discard_read_cache

BLOCK = 4*2**20
PART = 64*2**20


def regular(path, boundary):
    for item in (path,)+tuple(path.parents):
        value=item.lstat()
        if stat.S_ISLNK(value.st_mode) or getattr(value,'st_file_attributes',0)&0x400:
            raise ValueError('Linked download fragments cannot be moved between sources')
        if item==path and not stat.S_ISREG(value.st_mode):
            raise ValueError('Download fragment is not a regular file')
        if item==boundary:break


def merge_prefix(source, target):
    if source == target:
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        publish(source,target)
        return
    # Both files describe the same byte interval. Compare overlap before
    # choosing the longer prefix; never splice conflicting mirror content.
    remaining=min(source.stat().st_size,target.stat().st_size)
    with source.open('rb') as left,target.open('rb') as right:
        while remaining:
            count=min(BLOCK,remaining)
            if left.read(count)!=right.read(count):
                raise ValueError('Retained download fragments disagree; both files retained')
            discard_read_cache(left.fileno(),left.tell()-count,count)
            discard_read_cache(right.fileno(),right.tell()-count,count)
            remaining-=count
    if source.stat().st_size>target.stat().st_size:
        network.retain_partial(target,'switch-retained')
        publish(source,target)
    else:
        network.retain_partial(source,'switch-retained')


def migrate(row, path, candidates, *, allow_restart=False, part_bytes=PART):
    """Called only after the old worker tree has stopped and its log is closed."""
    from .model_transfer import stage_directory, verify
    from .provider_resume import ms_completed
    path=Path(path)
    stage=stage_directory('manual',row,path)
    target=stage/row['file']
    prefixes=[]; parts=[]; completed=[]; unmapped=0
    for name in dict.fromkeys(['manual']+[name for name,_ in candidates]):
        directory=stage_directory(name,row,path)
        item=directory/row['file']
        for unknown in directory.rglob('*.incomplete') if directory.exists() else ():
            regular(unknown,directory)
            if any(marker in str(unknown.relative_to(directory)) for marker in
                   ('.rejected-', '.hash-rejected-', '.interrupted-merge-', '.restart-approved-')):
                continue
            if not unknown.stat().st_size:
                continue
            if unknown.stat().st_size==row['bytes'] and verify(unknown,row):
                completed.append(unknown)
                continue
            if not allow_restart:
                raise RuntimeError('Source switch paused: Xet/legacy fragments have no portable byte-range map. '
                    'The old download is stopped and retained. Enable Allow restarting incomplete downloads '
                    'or --allow-model-restart, then retry to download this file from the new source.')
            unmapped+=unknown.stat().st_size
            network.retain_partial(unknown,'restart-approved')
        if item.is_file():
            regular(item,directory);completed.append(item)
        for suffix in ('.partial','.parallel_tmp'):
            prefix=item.with_suffix(item.suffix+suffix)
            if prefix.is_file():
                regular(prefix,directory);prefixes.append(prefix)
        for piece in item.parent.glob(item.name+'_*') if item.parent.exists() else ():
            match=re.fullmatch(re.escape(item.name)+r'_(\d+)_(\d+)',piece.name)
            if not match:continue
            regular(piece,directory)
            start,end=map(int,match.groups())
            if start%part_bytes or end!=min(row['bytes'],start+part_bytes)-1 or not 0<=piece.stat().st_size<=end-start+1:
                raise ValueError('Retained range layout differs; all fragments retained')
            parts.append(piece)
    if completed:
        for item in completed:
            if verify(item,row):
                if item!=path:publish(item,path)
                return dict(ranges=False,bytes=row['bytes'],unmapped_bytes=unmapped)
        raise ValueError('Completed retained file failed verification; files retained')
    main_partial=path.with_suffix(path.suffix+'.partial')
    if not prefixes and not parts:
        return dict(ranges=False,bytes=main_partial.stat().st_size if main_partial.exists() else 0,unmapped_bytes=unmapped)
    if main_partial.is_file():prefixes.append(main_partial)
    for item in prefixes:
        if item.stat().st_size>row['bytes']:
            raise ValueError('Retained prefix exceeds model size; files retained')
        merge_prefix(item,target.with_suffix(target.suffix+'.parallel_tmp'))
    for item in parts:
        merge_prefix(item,target.parent/item.name)
    return dict(ranges=True,bytes=ms_completed(target,row['bytes'],part_bytes),unmapped_bytes=unmapped)
