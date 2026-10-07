"""Vendor the pinned FreeVideo Prism inference dependency closure only."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prism.acceleration import FREEVIDEO_COMMIT

REPO = "FlashML-org/FreeVideo"
CACHE = ROOT / ".research/FreeVideo-source" / FREEVIDEO_COMMIT
DEST = ROOT / "prism/acceleration/vendor"


def adapt(path, data):
    if path == 'freevideo_engine/prism_model/ivpq_fast.py':
        source = data.decode('utf-8')
        before = 'env = os.environ.get("PRISM_SAGE_BSA")'
        if source.count(before) != 1:
            raise ValueError('Pinned Sage attention selector changed')
        return source.replace(before, 'env = os.environ.get("PRISM_SAGE_BSA", "auto")').encode('utf-8')
    if path != 'freevideo_engine/prism_model/sampling.py':
        return data
    source = data.decode('utf-8')
    replacements = {
        'kv_width=None, kv_layers=None, audio_maps=None, audio_callback=None):':
        'kv_width=None, kv_layers=None, audio_maps=None, audio_callback=None,\n           audio_prompt_embeds=None, video_fps=24.0):',
        'ape = prompt_embeds  # audio prompt = prompt (research runner)':
        'ape = prompt_embeds if audio_prompt_embeds is None else audio_prompt_embeds',
        'video_fps=24.0, num_train_timesteps=scheduler.num_train_timesteps, residency=residency,':
        'video_fps=video_fps, num_train_timesteps=scheduler.num_train_timesteps, residency=residency,',
        'audio_timestep=tat, num_train_timesteps=scheduler.num_train_timesteps,':
        'audio_timestep=tat, video_fps=video_fps, num_train_timesteps=scheduler.num_train_timesteps,',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError('Pinned acceleration patch no longer matches: ' + before)
        source = source.replace(before, after)
    return source.encode('utf-8')


def download(path):
    target = CACHE / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/{REPO}/{FREEVIDEO_COMMIT}/{path}", timeout=40) as response:
            target.write_bytes(response.read())
    return target.read_bytes()


def main():
    tree_cache = ROOT / "outputs/freevideo-plan/tree.json"
    if tree_cache.exists():
        tree = json.loads(tree_cache.read_text(encoding="utf-8-sig"))
    else:
        with urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/git/trees/{FREEVIDEO_COMMIT}?recursive=1", timeout=40) as response:
            tree = json.load(response)
    available = {entry["path"] for entry in tree["tree"] if entry["type"] == "blob"}
    todo = ["freevideo_engine/prism_runtime.py", "freevideo_engine/prism_policy.py",
            "freevideo_engine/prism_prepare.py", "freevideo_engine/prism_model/fast_vae.py"]
    seen, rows = set(), []
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        data = download(path)
        relative = Path(path).relative_to("freevideo_engine")
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        rendered = adapt(path, data)
        target.write_bytes(rendered)
        rows.append({"source": path, "sha256": hashlib.sha256(data).hexdigest(),
                     "vendored_sha256": hashlib.sha256(rendered).hexdigest()})
        package = path.split("/")[:-1]
        for node in ast.walk(ast.parse(data.decode("utf-8"))):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            base = package[:len(package) - node.level + 1]
            module = base + (node.module.split(".") if node.module else [])
            candidates = ["/".join(module) + ".py", "/".join(module) + "/__init__.py"]
            candidates += ["/".join(module + [alias.name]) + ".py" for alias in node.names]
            for candidate in candidates:
                if candidate in available and candidate.startswith("freevideo_engine/"):
                    todo.append(candidate)
        parent = target.parent
        while parent != DEST.parent:
            init = parent / "__init__.py"
            if not init.exists():
                init.write_text('"""Pinned FreeVideo Prism inference; see SOURCE.json and NOTICE."""\n', encoding="utf-8")
            parent = parent.parent
    for source, target_name in [("LICENSE", "LICENSE.Apache-2.0"),
                                ("freevideo_engine/prism_model/NOTICE", "NOTICE"),
                                ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
                                ("freevideo_engine/prism_tiers.json", "prism_tiers.json")]:
        data = download(source)
        (DEST / target_name).write_bytes(data)
        rows.append({"source": source, "sha256": hashlib.sha256(data).hexdigest()})
    (DEST / "SOURCE.json").write_text(json.dumps({"repository": REPO, "commit": FREEVIDEO_COMMIT,
        "files": sorted(rows, key=lambda r: r["source"])}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"commit": FREEVIDEO_COMMIT, "source_files": len(rows)}))


if __name__ == "__main__":
    main()
