"""Package only the complete canvas workflows, instructions and reference image."""
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "examples"
    workflows = sorted(folder.glob("0*.json"))
    if len(workflows) != 9:
        raise ValueError("Expected five native and four FreeVideo canvas workflows")
    for path in workflows:
        canvas = json.loads(path.read_text(encoding="utf-8"))
        if canvas.get("version") != 0.4 or not canvas.get("nodes") or "links" not in canvas:
            raise ValueError(f"Not a canvas workflow: {path.name}")
    target = folder / "Prism-canvas-workflows.zip"
    files = workflows + [folder / "README.md", folder / "prism_official_case5.png"]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Build alongside the final archive so installation is an atomic replace.
    # A missing input or interrupted write must preserve an existing package.
    with tempfile.NamedTemporaryFile(prefix=".prism_workflows_", suffix=".zip.part",
                                     dir=folder, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, arcname=path.name)
        with zipfile.ZipFile(temporary_path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Workflow archive CRC validation failed")
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved {target}: {len(files)} files, {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
