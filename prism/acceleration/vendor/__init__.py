"""FreeVideo inference policies layered over the pinned VDN model."""
__version__ = "2026.9.16.41444"

# Release builds stamp one UTC version into both the GUI and deployed engine.
from pathlib import Path as _Path
_build_version = _Path(__file__).with_name('build-version.txt')
if _build_version.is_file():
    __version__ = _build_version.read_text(encoding='utf-8').strip()
