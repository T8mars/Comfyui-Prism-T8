"""Self-contained safetensors components, with bounded-memory atomic writing."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import struct
import tempfile
import mmap
import math
import warnings

import torch
from safetensors import safe_open

from . import FORMAT_VERSION

COMPONENTS = ("video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge",
              "text_encoder", "video_vae", "audio_vae")
DTYPES = {torch.float64: "F64", torch.float32: "F32", torch.float16: "F16",
          torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32",
          torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}


class TensorReader:
    """Validated safetensors reader without PyTorch's large private file mapping.

    Conversion reads one owned tensor at a time. Inference uses a read-only mmap:
    frombuffer keeps that mapping alive after this reader exits. Loaded tensors
    are inference weights and must never be modified in place.
    """
    def __init__(self, path, copy=False):
        self.path = Path(path)
        self.copy = copy
        # Rust validates offsets, shape sizes and the complete file layout.
        # The numpy backend does not invoke torch.UntypedStorage.from_file.
        with safe_open(str(self.path), framework="numpy") as validated:
            self._metadata = validated.metadata() or {}
        self._file = open(self.path, "rb")
        try:
            length = struct.unpack("<Q", self._file.read(8))[0]
            if length > 100 * 1024 * 1024:
                raise ValueError("Safetensors header exceeds supported size")
            self._header = json.loads(self._file.read(length))
            self._header.pop("__metadata__", None)
            self._data_start = 8 + length
            self._mapping = None if copy else mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self._file.close()
            raise

    def keys(self):
        return list(self._header)

    def metadata(self):
        return self._metadata

    def get_tensor(self, key):
        info = self._header[key]
        dtype = next((dtype for dtype, name in DTYPES.items() if name == info["dtype"]), None)
        if dtype is None:
            raise ValueError(f"Unsupported tensor dtype {info['dtype']} for {key}")
        count = math.prod(info["shape"])
        if count == 0:
            return torch.empty(info["shape"], dtype=dtype)
        start, end = info["data_offsets"]
        if self.copy:
            self._file.seek(self._data_start + start)
            data = bytearray(self._file.read(end - start))
            if len(data) != end - start:
                raise EOFError(f"Truncated tensor {key}")
            tensor = torch.frombuffer(data, dtype=dtype, count=count)
        else:
            # PyTorch warns because it also permits writes to buffer tensors.
            # This adapter uses them exclusively as immutable inference weights.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given buffer is not writable.*")
                tensor = torch.frombuffer(self._mapping, dtype=dtype, count=count,
                                          offset=self._data_start + start)
        return tensor.reshape(info["shape"])

    def close(self):
        self._file.close()
        # Tensor buffer owners retain the mmap object; dropping our reference
        # releases it immediately only when no returned tensor needs it anymore.
        self._mapping = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True)
class Component:
    path: Path
    kind: str
    config: dict
    metadata: dict

    @classmethod
    def inspect(cls, path, expected=None):
        path = Path(path).resolve()
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            meta = reader.metadata() or {}
        if meta.get("prism.format_version") != FORMAT_VERSION:
            raise ValueError(f"{path.name}: not a Prism standalone component. Convert official weights first.")
        kind = meta.get("prism.component")
        if kind not in COMPONENTS or (expected and kind != expected):
            raise ValueError(f"{path.name}: expected {expected}, got {kind}")
        config = json.loads(meta["prism.config"])
        if not isinstance(config, dict):
            raise ValueError("Component config must be an object")
        return cls(path, kind, config, meta)


def tokenizer_metadata(folder):
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "spiece.model", "added_tokens.json")
    assets = {name: base64.b64encode((Path(folder) / name).read_bytes()).decode("ascii")
              for name in names if (Path(folder) / name).is_file()}
    if "tokenizer.json" not in assets and "spiece.model" not in assets:
        raise ValueError(f"No tokenizer.json or spiece.model in {folder}")
    return json.dumps(assets, separators=(",", ":"))


def load_tokenizer(component):
    from transformers import T5TokenizerFast
    assets = json.loads(component.metadata["prism.tokenizer_files"])
    with tempfile.TemporaryDirectory(prefix="prism_tokenizer_") as folder:
        for name, data in assets.items():
            if name not in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                            "spiece.model", "added_tokens.json"}:
                raise ValueError(f"Unexpected tokenizer asset: {name}")
            (Path(folder) / name).write_bytes(base64.b64decode(data, validate=True))
        return T5TokenizerFast.from_pretrained(folder, local_files_only=True)


class StreamingWriter:
    """Write data incrementally, then prepend the safetensors header atomically.

    At most one tensor is in RAM. Scratch space is one output component.
    Existing files are never overwritten unless overwrite was explicitly selected.
    """
    def __init__(self, path, metadata, overwrite=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise FileExistsError(self.path)
        self.overwrite = overwrite
        self.metadata = metadata
        self.header = {}
        self.offset = 0
        self.scratch = tempfile.TemporaryDirectory(prefix=".prism_convert_", dir=self.path.parent)
        self.data = open(Path(self.scratch.name) / "data", "wb")

    def add(self, key, tensor):
        if key in self.header or key == "__metadata__":
            raise ValueError(f"Duplicate/reserved tensor key: {key}")
        if tensor.dtype not in DTYPES:
            raise ValueError(f"Unsupported source dtype {tensor.dtype} for {key}; scaled FP8 requires explicit conversion")
        tensor = tensor.detach().cpu().contiguous()
        size = tensor.numel() * tensor.element_size()
        self.header[key] = {"dtype": DTYPES[tensor.dtype], "shape": list(tensor.shape),
                            "data_offsets": [self.offset, self.offset + size]}
        self.data.write(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
        self.offset += size

    def finish(self):
        self.data.close()
        if not self.header:
            raise ValueError("Refusing to write an empty component")
        header = json.dumps({"__metadata__": self.metadata, **self.header}, separators=(",", ":")).encode("utf-8")
        header += b" " * ((-len(header)) % 8)
        temporary = Path(self.scratch.name) / "complete.safetensors"
        with open(temporary, "wb") as target, open(Path(self.scratch.name) / "data", "rb") as source:
            target.write(struct.pack("<Q", len(header)))
            target.write(header)
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        with safe_open(str(temporary), framework="pt", device="cpu") as reader:
            if len(reader.keys()) != len(self.header):
                raise RuntimeError("Written safetensors header failed validation")
        if self.overwrite:
            temporary.replace(self.path)
        else:
            # Atomic no-clobber install, also detects concurrent converters.
            import os
            os.link(temporary, self.path)
        self.scratch.cleanup()

    def close(self):
        self.data.close()
        self.scratch.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.close()
