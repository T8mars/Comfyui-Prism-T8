"""INT8 denoising with verified, previously saved BF16 native text embeddings.

The seven selected files must be the INT8 standalone bundle plus its two BF16
codecs. No BF16 encoder/transformer checkpoint or baseline directory is loaded.
--validate-only checks actual cache data and patch restoration entirely on CPU.
"""
import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import save_file

from prism.format import Component, COMPONENTS, TensorReader, load_tokenizer
from prism.settings import validate_generation
from diagnose_vae_precision import file_hash, tensor_stats


def select_int8_bundle(folder):
    parts = {}
    for kind in COMPONENTS:
        precision = "bf16" if kind.endswith("vae") else "int8_convrot"
        component = Component.inspect(Path(folder) / f"prism_alpha_{kind}_{precision}.safetensors", kind)
        if component.metadata["prism.precision"] != precision:
            raise ValueError(f"{kind}: cached-text test requires {precision}, refuses a BF16 denoiser/encoder")
        parts[kind] = component
    from prism.runtime import check_bundle
    return check_bundle(parts)


class VerifiedEmbeddingBank:
    def __init__(self, report_path, folder, text_component):
        from prism.native.diffusion.pipelines.mova_pipeline import _prompt_clean
        self.clean = _prompt_clean
        self.report_path = Path(report_path).resolve()
        self.folder = Path(folder).resolve()
        source = json.loads(self.report_path.read_text(encoding="utf-8"))
        if self.folder != (self.report_path.parent / "bf16").resolve():
            raise ValueError("Cache must be the BF16 output folder beside its actual embedding A/B report")
        if source.get("native_max_sequence_length") != 512 or source.get("bf16", {}).get("precision") != "bf16" or source["bf16"].get("quantized_linears") != 0:
            raise ValueError("Source report must contain actual 512-token, unquantized BF16 UMT5 encoding")
        labels = {"i2va", "official_negative", "white_ocean"}
        if set(source["texts"]) != labels or set(source["cleaned_texts"]) != labels or set(source["bf16"]["prompts"]) != labels:
            raise ValueError("Source report must cover exactly the three intended texts")
        if Path(source["components"]["int8"]).resolve() != text_component.path:
            raise ValueError("Source embeddings were compared against a different INT8 text component")
        for name, encoded in json.loads(text_component.metadata["prism.tokenizer_files"]).items():
            digest = hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest()
            if digest != source["official_tokenizer_files"].get(name):
                raise ValueError(f"Current embedded tokenizer asset differs from recorded official {name}")
        tokenizer = load_tokenizer(text_component)
        self.entries, self.calls = {}, []
        self.provenance = {"source_report": str(self.report_path), "source_report_sha256": file_hash(self.report_path),
                           "source_encoder_checkpoint": source["components"]["bf16"],
                           "source_encoder_checkpoint_loaded": False,
                           "source_producer_script_sha256": source["script_sha256"], "entries": {}}
        for label in sorted(labels):
            original, cleaned = source["texts"][label], source["cleaned_texts"][label]
            if not isinstance(original, str) or self.clean(original) != cleaned or cleaned in self.entries:
                raise ValueError(f"{label}: source prompt cleaning/uniqueness mismatch")
            token_path = self.report_path.parent / f"tokens-{label}.safetensors"
            with TensorReader(token_path, copy=True) as reader:
                official_ids, official_mask = reader.get_tensor("official.input_ids"), reader.get_tensor("official.attention_mask")
                for name in ("int8_embedded", "bf16_embedded"):
                    if not torch.equal(reader.get_tensor(f"{name}.input_ids"), official_ids) or not torch.equal(reader.get_tensor(f"{name}.attention_mask"), official_mask):
                        raise ValueError(f"{label}: saved token IDs/mask are inconsistent")
            actual = tokenizer(cleaned, padding="max_length", max_length=512, truncation=True,
                               add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
            if official_ids.shape != (1, 512) or official_mask.shape != (1, 512) or not torch.equal(actual.input_ids, official_ids) or not torch.equal(actual.attention_mask, official_mask):
                raise ValueError(f"{label}: current actual tokenizer IDs/mask differ from recorded official values")
            declared = source["tokenizers"][label]
            if declared["official_token_ids"] != official_ids.tolist() or declared["official_attention_mask"] != official_mask.tolist():
                raise ValueError(f"{label}: source report does not match actual saved token values")
            active = int(official_mask.sum())
            if active != declared["active_tokens"] or not torch.equal(official_mask, (torch.arange(512)[None] < active).long()):
                raise ValueError(f"{label}: expected contiguous right-padded native attention mask")
            embedding_path = self.folder / f"{label}.safetensors"
            with TensorReader(embedding_path, copy=True) as reader:
                embedding = reader.get_tensor("native_prompt_embedding")
                raw = reader.get_tensor("raw_last_hidden_state")
            shape = (1, 512, text_component.config["d_model"])
            if embedding.shape != shape or raw.shape != shape or embedding.dtype != torch.bfloat16 or raw.dtype != torch.bfloat16 or not torch.isfinite(embedding).all() or not torch.isfinite(raw).all():
                raise ValueError(f"{label}: expected finite BF16 native/raw tensors {shape}")
            if not torch.equal(embedding[:, :active], raw[:, :active]) or not (embedding[:, active:] == 0).all():
                raise ValueError(f"{label}: cached native embedding has incorrect padding/prefix")
            if tensor_stats(embedding) != source["bf16"]["prompts"][label]["native"] or tensor_stats(raw) != source["bf16"]["prompts"][label]["raw"]:
                raise ValueError(f"{label}: actual embedding statistics differ from its encoding report")
            digest = file_hash(embedding_path)
            self.entries[cleaned] = {"label": label, "embedding": embedding, "sha256": digest}
            self.provenance["entries"][label] = {"text": original, "cleaned_text": cleaned,
                "embedding_file": str(embedding_path), "embedding_sha256": digest,
                "token_file": str(token_path.resolve()), "token_sha256": file_hash(token_path),
                "shape": list(shape), "dtype": str(embedding.dtype), "active_tokens": active,
                "actual_current_tokenizer_equals_saved_official": True, "actual_embedding_matches_source_statistics": True}

    def labels_for(self, prompts):
        texts = [prompts] if isinstance(prompts, str) else prompts
        if not isinstance(texts, (list, tuple)) or not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("Cached-text diagnostic requires a nonempty text prompt/batch")
        entries = []
        for text in texts:
            cleaned = self.clean(text)
            if cleaned not in self.entries:
                raise ValueError("Prompt was not actually encoded in the verified BF16 source cache")
            entries.append(self.entries[cleaned])
        return entries

    def method(self):
        bank = self
        def cached(pipe, prompt, num_videos_per_prompt=1, max_sequence_length=512, device=None, dtype=None):
            if max_sequence_length != 512 or not isinstance(num_videos_per_prompt, int) or isinstance(num_videos_per_prompt, bool) or num_videos_per_prompt < 1:
                raise ValueError("Cached embeddings support native length 512 and positive integer repeat")
            device, dtype = device or pipe.device, dtype or pipe.text_encoder.dtype
            entries = bank.labels_for(prompt)
            values = torch.cat([entry["embedding"] for entry in entries], dim=0).to(device=device, dtype=dtype)
            output = values.repeat(1, num_videos_per_prompt, 1).view(len(entries) * num_videos_per_prompt, 512, values.shape[-1])
            bank.calls.append({"labels": [entry["label"] for entry in entries], "source_sha256": [entry["sha256"] for entry in entries],
                               "device": str(device), "dtype": str(dtype), "num_videos_per_prompt": num_videos_per_prompt,
                               "shape": list(output.shape)})
            return output
        return cached


@contextmanager
def patched_native_text(bank, latent_target=None):
    from diffusers import AutoencoderKLWan
    from transformers import UMT5EncoderModel
    from prism.native.diffusion.pipelines.mova_pipeline import MOVAPipeline
    originals = [(MOVAPipeline, "_get_t5_prompt_embeds", MOVAPipeline._get_t5_prompt_embeds),
                 (UMT5EncoderModel, "forward", UMT5EncoderModel.forward),
                 (AutoencoderKLWan, "decode", AutoencoderKLWan.decode)]
    def forbidden_forward(*args, **kwargs):
        raise RuntimeError("Cached-text diagnostic must never execute a UMT5 encoder forward")
    def retained_decode(model, latents, *args, **kwargs):
        save_file({"latents": latents.detach().float().cpu().contiguous()}, str(latent_target))
        return originals[2][2](model, latents, *args, **kwargs)
    try:
        MOVAPipeline._get_t5_prompt_embeds = bank.method()
        UMT5EncoderModel.forward = forbidden_forward
        if latent_target is not None:
            AutoencoderKLWan.decode = retained_decode
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def cpu_structure_check(bank):
    from types import SimpleNamespace
    from diffusers import AutoencoderKLWan
    from transformers import UMT5EncoderModel
    from prism.native.diffusion.pipelines.mova_pipeline import MOVAPipeline
    originals = (MOVAPipeline._get_t5_prompt_embeds, UMT5EncoderModel.forward, AutoencoderKLWan.decode)
    pipe = object.__new__(MOVAPipeline)
    pipe._device = torch.device("cpu")
    pipe.text_encoder = SimpleNamespace(dtype=torch.bfloat16)
    texts = list(bank.entries)
    with patched_native_text(bank, latent_target=Path("unused_cpu_validation_latent.safetensors")):
        assert originals != (MOVAPipeline._get_t5_prompt_embeds, UMT5EncoderModel.forward, AutoencoderKLWan.decode)
        for text in texts:
            assert torch.equal(pipe._get_t5_prompt_embeds(text), bank.entries[text]["embedding"])
        repeated = pipe._get_t5_prompt_embeds(texts[:2], num_videos_per_prompt=2, dtype=torch.float32)
        expected = torch.cat([bank.entries[text]["embedding"].float().repeat(2, 1, 1) for text in texts[:2]])
        assert torch.equal(repeated, expected)
        try:
            UMT5EncoderModel.forward(None)
            raise AssertionError("Encoder forward prohibition was not active")
        except RuntimeError as error:
            assert "must never execute" in str(error)
        for kwargs in ({"prompt": "this text was never encoded"}, {"prompt": texts[0], "max_sequence_length": 256}):
            try:
                pipe._get_t5_prompt_embeds(**kwargs)
                raise AssertionError("Unsupported cached input was silently accepted")
            except ValueError:
                pass
    assert originals == (MOVAPipeline._get_t5_prompt_embeds, UMT5EncoderModel.forward, AutoencoderKLWan.decode)
    try:
        with patched_native_text(bank, latent_target=Path("unused_cpu_validation_latent.safetensors")):
            raise RuntimeError("injected cancellation for patch restoration")
    except RuntimeError:
        pass
    assert originals == (MOVAPipeline._get_t5_prompt_embeds, UMT5EncoderModel.forward, AutoencoderKLWan.decode)
    assert not torch.cuda.is_initialized(), "CPU cache validation must not initialize CUDA"
    bank.calls.clear()
    return {"actual_cache_shape_padding_tokens_verified": True, "cpu_single_batch_repeat_dtype_exact": True,
            "unknown_prompt_and_length_rejected": True, "encoder_forward_forbidden": True,
            "patch_restored_after_success_and_exception": True, "gpu_or_model_weights_loaded": False,
            "torch_cuda_initialized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="models/standalone")
    parser.add_argument("--embedding-report", default="outputs/umt5_embedding_precision/report.json")
    parser.add_argument("--embedding-dir", default="outputs/umt5_embedding_precision/bf16")
    parser.add_argument("--source-report", default="outputs/quality_int8_latent_probe/diagnostic.json")
    parser.add_argument("--image", default="examples/prism_official_case5.png")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--validate-only", action="store_true", help="Actual cache checks and restoration tests only, CPU")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source_path = Path(args.source_report).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    settings = dict(source["sampler"])
    overrides = {}
    for key in ("width", "height", "num_frames", "vae_tiling"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = settings[key] = value
    settings = validate_generation(settings)
    parts = select_int8_bundle(args.models)
    bank = VerifiedEmbeddingBank(args.embedding_report, args.embedding_dir, parts["text_encoder"])
    requested = [settings["prompt"], settings["negative_prompt"]]
    if settings["audio_prompt"]:
        requested.append(settings["audio_prompt"])
    bank.labels_for(requested)
    validation = cpu_structure_check(bank)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Standalone INT8 denoising; only native text embedding outputs replaced from verified BF16 cache",
              "script_sha256": file_hash(__file__), "sampler": settings, "overrides": overrides,
              "source_report": str(source_path), "source_report_sha256": file_hash(source_path),
              "embedding_source": bank.provenance, "cpu_validation": validation,
              "replacement_scope": ["MOVAPipeline._get_t5_prompt_embeds outputs for exactly three verified texts",
                                    "UMT5EncoderModel.forward assertion prevents live text encoding",
                                    "AutoencoderKLWan.decode observer saves denormalized latent"],
              "components": {kind: {"path": str(value.path), "precision": value.metadata["prism.precision"]} for kind, value in parts.items()},
              "bf16_encoder_or_transformer_checkpoint_loaded": False, "baseline_directory_enabled": False,
              "quality_acceptance": "not established by CPU validation or finite values"}
    (output / "diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if args.validate_only:
        report["status"] = "cpu-validated-no-sampling"
        (output / "diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Verified actual BF16 cache on CPU; no weights/CUDA/sampling: {output / 'diagnostic.json'}", flush=True)
        return
    if not torch.cuda.is_available():
        parser.error("Sampling requires the CUDA GPU after the parent releases it")
    from PIL import Image
    from prism.runtime import run
    from prism.media import save_video
    from review_samples import decode
    from prism.native.diffusion.pipelines.mova_pipeline import MOVAPipeline
    from transformers import UMT5EncoderModel
    from diffusers import AutoencoderKLWan
    originals = (MOVAPipeline._get_t5_prompt_embeds, UMT5EncoderModel.forward, AutoencoderKLWan.decode)
    image = Image.open(args.image).convert("RGB") if settings["mode"] == "i2va" else None
    if image is not None:
        report["reference_image"] = str(Path(args.image).resolve())
        report["reference_sha256"] = file_hash(args.image)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        with patched_native_text(bank, output / "video-latents.safetensors"):
            frames, audio, fps = run(parts, image, settings, sparse={},
                                     callback=lambda step, total: print(f"INT8 cached-text step {step + 1}/{total}", flush=True))
        save_video(frames, audio, fps, output / "sample.mp4")
        report["media"] = decode(output / "sample.mp4", settings, output / "review")
        report["status"] = "sampled-requires-real-visual-audio-inspection"
    except BaseException as error:
        report["status"], report["error"], report["traceback"] = "failed", str(error), traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.monotonic() - start
        report["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        report["actual_replacement_calls"] = bank.calls
        report["monkey_patches_restored"] = originals == (MOVAPipeline._get_t5_prompt_embeds, UMT5EncoderModel.forward, AutoencoderKLWan.decode)
        report["source_embeddings_unchanged"] = all(file_hash(value["embedding_file"]) == value["embedding_sha256"] for value in bank.provenance["entries"].values())
        (output / "diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"INT8 cached-BF16-text diagnostic: {output / 'diagnostic.json'}", flush=True)


if __name__ == "__main__":
    main()
