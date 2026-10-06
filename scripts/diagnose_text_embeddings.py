"""Actual tokenizer and UMT5 INT8/BF16 embedding A/B; no diffusion sampling.

Use --tokenizer-only for a CPU comparison while the sampling GPU is busy.
The full mode encodes each prompt separately, as the native pipeline does.
Numerical similarity/finite values do not establish generated image quality.
"""
import argparse
import ast
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import save_file

from prism.format import Component, load_tokenizer
from prism.loading import load_component
from prism.quantization import ConvRotLinear
from diagnose_vae_precision import file_hash, tensor_stats


def canvas_prompt(path):
    canvas = json.loads(Path(path).read_text(encoding="utf-8"))
    node = next(item for item in canvas["nodes"] if item["type"] == "PrismNativeSampler")
    widget_names = [item["widget"]["name"] for item in node["inputs"] if "widget" in item]
    prompt = node["widgets_values"][widget_names.index("prompt")]
    if not isinstance(prompt, str):
        raise ValueError("Canvas prompt must be text")
    return prompt


def official_negative(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "NEGATIVE_PROMPT" for target in statement.targets):
            result = ast.literal_eval(statement.value)
            if isinstance(result, str):
                return result
    raise ValueError("Pinned official source has no literal NEGATIVE_PROMPT")


def embedding_metrics(candidate, reference, ids, mask):
    a, b = candidate.double()[0], reference.double()[0]
    difference = a - b
    a_norm, b_norm = torch.linalg.vector_norm(a, dim=-1), torch.linalg.vector_norm(b, dim=-1)
    d_norm = torch.linalg.vector_norm(difference, dim=-1)
    dot = (a * b).sum(dim=-1)
    used = mask[0].bool()
    aa, bb, dd = a[used], b[used], difference[used]
    denominator = torch.linalg.vector_norm(bb).item()
    metrics = {"active_tokens": int(used.sum()), "max_abs": dd.abs().max().item(),
               "mean_abs": dd.abs().mean().item(), "relative_l2": torch.linalg.vector_norm(dd).item() / max(denominator, 1e-30),
               "cosine": ((aa * bb).sum() / (torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)).clamp_min(1e-30)).item(),
               "per_token": []}
    for index in range(a.shape[0]):
        metrics["per_token"].append({"token_index": index, "token_id": int(ids[0, index]), "active": bool(used[index]),
            "max_abs": difference[index].abs().max().item(), "difference_l2": d_norm[index].item(),
            "relative_l2": (d_norm[index] / b_norm[index]).item() if b_norm[index] > 0 else (0. if d_norm[index] == 0 else None),
            "cosine": (dot[index] / (a_norm[index] * b_norm[index])).item() if a_norm[index] > 0 and b_norm[index] > 0 else None})
    return metrics


def encode_component(component, tokens, output, device, backend):
    start = time.monotonic()
    model = load_component(component, dtype=torch.bfloat16, backend=backend).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    results, embeddings = {}, {}
    try:
        for label, tokenized in tokens.items():
            ids, mask = tokenized["input_ids"], tokenized["attention_mask"]
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                hidden = model(ids.to(device), mask.to(device)).last_hidden_state
            raw = hidden.detach().cpu().contiguous()
            if not torch.isfinite(raw).all():
                raise RuntimeError(f"{component.metadata['prism.precision']} {label} non-finite hidden state")
            native = raw.clone()
            # Native _get_t5_prompt_embeds discards masked positions and appends zeros.
            native[:, int(mask.gt(0).sum()):] = 0
            save_file({"raw_last_hidden_state": raw, "native_prompt_embedding": native}, str(output / f"{label}.safetensors"))
            results[label] = {"raw": tensor_stats(raw), "native": tensor_stats(native),
                              "padding_zero": bool((native[:, int(mask.gt(0).sum()):] == 0).all())}
            embeddings[label] = native
        torch.cuda.synchronize(device)
        return {"seconds": time.monotonic() - start, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30,
                "dtype": str(model.dtype), "autocast": False, "class": type(model).__name__,
                "precision": component.metadata["prism.precision"],
                "quantized_linears": sum(isinstance(child, ConvRotLinear) for child in model.modules()),
                "prompts": results}, embeddings
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int8", default="models/standalone/prism_alpha_text_encoder_int8_convrot.safetensors")
    parser.add_argument("--bf16", default="models/baseline_bf16/prism_alpha_text_encoder_bf16.safetensors")
    parser.add_argument("--official-tokenizer", default="checkpoints/official/pretrained_models/MOVA-360p/tokenizer")
    parser.add_argument("--official-source", default=".research/Prism/hymm/sample/sample_mova_single.py")
    parser.add_argument("--source-report", help="Use its actual sampler I2VA prompt when available")
    parser.add_argument("--i2va-workflow", default="examples/05_native_i2va_validation.json")
    parser.add_argument("--white-workflow", default="examples/03_native_t2va_white_reference.json")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--tokenizer-only", action="store_true", help="CPU token IDs/masks check; no weights/GPU")
    parser.add_argument("--backend", choices=("portable", "kitchen"), default="portable")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    from transformers import T5TokenizerFast, UMT5Config
    from prism.native.diffusion.pipelines.mova_pipeline import _prompt_clean
    int8, bf16 = Component.inspect(args.int8, "text_encoder"), Component.inspect(args.bf16, "text_encoder")
    raw_config_equal = int8.config == bf16.config
    raw_config_differences = {key: [int8.config.get(key, "<missing>"), bf16.config.get(key, "<missing>")]
                              for key in int8.config.keys() | bf16.config.keys()
                              if int8.config.get(key, "<missing>") != bf16.config.get(key, "<missing>")}
    # Transformers may serialize attn_implementation:null instead of omitting it.
    # Resolve both with the actual loader class; preserve raw differences as evidence.
    resolved_int8 = UMT5Config.from_dict(dict(int8.config))
    resolved_bf16 = UMT5Config.from_dict(dict(bf16.config))
    if resolved_int8.to_dict() != resolved_bf16.to_dict() or json.loads(int8.metadata["prism.tokenizer_files"]) != json.loads(bf16.metadata["prism.tokenizer_files"]):
        parser.error("Actual resolved INT8/BF16 configs and embedded tokenizer assets must be identical")
    if int8.metadata["prism.precision"] != "int8_convrot" or bf16.metadata["prism.precision"] != "bf16":
        parser.error("Expected an INT8 ConvRot encoder and its standalone BF16 baseline")
    source = json.loads(Path(args.source_report).read_text(encoding="utf-8")) if args.source_report else None
    texts = {"i2va": source["sampler"]["prompt"] if source else canvas_prompt(args.i2va_workflow),
             "official_negative": official_negative(args.official_source), "white_ocean": canvas_prompt(args.white_workflow)}
    cleaned = {label: _prompt_clean(value) for label, value in texts.items()}
    tokenizers = {"official": T5TokenizerFast.from_pretrained(args.official_tokenizer, local_files_only=True),
                  "int8_embedded": load_tokenizer(int8), "bf16_embedded": load_tokenizer(bf16)}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokens, comparisons = {}, {}
    for label, text in cleaned.items():
        tokenized = {name: tokenizer(text, padding="max_length", max_length=512, truncation=True,
                                    add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
                     for name, tokenizer in tokenizers.items()}
        values = {}
        comparisons[label] = {}
        for name, result in tokenized.items():
            for key in ("input_ids", "attention_mask"):
                values[f"{name}.{key}"] = result[key]
                comparisons[label][f"{name}.{key}_equals_official"] = bool(torch.equal(result[key], tokenized["official"][key]))
        save_file(values, str(output / f"tokens-{label}.safetensors"))
        tokens[label] = tokenized["official"]
        comparisons[label]["active_tokens"] = int(tokenized["official"]["attention_mask"].sum())
        comparisons[label]["official_token_ids"] = tokenized["official"]["input_ids"].tolist()
        comparisons[label]["official_attention_mask"] = tokenized["official"]["attention_mask"].tolist()
    report = {"scope": "UMT5/tokenizer components only; no denoiser sampling", "script_sha256": file_hash(__file__),
              "texts": texts, "cleaned_texts": cleaned, "native_max_sequence_length": 512, "tokenizers": comparisons,
              "checkpoint_config_json_equal": raw_config_equal, "checkpoint_config_differences": raw_config_differences,
              "resolved_umt5_config_equal": True,
              "resolved_attention_selectors": [resolved_int8._attn_implementation, resolved_bf16._attn_implementation],
              "i2va_source": args.source_report or args.i2va_workflow, "white_source": args.white_workflow,
              "negative_source": args.official_source, "official_tokenizer": str(Path(args.official_tokenizer).resolve()),
              "official_tokenizer_files": {str(path.name): file_hash(path) for path in Path(args.official_tokenizer).iterdir() if path.is_file()},
              "components": {"int8": str(int8.path), "bf16": str(bf16.path)}, "quality_acceptance": "No image-quality inference from embeddings"}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if not all(value for checks in comparisons.values() for key, value in checks.items() if key.endswith("equals_official")):
        raise RuntimeError("Actual embedded tokenizer IDs/masks differ from original official tokenizer")
    if args.tokenizer_only:
        print(f"CPU actual official/embedded IDs and masks all equal: {output / 'report.json'}", flush=True)
        return
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("Run embedding A/B only after the sampling GPU is released")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["tf32_matmul"] = report["tf32_cudnn"] = False
    embeddings = {}
    for name, component in (("int8", int8), ("bf16", bf16)):
        folder = output / name
        folder.mkdir()
        print(f"Encoding actual three prompts with {name} UMT5 only", flush=True)
        report[name], embeddings[name] = encode_component(component, tokens, folder, device, args.backend)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report["comparison"] = {label: embedding_metrics(embeddings["int8"][label], embeddings["bf16"][label],
                                                     tokens[label]["input_ids"], tokens[label]["attention_mask"])
                            for label in tokens}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Embedding-only A/B report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
