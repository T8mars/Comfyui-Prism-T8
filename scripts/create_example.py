"""Generate complete ComfyUI canvas workflows, not API prompt JSON."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prism.settings import GENERATION_DEFAULTS

LOADERS = {"video_dit": "PrismVideoDiTLoader", "video_dit_2": "PrismLowNoiseDiTLoader",
           "audio_dit": "PrismAudioDiTLoader", "dual_tower_bridge": "PrismBridgeLoader",
           "text_encoder": "PrismTextEncoderLoader", "video_vae": "PrismVideoVAELoader",
           "audio_vae": "PrismAudioVAELoader"}
PROMPT = ("Medium shot on a fishing boat in daylight. A smiling man wearing a gray cap, "
          "blue sunglasses and a gray shirt holds a large grouper. The boat gently rocks, "
          "the fish moves slightly in his hands, and sunlight glints on the ocean. "
          "The camera stays steady. <sfx>Gentle waves and water lapping around the boat.</sfx>")


class Canvas:
    def __init__(self):
        self.nodes, self.links = [], []

    def node(self, kind, title, pos, size, sockets=(), outputs=(), widgets=()):
        node = {"id": len(self.nodes) + 1, "type": kind, "title": title,
                "pos": list(pos), "size": list(size), "flags": {},
                "order": len(self.nodes), "mode": 0,
                "inputs": [{"name": n, "type": t, "link": None} for n, t in sockets],
                "outputs": [{"name": n, "type": t, "links": None} for n, t in outputs],
                "properties": {"Node name for S&R": kind},
                "widgets_values": [v for n, v in widgets]}
        self.nodes.append(node)
        return node

    def connect(self, source, slot, target, name):
        index = next(i for i, item in enumerate(target["inputs"]) if item["name"] == name)
        link, socket = len(self.links) + 1, source["outputs"][slot]
        if socket["type"] != target["inputs"][index]["type"]:
            raise ValueError(f"Socket mismatch: {name}")
        self.links.append([link, source["id"], slot, target["id"], index, socket["type"]])
        socket["links"] = (socket["links"] or []) + [link]
        target["inputs"][index]["link"] = link

    def serialize(self, name):
        groups = [("1. Standalone native components", [20, 10, 470, 850]),
                  ("2. Reference / sparse options", [510, 10, 460, 850]),
                  ("3. Joint video / audio sampling", [980, 10, 530, 1150]),
                  ("4. Frames / audio / MP4", [1520, 10, 490, 1150])]
        return {"last_node_id": len(self.nodes), "last_link_id": len(self.links),
                "nodes": self.nodes, "links": self.links,
                "groups": [{"title": t, "bounding": b, "color": "#3f789e", "font_size": 22, "flags": {}}
                           for t, b in groups], "config": {},
                "extra": {"ds": {"scale": 0.64, "offset": [30, 30]},
                          "prism": {"name": name, "format": "ComfyUI canvas",
                                    "reference_source": "Tencent-Hunyuan/Prism assets/ti2va_cases/case-5"}},
                "version": 0.4}


def workflow(name, settings, sparse=False, white=False):
    c, parts = Canvas(), {}
    for index, (kind, loader) in enumerate(LOADERS.items()):
        precision = "bf16" if kind.endswith("vae") else "int8_convrot"
        parts[kind] = c.node(loader, kind.replace("_", " ").title(), (50, 60 + index * 104), (410, 80),
            outputs=[(kind, "PRISM_" + kind.upper())],
            widgets=[("model_name", f"prism_alpha_{kind}_{precision}.safetensors")])
    pipeline = c.node("PrismNativePipeline", "Official Native Pipeline", (1010, 60), (460, 195),
        sockets=[(kind, "PRISM_" + kind.upper()) for kind in LOADERS],
        outputs=[("PRISM_PIPELINE", "PRISM_PIPELINE")])
    for kind, part in parts.items():
        c.connect(part, 0, pipeline, kind)
    image = None
    if not white:
        image = c.node("LoadImage", "Reference Image (official case 5)", (540, 60), (400, 360),
            outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
            widgets=[("image", "prism_official_case5.png"), ("upload", "image")])
    options = c.node("PrismSparseOptions", "Native Sparse Attention Options", (540, 460), (400, 290),
        outputs=[("PRISM_SPARSE_OPTIONS", "PRISM_SPARSE_OPTIONS")],
        widgets=[("enable_bsa", sparse), ("sparsity", .75), ("dynamic_blocks", "ivpq" if sparse else "fixed"),
                 ("high_noise_only", False), ("advanced_json", json.dumps({"enable_bsa_v2a": sparse,
                    "bsa_v2a_sparsity": .75}, indent=2))])
    values = {**GENERATION_DEFAULTS, "prompt": PROMPT, **settings}
    if white:
        values.update(mode="t2va_white_reference", prompt="A wide view of turquoise ocean waves gently rolling onto a sandy beach in daylight. <sfx>Gentle surf and distant seabirds.</sfx>")
    widgets = []
    for key, value in values.items():
        widgets.append((key, value))
        if key == "seed":
            widgets.append(("control_after_generate", "fixed"))
    sampler = c.node("PrismNativeSampler", "Native Joint Video + Audio Sampler", (1010, 300), (460, 820),
        sockets=[("pipeline", "PRISM_PIPELINE"), ("reference_image", "IMAGE"), ("sparse_options", "PRISM_SPARSE_OPTIONS")],
        outputs=[("frames", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT")], widgets=widgets)
    c.connect(pipeline, 0, sampler, "pipeline")
    if image:
        c.connect(image, 0, sampler, "reference_image")
    c.connect(options, 0, sampler, "sparse_options")
    prefix = "Prism/" + name
    frames = c.node("SaveImage", "Save Every Frame (workflow embedded)", (1550, 60), (420, 300),
        sockets=[("images", "IMAGE")], widgets=[("filename_prefix", prefix + "/frame")])
    audio = c.node("SaveAudio", "Save Audio (48 kHz FLAC)", (1550, 410), (420, 200),
        sockets=[("audio", "AUDIO")], outputs=[("audio", "AUDIO")], widgets=[("filename_prefix", prefix + "/audio")])
    video = c.node("PrismSaveVideo", "Save MP4 + Audio / Canvas Preview", (1550, 670), (420, 380),
        sockets=[("frames", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT")], outputs=[("video_path", "STRING")],
        widgets=[("fps", values["fps"]), ("filename_prefix", prefix + "/video")])
    video["inputs"][2]["widget"] = {"name": "fps"}
    for slot, output, input_name in [(0, frames, "images"), (1, audio, "audio"),
                               (0, video, "frames"), (1, video, "audio"), (2, video, "fps")]:
        c.connect(sampler, slot, output, input_name)
    return c.serialize(name)


def main():
    folder = ROOT / "examples"
    folder.mkdir(exist_ok=True)
    presets = {"01_native_i2va": workflow("01_native_i2va", {}),
        "02_native_i2va_kitchen_bsa": workflow("02_native_i2va_kitchen_bsa", {"int8_backend": "kitchen"}, sparse=True),
        "03_native_t2va_white_reference": workflow("03_native_t2va_white_reference", {}, white=True),
        "04_native_i2va_720p": workflow("04_native_i2va_720p", {"width": 1280, "height": 720, "num_frames": 205, "vae_tiling": True}),
        "05_native_i2va_validation": workflow("05_native_i2va_validation", {"int8_backend": "kitchen",
            "width": 848, "height": 480, "num_frames": 49})}
    for name, data in presets.items():
        (folder / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(presets)} canvas workflows to {folder}")


if __name__ == "__main__":
    main()
