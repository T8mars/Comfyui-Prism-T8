"""Validate delivered canvas files against a real ComfyUI object_info response."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prism.settings import validate_generation, validate_sparse


def serialized_widgets(node, definition):
    """Read the actual frontend array, rather than an auxiliary named copy."""
    values = iter(node.get("widgets_values") or [])
    result = {}
    for section in ("required", "optional"):
        specs = definition["input"].get(section, {})
        order = definition.get("input_order", {}).get(section, list(specs))
        for name in order:
            spec = specs[name]
            if isinstance(spec[0], list) or spec[0] in ("STRING", "INT", "FLOAT", "BOOLEAN"):
                try:
                    result[name] = next(values)
                    # ComfyUI adds the seed control widget to this node's INT seed.
                    if node["type"] in ("PrismNativeSampler", "PrismAcceleratedSampler") and name == "seed":
                        control = next(values)
                        assert control in ("fixed", "increment", "decrement", "randomize")
                except StopIteration as error:
                    raise AssertionError(f"Missing serialized widget: {node['type']}.{name}") from error
    return result


def validate(path, definitions):
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow["version"] == 0.4 and isinstance(workflow["nodes"], list), "Not canvas JSON"
    nodes = {str(n["id"]): n for n in workflow["nodes"]}
    links = {str(link[0]): link for link in workflow["links"]}
    assert len(nodes) == len(workflow["nodes"]) and len(links) == len(workflow["links"])
    for link in links.values():
        _, source_id, source_slot, target_id, target_slot, socket = link
        source, target = nodes[str(source_id)], nodes[str(target_id)]
        output, input_ = source["outputs"][source_slot], target["inputs"][target_slot]
        assert output["type"] == socket == input_["type"], f"Wrong socket: {link}"
        assert str(input_["link"]) == str(link[0]), f"Wrong input link: {link}"
        assert link[0] in output["links"], f"Wrong output link: {link}"
    for node in nodes.values():
        definition = definitions[node["type"]]
        named = serialized_widgets(node, definition)
        inputs = {i["name"]: i for i in node.get("inputs", [])}
        for name, spec in definition["input"].get("required", {}).items():
            connected = inputs.get(name, {}).get("link") is not None
            if connected:
                assert str(inputs[name]["link"]) in links
            elif isinstance(spec[0], list):
                assert named[name] in spec[0], f"Invalid combo {node['type']}.{name}: {named.get(name)}"
            elif spec[0] in ("STRING", "INT", "FLOAT", "BOOLEAN"):
                assert name in named, f"Missing widget {node['type']}.{name}"
            else:
                raise AssertionError(f"Missing required socket {node['type']}.{name}")
        if node["type"] == "PrismNativeSampler":
            settings = {k: v for k, v in named.items() if k != "control_after_generate"}
            validate_generation(settings)
            assert node["outputs"][0]["links"] and node["outputs"][1]["links"] and node["outputs"][2]["links"]
        if node["type"] == "PrismSparseOptions":
            validate_sparse({"enable_bsa": named["enable_bsa"], "bsa_sparsity": named["sparsity"],
                "enable_ivpq_dynamic_block": named["dynamic_blocks"] == "ivpq",
                "enable_penalty_dynamic_block": named["dynamic_blocks"] == "penalty",
                "sparse_high_noise_only": named["high_noise_only"], **json.loads(named["advanced_json"])})
    output_types = {n["type"] for n in nodes.values() if definitions[n["type"]]["output_node"]}
    assert {"SaveImage", "SaveAudio", "PrismSaveVideo"} <= output_types
    return {"file": path.name, "nodes": len(nodes), "links": len(links), "valid": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8198")
    parser.add_argument("--object-info", help="Saved real ComfyUI object_info JSON")
    parser.add_argument("--output", default="outputs/comfyui-review/canvas-validation.json")
    args = parser.parse_args()
    if args.object_info:
        definitions = json.loads(Path(args.object_info).read_text(encoding="utf-8-sig"))
    else:
        with urllib.request.urlopen(args.server.rstrip("/") + "/object_info") as response:
            definitions = json.load(response)
    result = {"scope": "canvas schema, registered node inputs, widget values and all bidirectional links",
              "source": args.object_info or args.server, "workflows":
              [validate(path, definitions) for path in sorted((ROOT / "examples").glob("0*.json"))]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
