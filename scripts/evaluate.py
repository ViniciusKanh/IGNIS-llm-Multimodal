"""Compara base e LoRA com prompts, parâmetros e sementes idênticos."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.common import ROOT, inside, read_json, write_json
from backend.engine import engine
from PIL import Image, ImageDraw

PROMPTS = ["A detailed oil painting of a quiet garden, soft morning light",
           "A watercolor landscape with mountains and a lake",
           "A still life painting of flowers in a ceramic vase"]
SEEDS = [42, 2026]


def evaluate(adapter="ignis-image-lora-v1/final"):
    folder = inside(ROOT / "training", adapter)
    if not (folder / "pytorch_lora_weights.safetensors").exists():
        raise FileNotFoundError("Não há adaptador treinado para comparar.")
    adapter_hash = hashlib.sha256((folder / "pytorch_lora_weights.safetensors").read_bytes()).hexdigest()
    name = folder.parent.name + "-" + folder.name
    destination = ROOT / "outputs/evaluations" / name
    destination.mkdir(parents=True, exist_ok=True)
    pairs = []
    try:
        for prompt in PROMPTS:
            for seed in SEEDS:
                base = engine.generate(prompt, seed=seed)
                tuned = engine.generate(prompt, seed=seed, adapter=adapter)
                pairs.append({"prompt": prompt, "seed": seed, "base": base, "lora": tuned})
                canvas = Image.new("RGB", (1024, 548), "white")
                for x, record in ((0, base), (512, tuned)):
                    with Image.open(ROOT / "outputs" / record["file"]) as image:
                        canvas.paste(image, (x, 36))
                ImageDraw.Draw(canvas).text((12, 10), f"Base | LoRA {adapter} | seed {seed}", fill="black")
                canvas.save(destination / f"pair-{len(pairs)}.png")
        result = {"adapter": adapter, "adapter_sha256": adapter_hash, "pairs": pairs,
                  "base_revision": read_json(ROOT / "models/manifest.json", {}).get("sd15", {}).get("revision"),
                  "training": read_json(folder.parent / "report.json"),
                  "note": "Comparação visual pareada. FID/CLIP não calculados; não inferir qualidade apenas pela loss."}
        write_json(destination / "report.json", result)
        print(f"Avaliação salva em {destination}")
        return result
    finally:
        engine.unload()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="ignis-image-lora-v1/final")
    evaluate(parser.parse_args().adapter)
