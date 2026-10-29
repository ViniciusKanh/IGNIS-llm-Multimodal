"""Configuração pública e receitas reproduzíveis do IGNIS Image Lab."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def document(relative, default=None):
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


RELEASE = document("release.json", {})
MODEL_ID = RELEASE.get("model_id", "ViniciusKhan/ignis-image-lora-v1")
BASE_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TRAINING = document("assets/training.json", {})
CONFIG = document("assets/config.json", {})
EVALUATION = document("assets/evaluation.json", {})
SIZES = {"Quadrado · 512 × 512": (512, 512), "Paisagem · 640 × 512": (640, 512), "Retrato · 512 × 640": (512, 640)}
SAMPLERS = ["DPM++ Karras", "Euler ancestral"]
PRESETS = {"Sem estilo adicional": "", "Aquarela": "watercolor painting, translucent washes, textured paper, delicate brushwork",
           "Pintura a óleo": "oil painting, expressive brush strokes, detailed composition, natural light",
           "Fotografia": "photograph, natural lighting, realistic materials, detailed composition",
           "Ilustração científica": "scientific illustration, precise shapes, clean composition, white background",
           "Arte conceitual": "concept art, cinematic composition, atmospheric perspective, detailed environment"}
EXAMPLES = [["A quiet botanical garden at sunrise, a stone path between flowers", "Aquarela"],
            ["A mountain lake reflecting the sky, pine trees in the foreground", "Pintura a óleo"],
            ["A small observatory on a rocky island under a starry sky", "Arte conceitual"],
            ["A detailed botanical study of a fern and its leaves", "Ilustração científica"]]


def recipe(prompt, negative, seed, randomize, size, steps, guidance, lora_scale, model, sampler, preset, count=1):
    import secrets
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Descreva a imagem antes de gerar.")
    if len(prompt) > 2000 or len(negative or "") > 2000:
        raise ValueError("Use até 2.000 caracteres em cada prompt.")
    if size not in SIZES or sampler not in SAMPLERS or preset not in PRESETS:
        raise ValueError("Selecione resolução, estilo e amostrador válidos.")
    if model not in ("IGNIS LoRA", "Stable Diffusion 1.5"):
        raise ValueError("Modelo desconhecido.")
    seed = secrets.randbelow(2**31) if randomize else int(seed)
    if not 0 <= seed < 2**32 or not 5 <= int(steps) <= 40 or not 1 <= float(guidance) <= 15:
        raise ValueError("Seed, passos ou orientação fora dos limites.")
    if not 0 <= float(lora_scale) <= 1.5 or int(count) not in (1, 2):
        raise ValueError("Use LoRA de 0 a 1,5 e uma ou duas imagens.")
    width, height = SIZES[size]
    effective = prompt.strip() + (", " + PRESETS[preset] if PRESETS[preset] else "")
    return {"prompt": effective, "original_prompt": prompt.strip(), "negative_prompt": negative or "", "seed": seed,
            "width": width, "height": height, "steps": int(steps), "guidance_scale": float(guidance),
            "lora_scale": float(lora_scale), "model": model, "sampler": sampler, "preset": preset, "count": int(count),
            "model_id": MODEL_ID, "model_revision": RELEASE.get("model_revision"),
            "base_model": BASE_ID, "base_revision": RELEASE.get("base_revision")}


def restore_recipe(raw):
    if not raw or len(raw) > 100000:
        raise ValueError("Envie um JSON de receita com até 100 KB.")
    data = json.loads(raw)
    size = next((name for name, dims in SIZES.items() if list(dims) == [data.get("width"), data.get("height")]), None)
    restored = recipe(data.get("prompt", ""), data.get("negative_prompt", ""), data.get("seed", 42), False,
                      size, data.get("steps", 25), data.get("guidance_scale", 7.5), data.get("lora_scale", 1),
                      data.get("model", "IGNIS LoRA"), data.get("sampler", SAMPLERS[0]), "Sem estilo adicional")
    return restored, size
