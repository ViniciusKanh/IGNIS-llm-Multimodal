"""Orquestra solicitações explícitas e combina compreensão visual com geração."""
from __future__ import annotations

import re
import unicodedata

from backend.engine import engine


def infer_task(prompt: str, has_image=False, has_mask=False) -> str:
    """Roteamento conservador e transparente, sem alegar um classificador treinado."""
    normalized = "".join(c for c in unicodedata.normalize("NFD", prompt.lower()) if not unicodedata.combining(c)).strip()
    if has_mask and has_image:
        return "inpainting"
    edit = re.match(r"^(edite|altere|transforme|recrie|edit|change|transform|recreate)\b", normalized)
    generate = re.match(r"^(gere|crie|desenhe|generate|create|draw)\b", normalized)
    if has_image and (edit or generate):
        return "image-to-image"
    if generate:
        return "text-to-image"
    return "vision"


def execute(prompt, task="auto", image=None, mask=None, vision="smolvlm", seed=42):
    if task == "auto":
        task = infer_task(prompt, image is not None, mask is not None)
    if task == "vision":
        return {"task": task, "result": engine.vision(prompt, [image] if image else [], key=vision)}
    if task == "describe-and-generate":
        if image is None:
            raise ValueError("A combinação de visão e geração requer imagem inicial.")
        description = engine.vision("Describe the visible composition, objects, materials and lighting in English.", [image], key=vision)
        combined = f"{prompt}. Visual reference: {description['text']}"
        result = engine.generate(combined, seed=seed)
        return {"task": task, "description": description, "result": result}
    if task not in {"text-to-image", "image-to-image", "inpainting"}:
        raise ValueError("Tarefa desconhecida.")
    return {"task": task, "result": engine.generate(prompt, mode=task, seed=seed, image=image, mask=mask)}
