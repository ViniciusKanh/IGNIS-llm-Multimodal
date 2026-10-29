"""Diretórios, registros atômicos e configuração local de caches."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key, value in {
    "HF_HOME": ROOT / ".cache/huggingface",
    "HF_HUB_CACHE": ROOT / ".cache/huggingface/hub",
    "HF_DATASETS_CACHE": ROOT / ".cache/huggingface/datasets",
    "TORCH_HOME": ROOT / ".cache/torch",
    "XDG_CACHE_HOME": ROOT / ".cache",
    "TMPDIR": ROOT / ".cache/tmp",
    "TEMP": ROOT / ".cache/tmp",
    "TMP": ROOT / ".cache/tmp",
}.items():
    os.environ[key] = str(value)
    Path(value).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
for name in ("models", "datasets", "training", "outputs", "logs"):
    (ROOT / name).mkdir(exist_ok=True)

MODELS = {
    "smolvlm": {"repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "license": "apache-2.0", "kind": "vision"},
    "qwen": {"repo": "Qwen/Qwen3-VL-2B-Instruct", "license": "apache-2.0", "kind": "vision"},
    "sd15": {"repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "license": "creativeml-openrail-m", "kind": "image"},
}


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, data):
    """Substitui um JSON completo, evitando estado parcial após interrupção."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def inside(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Caminho fora do diretório permitido.")
    return path


def model_path(key: str) -> Path:
    if key not in MODELS:
        raise ValueError("Modelo desconhecido.")
    record = read_json(ROOT / "models/manifest.json", {}).get(key, {})
    if record.get("status") != "installed":
        raise FileNotFoundError(f"Modelo {key} não instalado. Execute scripts/download-models.ps1 -Models {key}.")
    return ROOT / "models" / key
