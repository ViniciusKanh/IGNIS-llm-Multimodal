"""Baixa uma única variante Safetensors e fixa a revisão no manifesto."""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.common import MODELS, ROOT, read_json, write_json
from filelock import FileLock


def select_files(names: list[str], key: str) -> list[str]:
    selected = []
    for name in names:
        lower = name.lower()
        if lower.endswith((".bin", ".ckpt", ".onnx", ".h5", ".msgpack", ".pt", ".pth")):
            continue
        if key == "sd15":
            if "/" not in name and name not in ("model_index.json", "README.md", "LICENSE", "LICENSE.md"):
                continue
            if "/" in name and name.split("/")[0] not in {"unet", "vae", "text_encoder", "tokenizer", "scheduler", "feature_extractor", "safety_checker"}:
                continue
            if lower.endswith(".safetensors"):
                # Uma variante por componente; usa FP16 se existir no repositório.
                fp16 = name.replace(".safetensors", ".fp16.safetensors")
                if ".fp16." not in name and fp16 in names:
                    continue
                if any(mark in lower for mark in ("non_ema", "non-ema", "flax")):
                    continue
        else:
            if "/" in name or any(mark in lower for mark in ("onnx", "gguf", "tflite")):
                continue
        if lower.endswith((".json", ".safetensors", ".txt", ".model", ".jinja", ".md")) or "license" in lower:
            selected.append(name)
    if not any(n.endswith(".safetensors") for n in selected):
        raise RuntimeError("Nenhum peso Safetensors utilizável encontrado.")
    return selected


def download(key: str):
    from huggingface_hub import HfApi, snapshot_download
    with FileLock(str(ROOT / "models/download.lock"), timeout=0):
        info = HfApi().model_info(MODELS[key]["repo"], files_metadata=True)
        card = dict(info.card_data or {})
        actual_license = card.get("license")
        if actual_license != MODELS[key]["license"]:
            raise RuntimeError(f"Licença alterada ou ausente: {actual_license!r}; revisar antes do download.")
        files = select_files([x.rfilename for x in info.siblings], key)
        target = ROOT / "models" / key
        target.mkdir(exist_ok=True)
        sizes = {x.rfilename: x.size or 0 for x in info.siblings}
        missing = sum(sizes[n] for n in files if not (target / n).exists() or (target / n).stat().st_size != sizes[n])
        if shutil.disk_usage(ROOT).free < missing + 3 * 1024**3:
            raise RuntimeError("Espaço insuficiente: mantenha também 3 GB livres para operação.")
        manifest = read_json(ROOT / "models/manifest.json", {})
        manifest[key] = {**MODELS[key], "revision": info.sha, "status": "downloading", "files": files,
                         "bytes": sum(sizes[n] for n in files), "origin": f"https://huggingface.co/{info.id}"}
        write_json(ROOT / "models/manifest.json", manifest)
        try:
            # O Hub retoma arquivos .incomplete automaticamente; local_dir evita cópia dupla de pesos.
            snapshot_download(info.id, revision=info.sha, local_dir=target, allow_patterns=files, max_workers=2)
            for name in files:
                if not (target / name).is_file() or (sizes[name] and (target / name).stat().st_size != sizes[name]):
                    raise RuntimeError(f"Arquivo incompleto: {name}")
            manifest[key].update(status="installed", installed_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            manifest[key].update(status="interrupted", error=str(exc))
            raise
        finally:
            write_json(ROOT / "models/manifest.json", manifest)
        print(f"Instalado: {key}, revisão {info.sha}, {len(files)} arquivos.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[*MODELS, "all"], default=["all"])
    args = parser.parse_args()
    for key in MODELS if "all" in args.models else args.models:
        try:
            download(key)
        except Exception as exc:
            write_json(ROOT / "logs/last-download-error.json", {"model": key, "error": str(exc),
                       "time": datetime.now(timezone.utc).isoformat()})
            raise
