"""Confere manifesto e arquivos locais antes de retomar downloads incompletos."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.common import MODELS, ROOT, read_json
from scripts.download_models import download


def installed(key, manifest):
    record = manifest.get(key, {})
    files = record.get("files", [])
    return record.get("status") == "installed" and bool(files) and all(
        (ROOT / "models" / key / filename).is_file() and (ROOT / "models" / key / filename).stat().st_size > 0
        for filename in files
    )


if __name__ == "__main__":
    for key in MODELS:
        manifest = read_json(ROOT / "models/manifest.json", {})
        if installed(key, manifest):
            print(f"Modelo disponível: {key}", flush=True)
        else:
            download(key)
