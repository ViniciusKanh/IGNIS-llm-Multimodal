"""Prepara imagens públicas ou próprias com deduplicação e separação por conteúdo."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.common import ROOT, inside, write_json
from PIL import Image, ImageOps
import requests


def pixel_hash(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def split_records(records: list[dict], seed=42) -> dict:
    """Agrupa duplicatas antes de dividir, garantindo splits sem imagens idênticas."""
    unique = {row["sha256"]: row for row in records}
    rows = sorted(unique.values(), key=lambda x: x["sha256"])
    if len(rows) < 12:
        raise ValueError("São necessárias pelo menos 12 imagens distintas com legenda.")
    random.Random(seed).shuffle(rows)
    n_validation = max(2, round(len(rows) * 0.15))
    return {"train": rows[n_validation:], "validation": rows[:n_validation]}


def save_dataset(records, source):
    splits = split_records(records)
    digest = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    document = {"fingerprint": digest, "source": source, "seed": 42, **splits}
    write_json(ROOT / "datasets/manifest.json", document)
    print(f"Dataset: {len(splits['train'])} treino, {len(splits['validation'])} validação; {digest}")
    return document


def public_dataset(limit=96):
    """Seleciona obras marcadas individualmente como domínio público pela instituição."""
    session = requests.Session()
    session.headers["User-Agent"] = "IgnisMultimodal/0.1 (local research dataset)"
    folder = ROOT / "datasets/artic"
    folder.mkdir(exist_ok=True)
    records = []
    page = 1
    while len(records) < limit and page <= 20:
        response = session.get("https://api.artic.edu/api/v1/artworks", params={
            "page": page, "limit": 100,
            "fields": "id,title,image_id,is_public_domain,artwork_type_title,medium_display,artist_display,copyright_notice",
        }, timeout=30)
        response.raise_for_status()
        body = response.json()
        base = body["config"]["iiif_url"]
        if not base.startswith("https://www.artic.edu/iiif/2"):
            raise RuntimeError("Origem de imagens inesperada; revisar metadados da instituição.")
        for row in body["data"]:
            if row.get("is_public_domain") is not True or not row.get("image_id"):
                continue
            target = folder / f"{row['id']}.jpg"
            url = f"{base}/{row['image_id']}/full/843,/0/default.jpg"
            if not target.exists():
                try:
                    raw = session.get(url, timeout=40)
                    raw.raise_for_status()
                    image = Image.open(io.BytesIO(raw.content)).convert("RGB")
                    if min(image.size) < 384:
                        continue
                    image.save(target, quality=95)
                except (requests.RequestException, OSError) as exc:
                    print(f"Imagem {row['id']} ignorada: {exc}", flush=True)
                    continue
            with Image.open(target) as image:
                digest = pixel_hash(image)
            caption = f"{row['title']}. {row.get('artwork_type_title') or 'Artwork'}. {row.get('medium_display') or ''}"
            records.append({"file": target.relative_to(ROOT / "datasets").as_posix(), "caption": caption,
                            "sha256": digest, "license": "public-domain", "metadata": row,
                            "source": f"https://www.artic.edu/artworks/{row['id']}", "image_url": url})
            print(f"Imagem {len(records)}/{limit}: {row['title']}", flush=True)
            if len(records) >= limit:
                break
        page += 1
        if not body["data"]:
            break
    return save_dataset(records, {"name": "Art Institute of Chicago — obras em domínio público",
                                 "terms": "https://api.artic.edu/docs/#copyright", "metadata_license": "CC0-1.0",
                                 "scope": "Somente is_public_domain=true; metadados individuais preservados."})


def local_dataset(metadata: str):
    """Importa JSONL dentro de datasets: file, caption, license e source obrigatórios."""
    records = []
    for line in inside(ROOT / "datasets", metadata).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not all(row.get(k) for k in ("file", "caption", "license", "source")):
            raise ValueError("Cada imagem própria precisa de file, caption, license e source.")
        with Image.open(inside(ROOT / "datasets", row["file"])) as image:
            if min(image.size) < 256:
                raise ValueError(f"Imagem pequena: {row['file']}")
            row["sha256"] = pixel_hash(image)
        records.append(row)
    return save_dataset(records, {"name": "Dataset fornecido pelo usuário", "metadata": metadata})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=96)
    parser.add_argument("--metadata", help="JSONL relativo a datasets/; omita para dataset público.")
    args = parser.parse_args()
    if not 12 <= args.limit <= 1000:
        parser.error("limit deve estar entre 12 e 1000")
    local_dataset(args.metadata) if args.metadata else public_dataset(args.limit)
