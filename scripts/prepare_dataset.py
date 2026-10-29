"""Prepara imagens públicas ou próprias com deduplicação e separação por conteúdo."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import sys
import zipfile
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


def artic_dataset(limit=96):
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


def diffusiondb_records(archive_path, folder, limit, revision):
    """Lê apenas as imagens selecionadas do lote, sem extrair caminhos do ZIP."""
    records, captions, hashes = [], set(), set()
    excluded = re.compile(r"\b(nsfw|nude|naked|porn|sex|sexual|gore|blood|nipple|breast|penis|vagina)\b", re.I)
    with zipfile.ZipFile(archive_path) as archive:
        members = {Path(item.filename).name: item for item in archive.infolist() if not item.is_dir()}
        metadata_files = [item for name, item in members.items() if name.endswith(".json")]
        if len(metadata_files) != 1 or metadata_files[0].file_size > 20 * 1024**2:
            raise ValueError("Estrutura de metadados inesperada no lote DiffusionDB.")
        metadata = json.loads(archive.read(metadata_files[0]))
        candidates = sorted(metadata.items())
        random.Random(42).shuffle(candidates)
        folder.mkdir(parents=True, exist_ok=True)
        for name, info in candidates:
            caption = str(info.get("p") or info.get("prompt") or "").strip()
            normalized = " ".join(caption.lower().split())
            if len(caption) < 15 or excluded.search(caption) or normalized in captions:
                continue
            if name != Path(name).name or not re.fullmatch(r"[a-zA-Z0-9_-]+\.png", name):
                continue
            member = members.get(name)
            if member is None or member.file_size > 20 * 1024**2:
                continue
            with Image.open(io.BytesIO(archive.read(member))) as source:
                if min(source.size) < 256 or source.width * source.height > 16_000_000:
                    continue
                image = ImageOps.exif_transpose(source).convert("RGB")
                digest = pixel_hash(image)
                if digest in hashes:
                    continue
                target = folder / name
                image.save(target)
            captions.add(normalized)
            hashes.add(digest)
            records.append({"file": target.relative_to(ROOT / "datasets").as_posix(), "caption": caption,
                            "sha256": digest, "license": "cc0-1.0", "synthetic": True,
                            "source": f"https://huggingface.co/datasets/poloclub/diffusiondb/tree/{revision}",
                            "source_file": "images/part-000001.zip", "source_image": name})
            print(f"DiffusionDB: {len(records)}/{limit} imagens selecionadas", flush=True)
            if len(records) >= limit:
                break
    return records


def diffusiondb_dataset(limit=96):
    """Baixa um único lote público do HF e registra sua licença, revisão e origem sintética."""
    from huggingface_hub import HfApi, hf_hub_download
    repository = "poloclub/diffusiondb"
    info = HfApi().dataset_info(repository, files_metadata=True)
    card = info.card_data.to_dict() if info.card_data is not None else {}
    licenses = card.get("license", [])
    licenses = [licenses] if isinstance(licenses, str) else licenses
    if "cc0-1.0" not in licenses:
        raise RuntimeError(f"Licença do DiffusionDB ausente ou alterada: {licenses!r}.")
    filename = "images/part-000001.zip"
    entry = next((item for item in info.siblings if item.rfilename == filename), None)
    if entry is None or not entry.size or entry.size > 1024**3:
        raise RuntimeError("Lote ausente ou maior que o limite de download de 1 GB.")
    destination = ROOT / "datasets/diffusiondb/source"
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < entry.size + 1024**3:
        raise RuntimeError("Espaço insuficiente para o lote e as imagens selecionadas.")
    print(f"Fonte alternativa: DiffusionDB CC0; um lote de {entry.size / 1024**2:.0f} MiB, seleção de até {limit} imagens.", flush=True)
    archive_path = hf_hub_download(repository, filename, repo_type="dataset", revision=info.sha, local_dir=destination)
    hf_hub_download(repository, "README.md", repo_type="dataset", revision=info.sha, local_dir=destination)
    records = diffusiondb_records(archive_path, ROOT / "datasets/diffusiondb/images", limit, info.sha)
    if len(records) < min(limit, 24):
        raise RuntimeError(f"Apenas {len(records)} imagens elegíveis; não foi preparado um dataset de qualidade suficiente.")
    return save_dataset(records, {"name": "DiffusionDB — recorte experimental de imagens sintéticas",
                                 "repo": repository, "revision": info.sha, "license": "cc0-1.0",
                                 "terms": f"https://huggingface.co/datasets/{repository}/blob/{info.sha}/README.md",
                                 "synthetic": True, "source_file": filename,
                                 "scope": "Um exemplar por prompt normalizado; deduplicação por pixels; exclusão textual heurística. Não há certificação de segurança visual.",
                                 "limitation": "Imagens geradas por Stable Diffusion. Não são fotografias reais nem demonstram melhoria geral da base."})


def public_dataset(limit=96, source="auto"):
    if source == "diffusiondb":
        return diffusiondb_dataset(limit)
    if source == "artic":
        return artic_dataset(limit)
    try:
        return artic_dataset(limit)
    except (requests.RequestException, ValueError) as exc:
        print(f"Fonte do museu indisponível ({exc}); usando DiffusionDB no Hugging Face.", flush=True)
        write_json(ROOT / "logs/dataset-source-fallback.json", {"from": "artic", "to": "diffusiondb", "reason": str(exc)})
        return diffusiondb_dataset(limit)


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
    parser.add_argument("--source", choices=["auto", "diffusiondb", "artic"], default="auto")
    args = parser.parse_args()
    if not 12 <= args.limit <= 1000:
        parser.error("limit deve estar entre 12 e 1000")
    local_dataset(args.metadata) if args.metadata else public_dataset(args.limit, args.source)
