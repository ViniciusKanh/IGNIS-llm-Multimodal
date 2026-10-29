"""Prepara e publica somente README, logotipo e checksums do modelo existente."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from model_card import render_card, SOURCE


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prepare(project, output):
    package = project / "outputs/hf-release/model"
    release = read(project / "outputs/hf-release/space/release.json")
    model_id = release["model_id"]
    weights = package / "pytorch_lora_weights.safetensors"
    card = render_card(model_id, read(package / "training_report.json"),
                       read(package / "training_config.json"), read(package / "data_provenance.json"),
                       read(package / "evaluation_report.json"), weights, release["model_revision"])
    (output / "assets").mkdir(parents=True, exist_ok=True)
    logo = SOURCE / "huggingface/model/assets/ignis-multimodal.png"
    if not logo.is_file():
        raise FileNotFoundError("O logotipo não está no pacote de código.")
    shutil.copy2(logo, output / "assets/ignis-multimodal.png")
    (output / "README.md").write_text(card, encoding="utf-8")
    # Recalcula o manifesto sem copiar ou alterar pesos e demais evidências.
    sums = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob("*") if p.is_file() and p.name != "checksums.json"}
    for relative in ("README.md", "assets/ignis-multimodal.png"):
        sums[relative] = hashlib.sha256((output / relative).read_bytes()).hexdigest()
    (output / "checksums.json").write_text(json.dumps(sums, indent=2), encoding="utf-8")
    return release, sums


def publish(project, output, release, sums):
    os.environ["HF_HOME"] = str(project / ".cache/huggingface")
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info(release["model_id"], files_metadata=True)
    remote = next(row for row in info.siblings if row.rfilename == "pytorch_lora_weights.safetensors")
    if not remote.lfs or remote.lfs.sha256 != sums["pytorch_lora_weights.safetensors"]:
        raise RuntimeError("Os pesos remotos não correspondem ao experimento. Documentação não enviada.")
    # O commit é restrito à documentação; evita sobrescrever alterações concorrentes.
    commit = api.upload_folder(repo_id=release["model_id"], repo_type="model", folder_path=output,
        allow_patterns=["README.md", "assets/ignis-multimodal.png", "checksums.json"],
        parent_commit=info.sha,
        commit_message="Documentar IGNIS Image LoRA v1 com logotipo, método, resultados e exemplos")
    result = {"model_id": release["model_id"], "documentation_revision": commit.oid,
              "weights_sha256": sums["pytorch_lora_weights.safetensors"],
              "url": f"https://huggingface.co/{release['model_id']}"}
    (output / "publication-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("C:/dev/Ignis-Multimodal"))
    parser.add_argument("--output", type=Path, default=SOURCE / "outputs/model-documentation")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    release, sums = prepare(args.project_root, args.output)
    print(f"README e logotipo preparados: {args.output}", flush=True)
    if args.publish:
        publish(args.project_root, args.output, release, sums)
