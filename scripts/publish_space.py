"""Prepara e publica somente o Space IGNIS, preservando a revisão já publicada do modelo."""
import argparse
import json
import os
import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_space(project, output):
    model_folder = project / "outputs/hf-release/model"
    previous_release = project / "outputs/hf-release/space/release.json"
    if not previous_release.is_file() or not (model_folder / "README.md").is_file():
        raise RuntimeError("Pacote publicado do modelo não encontrado. Prepare o release antes de criar o Space.")
    release = read(previous_release)
    if not release.get("model_revision") or not release.get("base_revision"):
        raise RuntimeError("Falta a revisão exata do modelo publicado no release.json.")
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("app.py", "runtime.py", "settings.py", "styles.css", "README.md", "requirements.txt"):
        shutil.copy2(SOURCE / "huggingface/space" / filename, output / filename)
    card = (model_folder / "README.md").read_text(encoding="utf-8")
    card = card.replace("(samples/", f"(https://huggingface.co/{release['model_id']}/resolve/{release['model_revision']}/samples/")
    (output / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    (output / "release.json").write_text(json.dumps(release, indent=2), encoding="utf-8")
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    for original, destination in (("training_report.json", "training.json"), ("training_config.json", "config.json"),
                                   ("evaluation_report.json", "evaluation.json")):
        shutil.copy2(model_folder / original, assets / destination)
    (assets / "samples").mkdir(exist_ok=True)
    evaluation = read(model_folder / "evaluation_report.json")
    for pair in evaluation["pairs"]:
        relative = Path(pair["sample"])
        if relative.parts[0] != "samples" or len(relative.parts) != 2 or relative.suffix != ".png":
            raise RuntimeError("Caminho de amostra inválido na avaliação.")
        shutil.copy2(model_folder / relative, assets / relative)
    for path in model_folder.glob("BASE_*"):
        shutil.copy2(path, output / path.name)
    return release


def publish(project, output, space_id, release):
    os.environ["HF_HOME"] = str(project / ".cache/huggingface")
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
    api = HfApi()
    account = api.whoami()
    if account["name"].lower() != space_id.split("/")[0].lower():
        raise RuntimeError("A conta autenticada não corresponde ao proprietário indicado para o Space.")
    # Somente ZeroGPU: não há tentativa de contratar CPU Upgrade ou GPU dedicada.
    try:
        api.create_repo(space_id, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g",
                        space_variables=[{"key": "IGNIS_ZERO_GPU", "value": "1"}], exist_ok=True)
        current = api.get_space_runtime(space_id)
        if current.hardware != "zero-a10g" and current.requested_hardware != "zero-a10g":
            api.request_space_hardware(space_id, hardware="zero-a10g")
        api.add_space_variable(space_id, key="IGNIS_ZERO_GPU", value="1")
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code in (402, 403):
            raise RuntimeError("O Hugging Face não autorizou ZeroGPU nesta conta ou o token não permite criar o Space. Nenhum hardware pago foi solicitado. Confira a elegibilidade ZeroGPU nas configurações da conta.") from exc
        raise
    commit = api.upload_folder(repo_id=space_id, repo_type="space", folder_path=output,
        allow_patterns=["app.py", "runtime.py", "settings.py", "styles.css", "README.md", "MODEL_CARD.md",
                        "requirements.txt", "release.json", "assets/*.json", "assets/samples/*.png", "BASE_*"],
        commit_message="Publicar IGNIS Image Lab com ZeroGPU, estúdio, edição, comparação e documentação")
    runtime = api.get_space_runtime(space_id)
    status = {"space_id": space_id, "space_revision": commit.oid, "model_revision": release["model_revision"],
              "hardware": runtime.hardware, "requested_hardware": runtime.requested_hardware,
              "stage": runtime.stage, "url": f"https://huggingface.co/spaces/{space_id}"}
    (output / "deployment-result.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--space-id", default="ViniciusKhan/IGNIS-Image-Lab")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = (args.output or project / "outputs/hf-release/space").resolve()
    release = prepare_space(project, output)
    print(f"Space preparado: {output}", flush=True)
    if not args.prepare_only:
        publish(project, output, args.space_id, release)
