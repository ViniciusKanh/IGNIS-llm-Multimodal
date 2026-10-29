"""Empacota evidências reais, model card e Space; publica somente com --publish."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.common import ROOT, inside, read_json, write_json


def collect_evidence(adapter):
    folder = inside(ROOT / "training", adapter)
    weights = folder / "pytorch_lora_weights.safetensors"
    report = read_json(folder.parent / "report.json", {})
    config = read_json(folder.parent / "config.json", {})
    dataset = read_json(ROOT / "datasets/manifest.json", {})
    base = read_json(ROOT / "models/manifest.json", {}).get("sd15", {})
    evaluation_folder = ROOT / "outputs/evaluations" / (folder.parent.name + "-" + folder.name)
    evaluation = read_json(evaluation_folder / "report.json", {})
    if not weights.is_file() or report.get("status") != "completed":
        raise RuntimeError("Publicação não preparada: faltam pesos e relatório de treinamento concluído.")
    if report.get("step", 0) < config.get("min_steps", 300):
        raise RuntimeError("O experimento tem apenas passos de teste; não constitui a versão treinada.")
    if dataset.get("fingerprint") != config.get("dataset_fingerprint"):
        raise RuntimeError("O dataset atual não corresponde ao dataset registrado no treinamento.")
    if base.get("revision") != config.get("base_revision") or base.get("status") != "installed":
        raise RuntimeError("A revisão da base não corresponde à execução de treinamento.")
    if not evaluation.get("pairs") or evaluation.get("adapter") != adapter:
        raise RuntimeError("Falta comparação real base × LoRA para este adaptador.")
    for index in range(1, len(evaluation["pairs"]) + 1):
        if not (evaluation_folder / f"pair-{index}.png").is_file():
            raise RuntimeError("A avaliação referencia amostra inexistente.")
    return folder, report, config, dataset, base, evaluation_folder, evaluation


def render_card(model_id, report, config, dataset):
    card = (ROOT / "huggingface/model/README.md").read_text(encoding="utf-8")
    card = card.replace("ViniciusKhan/ignis-image-lora-v1", model_id)
    card = card.replace("**Rascunho de model card. Não há pesos treinados ou avaliação confirmados nesta versão do documento.**",
                        "**Versão experimental com adaptador treinado e evidências de execução anexas.**")
    card = card.replace("Rascunho anterior à publicação dos pesos", "Versão experimental preparada para publicação")
    start = card.index("O coletor implementado consulta")
    end = card.index("As legendas são construídas", start)
    card = card[:start] + (f"O manifesto desta execução registra **{len(dataset['train'])} imagens de treino** e "
                         f"**{len(dataset['validation'])} de validação**. Origem: `{json.dumps(dataset.get('source'), ensure_ascii=False)}`. "
                         f"Fingerprint do dataset: `{dataset['fingerprint']}`. A proveniência individual acompanha a versão em `data_provenance.json`.\n\n") + card[end:]
    start = card.index("## Configuração inicial planejada")
    end = card.index("## Resultados e avaliação", start)
    card = card[:start] + "## Configuração efetivamente executada\n\n```json\n" + json.dumps(config, ensure_ascii=False, indent=2) + "\n```\n\n" + card[end:]
    start = card.index("**Resultados ainda não disponíveis neste rascunho.")
    end = card.index("## Como carregar os pesos", start)
    results = (f"Execução concluída com **{report['step']} atualizações do otimizador**. Motivo de parada: `{report.get('stop_reason')}`.\n\n"
               f"Loss inicial de validação: `{report.get('initial_validation_loss')}`. Melhor loss de validação observada durante o treinamento: "
               f"`{report.get('best_validation_loss')}`. A melhor loss pode corresponder a um checkpoint anterior ao adaptador final publicado.\n\n"
               "As imagens em `samples/` comparam base e adaptador final com prompts e sementes idênticos. "
               "Consulte `evaluation_report.json` para os pares e `training_report.json` para a execução. "
               "Não foram calculados FID ou CLIP score. A comparação não demonstra superioridade geral sobre a base.\n\n"
               "![Comparação base e IGNIS, mesma seed](samples/pair-1.png)\n\n")
    card = card[:start] + results + card[end:]
    card = card.replace("A versão final deve conter", "Esta versão contém")
    card = card.replace("O código abaixo pressupõe que o repositório já tenha sido publicado com o adaptador e que exista GPU CUDA compatível:",
                        "Após o envio desta versão ao Hub, carregue a base e o adaptador em uma GPU CUDA compatível:")
    return card


def prepare(adapter="ignis-image-lora-v1/final", model_id="ViniciusKhan/ignis-image-lora-v1", space_id="ViniciusKhan/IGNIS-Image-Lab"):
    import re
    for identifier in (model_id, space_id):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", identifier):
            raise ValueError("Identificador HF inválido: use usuario/repositorio.")
    folder, report, config, dataset, base, eval_folder, evaluation = collect_evidence(adapter)
    release = ROOT / "outputs/hf-release"
    model_dir, space_dir = release / "model", release / "space"
    for path in (model_dir, space_dir, model_dir / "samples"):
        path.mkdir(parents=True, exist_ok=True)
    license_files = [p for p in (ROOT / "models/sd15").iterdir() if p.is_file() and "license" in p.name.lower()]
    if not license_files:
        # O espelho SD 1.5 pode referenciar a licença sem incluir arquivo próprio.
        import requests
        source = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE"
        response = requests.get(source, timeout=20)
        response.raise_for_status()
        text = response.text
        if "CreativeML Open RAIL-M" not in text or "Attachment A" not in text:
            raise RuntimeError("O documento de licença upstream não corresponde ao esperado.")
        license_path = ROOT / "models/sd15/LICENSE.upstream.txt"
        license_path.write_text(text, encoding="utf-8")
        license_files = [license_path]
        base = {**base, "license_source": source, "license_sha256": hashlib.sha256(text.encode()).hexdigest()}
    shutil.copy2(folder / "pytorch_lora_weights.safetensors", model_dir)
    for path in license_files:
        shutil.copy2(path, model_dir / ("BASE_" + path.name))
    card = render_card(model_id, report, config, dataset)
    (model_dir / "README.md").write_text(card, encoding="utf-8")
    write_json(model_dir / "training_config.json", config)
    write_json(model_dir / "training_report.json", report)
    write_json(model_dir / "base_model.json", base)
    provenance = {"fingerprint": dataset["fingerprint"], "source": dataset.get("source")}
    for split in ("train", "validation"):
        provenance[split] = [{k: row[k] for k in ("caption", "sha256", "source", "license")}
                             for row in dataset[split]]
    write_json(model_dir / "data_provenance.json", provenance)
    pairs = []
    for index, pair in enumerate(evaluation["pairs"], 1):
        sample = f"pair-{index}.png"
        shutil.copy2(eval_folder / sample, model_dir / "samples" / sample)
        pairs.append({"prompt": pair["prompt"], "seed": pair["seed"], "sample": "samples/" + sample})
    write_json(model_dir / "evaluation_report.json", {"adapter": adapter, "pairs": pairs,
                                                      "note": evaluation.get("note")})
    for path in (ROOT / "huggingface/space").iterdir():
        if path.is_file():
            shutil.copy2(path, space_dir)
    for path in model_dir.glob("BASE_*"):
        shutil.copy2(path, space_dir)
    space_card = (space_dir / "README.md").read_text(encoding="utf-8").replace("ViniciusKhan/ignis-image-lora-v1", model_id)
    (space_dir / "README.md").write_text(space_card, encoding="utf-8")
    (space_dir / "MODEL_CARD.md").write_text(card.replace("(samples/pair-1.png)", f"(https://huggingface.co/{model_id}/resolve/main/samples/pair-1.png)"), encoding="utf-8")
    write_json(space_dir / "release.json", {"model_id": model_id, "base_revision": base["revision"]})
    checksums = {p.relative_to(model_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in model_dir.rglob("*") if p.is_file() and p.name != "checksums.json"}
    write_json(model_dir / "checksums.json", checksums)
    return model_dir, space_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="ignis-image-lora-v1/final")
    parser.add_argument("--model-id", default="ViniciusKhan/ignis-image-lora-v1")
    parser.add_argument("--space-id", default="ViniciusKhan/IGNIS-Image-Lab")
    parser.add_argument("--publish", action="store_true", help="Cria ou atualiza os dois repositórios no HF; requer login com permissão de escrita.")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    model_dir, space_dir = prepare(args.adapter, args.model_id, args.space_id)
    print(f"Pacotes preparados: {model_dir} e {space_dir}")
    if args.publish:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
        api.create_repo(args.model_id, repo_type="model", private=args.private, exist_ok=True)
        commit = api.upload_folder(repo_id=args.model_id, repo_type="model", folder_path=model_dir,
                                   commit_message="Publicar adaptador IGNIS com configuração e avaliação reais",
                                   allow_patterns=["README.md", "*.json", "BASE_*", "pytorch_lora_weights.safetensors", "samples/pair-*.png"])
        release = read_json(space_dir / "release.json")
        release["model_revision"] = commit.oid
        write_json(space_dir / "release.json", release)
        api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=space_dir,
                          commit_message="Publicar demonstração IGNIS vinculada à revisão do adaptador",
                          allow_patterns=["README.md", "MODEL_CARD.md", "app.py", "requirements.txt", "release.json", "BASE_*"])
        print(f"Modelo: https://huggingface.co/{args.model_id}")
        print(f"Space: https://huggingface.co/spaces/{args.space_id}")
