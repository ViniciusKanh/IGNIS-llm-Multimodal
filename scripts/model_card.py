"""Gera a documentação do modelo exclusivamente a partir das evidências reais."""
import hashlib
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]


def render_card(model_id, report, config, dataset, evaluation, weights, model_revision="main"):
    from safetensors import safe_open

    weights = Path(weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != evaluation["adapter_sha256"]:
        raise ValueError("Os pesos não correspondem à avaliação documentada.")
    if report["status"] != "completed":
        raise ValueError("Treinamento sem conclusão registrada.")
    if dataset["fingerprint"] != config["dataset_fingerprint"]:
        raise ValueError("O dataset não corresponde ao treinamento.")
    train_ids = {row["sha256"] for row in dataset["train"]}
    validation_ids = {row["sha256"] for row in dataset["validation"]}
    if train_ids & validation_ids:
        raise ValueError("Há imagens compartilhadas entre treino e validação.")
    if config["base_revision"] != evaluation["base_revision"]:
        raise ValueError("A revisão da base diverge da avaliação.")
    total = 0
    with safe_open(weights, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            count = 1
            for size in tensors.get_slice(key).get_shape():
                count *= size
            total += count
    pairs = evaluation["pairs"]
    samples = "\n\n".join(
        f"### Comparação {i}: seed {pair['seed']}\n\n"
        f"Prompt: `{pair['prompt']}`\n\n"
        f"![Base à esquerda e IGNIS à direita, seed {pair['seed']}]({pair['sample']})"
        for i, pair in enumerate(pairs, 1))
    values = {
        "MODEL_ID": model_id, "MODEL_REVISION": model_revision,
        "BASE_REVISION": config["base_revision"], "DATA_REVISION": dataset["source"]["revision"],
        "FINGERPRINT": dataset["fingerprint"], "WEIGHT_HASH": digest,
        "WEIGHT_BYTES": weights.stat().st_size, "WEIGHT_MB": f"{weights.stat().st_size / 1e6:.2f}",
        "LORA_PARAMS": total, "STEPS": report["step"], "PAIR_COUNT": len(pairs),
        "IMAGE_COUNT": 2 * len(pairs), "SAMPLES": samples,
        "TRAIN_COUNT": len(dataset["train"]), "VAL_COUNT": len(dataset["validation"]),
        "TOTAL_COUNT": len(dataset["train"]) + len(dataset["validation"]), "OVERLAP": 0,
        "EFFECTIVE_BATCH": config["batch_size"] * config["gradient_accumulation"],
        "INITIAL_LOSS": report["initial_validation_loss"], "BEST_LOSS": report["best_validation_loss"],
        "FINAL_LOSS": report["last_metrics"]["validation_loss"],
        "TRAIN_LOSS": report["last_metrics"]["train_loss"], "BEST_STEP": report["best_step"],
        "STOP_REASON": report["stop_reason"], "ELAPSED": report["elapsed_seconds"],
    }
    for marker, field in {"RANK": "rank", "RESOLUTION": "resolution", "BATCH": "batch_size",
                          "ACCUM": "gradient_accumulation", "LR": "learning_rate", "WARMUP": "warmup_steps",
                          "MAX_STEPS": "max_steps", "SEED": "seed", "CHECKPOINT": "checkpoint_every",
                          "VALIDATE": "validation_every", "PATIENCE": "patience", "MIN_STEPS": "min_steps"}.items():
        values[marker] = config[field]
    template = (SOURCE / "huggingface/model/model-card.template.md").read_text(encoding="utf-8")
    for key, value in values.items():
        template = template.replace(f"@@{key}@@", str(value))
    if re.search(r"@@[A-Z_]+@@", template):
        raise ValueError("O README ainda contém campos não preenchidos.")
    return template
