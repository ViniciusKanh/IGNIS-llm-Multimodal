"""Treinamento real de LoRA UNet com cache de latentes e validação sem vazamento."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.common import ROOT, inside, model_path, read_json, write_json
from filelock import FileLock


def validate_manifest(manifest):
    train = manifest.get("train", [])
    validation = manifest.get("validation", [])
    if len(train) < 10 or len(validation) < 2:
        raise ValueError("Dataset insuficiente: mínimo 10 imagens de treino e 2 de validação.")
    train_ids = {row["sha256"] for row in train}
    val_ids = {row["sha256"] for row in validation}
    if train_ids & val_ids:
        raise ValueError("Vazamento: imagem presente em treino e validação.")
    if len(train_ids) != len(train) or len(val_ids) != len(validation):
        raise ValueError("Dataset contém duplicatas dentro de um split.")
    from scripts.prepare_dataset import pixel_hash
    from PIL import Image
    for row in train + validation:
        if not all(row.get(k) for k in ("caption", "license", "source")):
            raise ValueError("Dataset sem legenda, licença ou origem.")
        with Image.open(inside(ROOT / "datasets", row["file"])) as image:
            if pixel_hash(image) != row["sha256"]:
                raise ValueError(f"Imagem mudou desde a preparação: {row['file']}")


def clear_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cache_data(base, manifest, resolution, dtype, device):
    """VAE e encoder ocupam a GPU em momentos distintos; UNet só é carregado depois."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from diffusers import AutoencoderKL
    from transformers import CLIPTokenizer, CLIPTextModel
    from safetensors.torch import load_file, save_file
    key = hashlib.sha256(json.dumps({"dataset": manifest, "resolution": resolution,
                                    "base": read_json(ROOT / "models/manifest.json", {}).get("sd15", {}).get("revision")},
                                   sort_keys=True).encode()).hexdigest()[:20]
    cache = ROOT / ".cache/latents" / key
    cache.mkdir(parents=True, exist_ok=True)
    rows = manifest["train"] + manifest["validation"]
    if all((cache / f"{row['sha256']}.safetensors").exists() for row in rows):
        return cache
    variant = "fp16" if any(base.rglob("*.fp16.safetensors")) else None
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", variant=variant, torch_dtype=dtype,
                                       local_files_only=True, use_safetensors=True).to(device)
    vae.requires_grad_(False).eval()
    vae.enable_slicing()
    vae.enable_tiling()
    latents = {}
    with torch.inference_mode():
        for index, row in enumerate(rows):
            with Image.open(inside(ROOT / "datasets", row["file"])) as image:
                image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (resolution, resolution))
                array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
            pixels = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
            generator = torch.Generator(device=device).manual_seed(42 + index)
            latent = vae.encode(pixels).latent_dist.sample(generator=generator) * vae.config.scaling_factor
            latents[row["sha256"]] = latent.squeeze(0).cpu().contiguous()
    del vae, pixels, latent
    clear_cuda()
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer", local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", variant=variant,
                                           torch_dtype=dtype, local_files_only=True, use_safetensors=True).to(device)
    encoder.requires_grad_(False).eval()
    with torch.inference_mode():
        for row in rows:
            ids = tokenizer(row["caption"], padding="max_length", truncation=True,
                            max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
            embedding = encoder(ids)[0].squeeze(0).cpu().contiguous()
            save_file({"latent": latents[row["sha256"]], "embedding": embedding},
                      str(cache / f"{row['sha256']}.safetensors"))
    del encoder, tokenizer, ids, embedding, latents
    clear_cuda()
    return cache


def run(config, resolution):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Treinamento não iniciado: PyTorch não dispõe de CUDA. Execute bootstrap.ps1 para instalar a distribuição CUDA.")
    import torch.nn.functional as F
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file, save_file

    manifest = read_json(ROOT / "datasets/manifest.json")
    if not manifest:
        raise FileNotFoundError("Prepare o dataset antes de treinar.")
    validate_manifest(manifest)
    base = model_path("sd15")
    config = {**config, "resolution": resolution, "dataset_fingerprint": manifest["fingerprint"],
              "base_revision": read_json(ROOT / "models/manifest.json")["sd15"]["revision"]}
    run_dir = ROOT / "training" / config["name"]
    run_dir.mkdir(exist_ok=True)
    previous = read_json(run_dir / "config.json")
    if previous and previous != config:
        raise ValueError("Configuração/dataset mudou. Use outro nome de execução para preservar a retomada.")
    write_json(run_dir / "config.json", config)
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=config["gradient_accumulation"])
    set_seed(config["seed"])
    cache = cache_data(base, manifest, resolution, torch.float16, "cuda")
    variant = "fp16" if any(base.rglob("*.fp16.safetensors")) else None
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", variant=variant,
                                               torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=config["rank"], lora_alpha=config["rank"], init_lora_weights="gaussian",
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    # O scaler FP16 exige parâmetros treináveis em FP32; a base congelada permanece FP16.
    parameters = [p for p in unet.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.01)
    unet, optimizer = accelerator.prepare(unet, optimizer)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler", local_files_only=True)

    def lr_factor(step):
        if step < config["warmup_steps"]:
            return (step + 1) / max(1, config["warmup_steps"])
        progress = (step - config["warmup_steps"]) / max(1, config["max_steps"] - config["warmup_steps"])
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lr_factor)
    accelerator.register_for_checkpointing(scheduler)

    def export(folder):
        folder.mkdir(parents=True, exist_ok=True)
        state = get_peft_model_state_dict(accelerator.unwrap_model(unet))
        state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
        save_file(state, str(folder / "adapter.safetensors"))
        StableDiffusionPipeline.save_lora_weights(folder, unet_lora_layers=convert_state_dict_to_diffusers(state),
                                                  safe_serialization=True)

    def save_hook(models, weights, output_dir):
        export(Path(output_dir))
        # Evita salvar cópias multigigabyte do UNet congelado em cada checkpoint.
        while weights:
            weights.pop()

    def load_hook(models, input_dir):
        state = load_file(str(Path(input_dir) / "adapter.safetensors"))
        while models:
            set_peft_model_state_dict(accelerator.unwrap_model(models.pop()), state)

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
    state = {"step": 0, "best_validation_loss": None, "bad_validations": 0, "status": "running"}
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    complete = [p for p in checkpoints if (p / "progress.json").exists()]
    if complete:
        latest = complete[-1]
        accelerator.load_state(str(latest))
        state = read_json(latest / "progress.json")
        state["status"] = "running"
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            kept = []
            for line in metrics_path.read_text().splitlines():
                try:
                    if json.loads(line)["step"] <= state["step"]:
                        kept.append(line)
                except json.JSONDecodeError:
                    continue
            metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"Retomando passo {state['step']}", flush=True)

    def get_row(row):
        tensors = load_file(str(cache / f"{row['sha256']}.safetensors"))
        return (tensors["latent"].unsqueeze(0).to("cuda", dtype=torch.float16),
                tensors["embedding"].unsqueeze(0).to("cuda", dtype=torch.float16))

    def loss_for(latent, embedding, generator=None):
        noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latent.shape[0],),
                          device=latent.device, generator=generator)
        noisy = noise_scheduler.add_noise(latent, noise, t)
        prediction = unet(noisy, t, encoder_hidden_states=embedding).sample
        target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latent, noise, t)
        return F.mse_loss(prediction.float(), target.float())

    def validate():
        unet.eval()
        values = []
        with torch.no_grad(), accelerator.autocast():
            for index, row in enumerate(manifest["validation"]):
                latent, embedding = get_row(row)
                values.append(float(loss_for(latent, embedding, torch.Generator(device="cuda").manual_seed(1000 + index)).item()))
        unet.train()
        return sum(values) / len(values)

    def checkpoint():
        folder = run_dir / f"checkpoint-{state['step']}"
        accelerator.save_state(str(folder))
        write_json(folder / "progress.json", state)
        write_json(run_dir / "progress.json", state)

    # Métrica inicial usa o adaptador de atualização zero, antes de qualquer gradiente.
    if state["step"] == 0:
        state["initial_validation_loss"] = validate()
        state["best_validation_loss"] = state["initial_validation_loss"]
    start = time.monotonic()
    unet.train()
    optimizer.zero_grad()
    step_loss = 0.0
    accumulation = config["gradient_accumulation"]
    try:
        for micro in range(state["step"] * accumulation, config["max_steps"] * accumulation):
            # Ordem reproduzível por época, calculada pelo índice global, inclusive após retomada.
            epoch, offset = divmod(micro, len(manifest["train"]))
            order = torch.randperm(len(manifest["train"]), generator=torch.Generator().manual_seed(config["seed"] + epoch)).tolist()
            latent, embedding = get_row(manifest["train"][order[offset]])
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    loss = loss_for(latent, embedding)
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss não finita; interrompendo sem promover checkpoint inválido.")
                accelerator.backward(loss)
                step_loss += float(loss.detach().item()) / accumulation
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if not accelerator.sync_gradients:
                    continue
                scheduler.step()
                state["step"] += 1
                report = {"step": state["step"], "train_loss": step_loss, "lr": scheduler.get_last_lr()[0]}
                step_loss = 0.0
                if state["step"] % config["validation_every"] == 0:
                    report["validation_loss"] = validate()
                    if report["validation_loss"] < state["best_validation_loss"] - 1e-5:
                        state["best_validation_loss"] = report["validation_loss"]
                        state["best_step"] = state["step"]
                        state["bad_validations"] = 0
                        export(run_dir / "best")
                    else:
                        state["bad_validations"] += 1
                state.update(last_metrics=report, elapsed_seconds=round(time.monotonic() - start, 2))
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(report) + "\n")
                write_json(run_dir / "progress.json", state)
                print(json.dumps(report), flush=True)
                if (ROOT / "logs/stop-training").exists():
                    state.update(status="interrupted", stop_reason="user_requested")
                    checkpoint()
                    raise KeyboardInterrupt("Parada solicitada; checkpoint salvo.")
                if state["step"] % config["checkpoint_every"] == 0:
                    checkpoint()
                if state["step"] >= config["min_steps"] and state["bad_validations"] >= config["patience"]:
                    state["stop_reason"] = "early_stopping_validation"
                    break
        state["status"] = "completed" if state["step"] >= config["min_steps"] else "smoke_test_only"
        state.setdefault("stop_reason", "max_steps")
        export(run_dir / "final")
        checkpoint()
        write_json(run_dir / "report.json", {**state, "config": config,
                                             "interpretation": "Experimento LoRA executado; qualidade visual requer comparação base/adaptador."})
        return run_dir
    except BaseException as exc:
        state.update(status="interrupted", error=str(exc))
        write_json(run_dir / "progress.json", state)
        raise
    finally:
        del unet, optimizer, parameters
        accelerator.free_memory()
        clear_cuda()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resolution", type=int, choices=[256, 384, 512])
    parser.add_argument("--name")
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()
    config = read_json(inside(ROOT, args.config))
    if args.steps:
        config["max_steps"] = args.steps
    if args.name:
        if not args.name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Nome de execução inválido.")
        config["name"] = args.name
    if config["batch_size"] != 1:
        raise ValueError("Este pipeline de baixa VRAM usa batch size 1; ajuste a acumulação.")
    for key in ("max_steps", "gradient_accumulation", "checkpoint_every", "validation_every", "rank"):
        if config[key] < 1:
            raise ValueError(f"Parâmetro precisa ser positivo: {key}")
    with FileLock(str(ROOT / "logs/gpu.lock"), timeout=0):
        (ROOT / "logs/stop-training").unlink(missing_ok=True)
        import torch
        resolutions = [args.resolution] if args.resolution else [config["resolution"], 384, 256]
        original_name = config["name"]
        run_dir = None
        for index, resolution in enumerate(dict.fromkeys(resolutions)):
            config["name"] = original_name if index == 0 else f"{original_name}-r{resolution}"
            try:
                run_dir = run(config, resolution)
                break
            except torch.cuda.OutOfMemoryError as exc:
                clear_cuda()
                write_json(ROOT / "logs/last-training-oom.json", {"resolution": resolution, "error": str(exc)})
        if run_dir is None:
            raise RuntimeError("Treinamento impedido por VRAM nas resoluções tentadas; consulte logs/last-training-oom.json.")
        write_json(ROOT / "training/latest-run.json", {"name": run_dir.name, "adapter": run_dir.name + "/final"})
    if not args.no_evaluate:
        from scripts.evaluate import evaluate
        evaluate(run_dir.relative_to(ROOT / "training").as_posix() + "/final")


if __name__ == "__main__":
    main()
