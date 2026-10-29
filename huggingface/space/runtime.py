"""Inferência real com pesos publicados e suporte explícito a ZeroGPU."""
import os
import threading
import time
from pathlib import Path

ZERO_GPU = os.environ.get("IGNIS_ZERO_GPU") == "1"
if ZERO_GPU:
    import spaces
else:
    spaces = None

import torch
from PIL import ImageOps
from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
                       DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler)
from settings import BASE_ID, MODEL_ID, RELEASE

LOCK = threading.Lock()
PIPE = EDIT_PIPE = INPAINT_PIPE = SCHEDULER_CONFIG = None


def gpu(function):
    return spaces.GPU(duration=60)(function) if spaces is not None else function


def initialize():
    global PIPE, EDIT_PIPE, INPAINT_PIPE, SCHEDULER_CONFIG
    if PIPE is not None:
        return
    if not RELEASE.get("model_revision"):
        raise RuntimeError("Versão do adaptador não configurada. Republique pelo script do projeto.")
    cuda = ZERO_GPU or torch.cuda.is_available()
    local_base = os.environ.get("IGNIS_LOCAL_BASE_PATH") if not ZERO_GPU else None
    local_adapter = os.environ.get("IGNIS_LOCAL_ADAPTER_PATH") if not ZERO_GPU else None
    PIPE = StableDiffusionPipeline.from_pretrained(local_base or BASE_ID,
        revision=None if local_base else RELEASE.get("base_revision"), variant="fp16",
        torch_dtype=torch.float16 if cuda else torch.float32, use_safetensors=True,
        local_files_only=bool(local_base))
    PIPE.load_lora_weights(local_adapter or MODEL_ID, revision=None if local_adapter else RELEASE["model_revision"],
                           adapter_name="ignis", weight_name="pytorch_lora_weights.safetensors", local_files_only=bool(local_adapter))
    PIPE.to("cuda" if cuda else "cpu")
    PIPE.enable_vae_slicing()
    SCHEDULER_CONFIG = dict(PIPE.scheduler.config)
    EDIT_PIPE = StableDiffusionImg2ImgPipeline(**PIPE.components)
    INPAINT_PIPE = StableDiffusionInpaintPipeline(**PIPE.components)


def sample(options, mode="text-to-image", image=None, mask=None, strength=.65):
    initialize()
    pipe = PIPE if mode == "text-to-image" else INPAINT_PIPE if mode == "inpainting" else EDIT_PIPE
    scheduler = DPMSolverMultistepScheduler if options["sampler"] == "DPM++ Karras" else EulerAncestralDiscreteScheduler
    pipe.scheduler = scheduler.from_config(SCHEDULER_CONFIG, **({"use_karras_sigmas": True} if scheduler is DPMSolverMultistepScheduler else {}))
    if options["model"] == "IGNIS LoRA":
        pipe.enable_lora()
        pipe.set_adapters("ignis", adapter_weights=options["lora_scale"])
    else:
        pipe.disable_lora()
    args = dict(prompt=options["prompt"], negative_prompt=options["negative_prompt"],
                num_inference_steps=options["steps"], guidance_scale=options["guidance_scale"],
                generator=torch.Generator(device="cpu").manual_seed(options["seed"]),
                width=options["width"], height=options["height"])
    if mode != "text-to-image":
        if image is None or image.width * image.height > 20_000_000:
            raise ValueError("Envie uma imagem inicial de até 20 megapixels.")
        args.update(image=ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (options["width"], options["height"])), strength=float(strength))
        if mode == "image-to-image":
            args.pop("width"); args.pop("height")
        if mode == "inpainting":
            if mask is None:
                raise ValueError("Envie a máscara: branco para editar, preto para preservar.")
            args["mask_image"] = ImageOps.fit(mask.convert("L"), (options["width"], options["height"]))
    with torch.inference_mode():
        result = pipe(**args)
    if result.nsfw_content_detected and any(result.nsfw_content_detected):
        raise ValueError("O filtro de segurança bloqueou a saída. Ajuste o prompt.")
    return result.images[0]


@gpu
def generate(options):
    with LOCK:
        start = time.monotonic()
        results = []
        for index in range(options["count"]):
            current = {**options, "seed": (options["seed"] + index) % 2**32}
            results.append((sample(current), {**current, "operation": "text-to-image"}))
        return results, round(time.monotonic() - start, 2)


@gpu
def compare(options):
    with LOCK:
        start = time.monotonic()
        results = []
        for name in ("Stable Diffusion 1.5", "IGNIS LoRA"):
            current = {**options, "model": name, "operation": "comparison"}
            results.append((sample(current), current))
        return results, round(time.monotonic() - start, 2)


@gpu
def edit(options, image, mask, mode, strength):
    if mode not in ("image-to-image", "inpainting") or not .1 <= float(strength) <= .95:
        raise ValueError("Modo ou intensidade de edição inválidos.")
    with LOCK:
        start = time.monotonic()
        result = sample(options, mode, image, mask, strength)
        return [(result, {**options, "operation": mode, "edit_strength": float(strength)})], round(time.monotonic() - start, 2)


if ZERO_GPU:
    # O ZeroGPU prepara os tensores CUDA na inicialização, fora das funções decoradas.
    initialize()
