"""Ignis Core: serialização da GPU e pipelines locais de visão, geração e edição."""
from __future__ import annotations

import gc
import json
import threading
import time
import uuid
from pathlib import Path

from filelock import FileLock, Timeout
from PIL import Image, ImageOps
from backend.common import MODELS, ROOT, inside, model_path, read_json, write_json


class Engine:
    def __init__(self):
        self.pipeline = None
        self.processor = None
        self.loaded = None
        self.mutex = threading.RLock()

    def unload(self):
        self.pipeline = None
        self.processor = None
        self.loaded = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def lease(self):
        return FileLock(str(ROOT / "logs/gpu.lock"), timeout=0)

    def status(self):
        import psutil
        import torch
        cuda = torch.cuda.is_available()
        return {"loaded": self.loaded, "cuda": cuda, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if cuda else None,
                "vram_allocated_mb": round(torch.cuda.memory_allocated() / 2**20) if cuda else 0,
                "vram_free_mb": round(torch.cuda.mem_get_info()[0] / 2**20) if cuda else None,
                "ram_free_mb": round(psutil.virtual_memory().available / 2**20),
                "models": [{"key": key, **meta, **read_json(ROOT / "models/manifest.json", {}).get(key, {"status": "missing"})}
                           for key, meta in MODELS.items()]}

    def load(self, key="smolvlm", mode="text-to-image", adapter=None, sequential=False):
        import torch
        path = model_path(key)
        signature = (key, mode, adapter, sequential)
        if self.loaded == signature:
            return
        self.unload()
        cuda = torch.cuda.is_available()
        dtype = torch.float16 if cuda else torch.float32
        try:
            if key == "sd15":
                from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
                cls = {"text-to-image": StableDiffusionPipeline, "image-to-image": StableDiffusionImg2ImgPipeline,
                       "inpainting": StableDiffusionInpaintPipeline}[mode]
                variant = "fp16" if any(path.rglob("*.fp16.safetensors")) else None
                self.pipeline = cls.from_pretrained(path, torch_dtype=dtype, use_safetensors=True,
                                                     variant=variant, local_files_only=True)
                if adapter:
                    folder = inside(ROOT / "training", adapter)
                    if not (folder / "pytorch_lora_weights.safetensors").is_file():
                        raise FileNotFoundError("Adaptador sem pesos exportados.")
                    self.pipeline.load_lora_weights(folder, local_files_only=True)
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_vae_tiling()
                if cuda:
                    if sequential:
                        self.pipeline.enable_sequential_cpu_offload()
                    else:
                        self.pipeline.enable_model_cpu_offload()
                else:
                    self.pipeline.to("cpu")
            else:
                from transformers import AutoProcessor, AutoModelForImageTextToText
                kwargs = {"local_files_only": True, "dtype": dtype, "attn_implementation": "sdpa"}
                if key == "qwen" and cuda:
                    kwargs.update(device_map="auto", max_memory={0: "4GiB", "cpu": "20GiB"},
                                  offload_folder=str(ROOT / ".cache/offload"))
                    try:
                        import bitsandbytes
                        from transformers import BitsAndBytesConfig
                        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
                    except ImportError:
                        pass
                self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
                if key == "qwen":
                    self.processor.image_processor.max_pixels = 512 * 512
                try:
                    self.pipeline = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
                except (ImportError, ValueError, RuntimeError):
                    if "quantization_config" not in kwargs:
                        raise
                    kwargs.pop("quantization_config")
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.pipeline = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
                if "device_map" not in kwargs:
                    self.pipeline.to("cuda" if cuda else "cpu")
                self.pipeline.eval()
            self.loaded = signature
        except Exception:
            self.unload()
            raise

    def vision(self, prompt, images: list[Image.Image], key="smolvlm", history=None):
        import torch
        with self.mutex, self.lease():
            self.load(key)
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": row["role"], "content": [{"type": "text", "text": row["content"]}]}
                        for row in (history or [])[-8:]]
            messages.append({"role": "user", "content": content})
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                                                        tokenize=True, return_dict=True, return_tensors="pt")
            inputs = inputs.to(self.pipeline.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.pipeline.dtype)
            try:
                with torch.inference_mode():
                    outputs = self.pipeline.generate(**inputs, max_new_tokens=384, do_sample=False)
                result = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            except torch.cuda.OutOfMemoryError:
                self.unload()
                raise RuntimeError("VRAM insuficiente para visão. Reduza páginas/imagens ou use SmolVLM2.")
            return {"text": result, "model": MODELS[key]["repo"], "seconds": None}

    def generate(self, prompt, mode="text-to-image", seed=42, width=512, height=512,
                 steps=25, strength=0.65, image=None, mask=None, adapter=None, negative_prompt=""):
        import torch
        with self.mutex, self.lease():
            start = time.monotonic()
            image = image.resize((width, height)) if image else None
            mask = mask.convert("L").resize((width, height)) if mask else None
            if mode != "text-to-image" and image is None:
                raise ValueError("Envie imagem inicial para edição.")
            if mode == "inpainting" and mask is None:
                raise ValueError("Envie máscara branca na área que deseja editar.")
            args = dict(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                        generator=torch.Generator(device="cpu").manual_seed(seed))
            if mode == "text-to-image":
                args.update(width=width, height=height)
            else:
                args.update(image=image, strength=strength)
            if mode == "inpainting":
                args.update(mask_image=mask, width=width, height=height)
            for sequential in (False, True):
                try:
                    self.load("sd15", mode, adapter, sequential)
                    # Reinicia a semente também na tentativa com offloading sequencial.
                    args["generator"].manual_seed(seed)
                    with torch.inference_mode():
                        result = self.pipeline(**args)
                    if result.nsfw_content_detected and any(result.nsfw_content_detected):
                        raise ValueError("A geração foi bloqueada pelo filtro de segurança do modelo-base.")
                    output = result.images[0]
                    break
                except torch.cuda.OutOfMemoryError:
                    self.unload()
                    if sequential:
                        raise RuntimeError("VRAM insuficiente mesmo com offloading sequencial; reduza a resolução.")
            name = f"{uuid.uuid4().hex}.png"
            output.save(ROOT / "outputs" / name)
            report = {"file": name, "url": f"/outputs/{name}", "prompt": prompt, "seed": seed,
                      "mode": mode, "adapter": adapter, "width": width, "height": height,
                      "seconds": round(time.monotonic() - start, 2), "steps": steps,
                      "base_model": MODELS["sd15"]["repo"]}
            write_json(ROOT / "outputs" / name.replace(".png", ".json"), report)
            return report


engine = Engine()
