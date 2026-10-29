"""Demonstração pública de inferência: sem ferramentas administrativas do projeto local."""
import gc
import json
import os
import threading
from pathlib import Path

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

RELEASE = json.loads(Path("release.json").read_text()) if Path("release.json").exists() else {}
MODEL_ID = os.environ.get("IGNIS_MODEL_ID", RELEASE.get("model_id", "ViniciusKhan/ignis-image-lora-v1"))
MODEL_REVISION = RELEASE.get("model_revision")
BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LOCK = threading.Lock()
pipeline = None
loaded_adapter = False


def load(use_adapter):
    global pipeline, loaded_adapter
    if pipeline is None:
        cuda = torch.cuda.is_available()
        pipeline = StableDiffusionPipeline.from_pretrained(
            BASE, revision=RELEASE.get("base_revision"),
            torch_dtype=torch.float16 if cuda else torch.float32, use_safetensors=True,
        )
        pipeline.enable_vae_slicing()
        if cuda:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to("cpu")
    if use_adapter and not loaded_adapter:
        pipeline.load_lora_weights(MODEL_ID, revision=MODEL_REVISION)
        loaded_adapter = True
    if loaded_adapter:
        pipeline.enable_lora() if use_adapter else pipeline.disable_lora()


def generate_locked(prompt, negative, seed, use_adapter):
    load(use_adapter)
    with torch.inference_mode():
        result = pipeline(prompt=prompt, negative_prompt=negative,
                          generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                          width=512 if torch.cuda.is_available() else 384,
                          height=512 if torch.cuda.is_available() else 384,
                          num_inference_steps=25 if torch.cuda.is_available() else 15)
    if result.nsfw_content_detected and any(result.nsfw_content_detected):
        raise ValueError("Saída bloqueada pelo filtro de segurança da base.")
    return result.images[0]


def run(prompt, negative, seed, model):
    if not prompt.strip():
        raise gr.Error("Informe um prompt.")
    with LOCK:
        try:
            image = generate_locked(prompt, negative, seed, model == "IGNIS LoRA")
            return image, f"{model} · seed {int(seed)} · {'CUDA' if torch.cuda.is_available() else 'CPU'}"
        except Exception as exc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error(f"Não foi possível executar a geração: {exc}") from exc


def compare(prompt, negative, seed):
    if not prompt.strip():
        raise gr.Error("Informe um prompt.")
    with LOCK:
        try:
            base = generate_locked(prompt, negative, seed, False)
            tuned = generate_locked(prompt, negative, seed, True)
            return base, tuned
        except Exception as exc:
            raise gr.Error(f"Comparação não concluída: {exc}") from exc


CSS = """.gradio-container{max-width:1250px!important;background:#f6f9fc;color:#10243b} .hero{padding:28px;border-left:4px solid #00abc2;background:white;border-radius:12px;margin-bottom:22px} .hero h1{font-size:32px;color:#10243b} .hero p{color:#64778c} footer{display:none!important}"""
with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"), css=CSS, title="IGNIS Image Lab") as demo:
    gr.HTML('<div class="hero"><small>IGNIS / IMAGE RESEARCH</small><h1>Da ideia à imagem.</h1><p>Explore o adaptador IGNIS e compare seus resultados com a base.</p></div>')
    gr.Markdown(f"Adaptador: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) · Base: Stable Diffusion 1.5 · {'GPU disponível' if torch.cuda.is_available() else 'CPU: as gerações podem demorar'}")
    with gr.Tab("Estúdio"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Descreva a imagem em inglês…")
                negative = gr.Textbox(label="Prompt negativo")
                seed = gr.Number(label="Seed", value=42, precision=0, minimum=0, maximum=4294967295)
                model = gr.Radio(["IGNIS LoRA", "Stable Diffusion 1.5"], value="IGNIS LoRA", label="Modelo")
                button = gr.Button("Gerar imagem", variant="primary")
            with gr.Column():
                output = gr.Image(label="Resultado", type="pil")
                details = gr.Textbox(label="Execução", interactive=False)
        gr.Examples([["A detailed oil painting of a quiet garden, soft morning light"],
                     ["A watercolor landscape with mountains and a lake"]], inputs=prompt)
        button.click(run, [prompt, negative, seed, model], [output, details])
    with gr.Tab("Base × IGNIS"):
        gr.Markdown("O mesmo prompt, a mesma seed e os mesmos parâmetros são usados nas duas gerações. A comparação visual não é uma métrica quantitativa de qualidade.")
        comparison_prompt = gr.Textbox(label="Prompt para comparação", lines=3)
        comparison_negative = gr.Textbox(label="Prompt negativo")
        comparison_seed = gr.Number(label="Seed", value=42, precision=0, minimum=0, maximum=4294967295)
        compare_button = gr.Button("Comparar", variant="primary")
        with gr.Row():
            base_output = gr.Image(label="Stable Diffusion 1.5 — base", type="pil")
            tuned_output = gr.Image(label="IGNIS — base + adaptador", type="pil")
        compare_button.click(compare, [comparison_prompt, comparison_negative, comparison_seed], [base_output, tuned_output])
    with gr.Tab("Documentação"):
        card = Path("MODEL_CARD.md")
        gr.Markdown(card.read_text(encoding="utf-8") if card.exists() else f"Consulte a [documentação completa do modelo](https://huggingface.co/{MODEL_ID}).")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), max_file_size="10mb")
