"""Interface pública IGNIS: criação, edição, comparação e receitas de imagens."""
import json
import os
import time
import uuid
import zipfile
from datetime import datetime, timezone

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from settings import ROOT, RELEASE, MODEL_ID, TRAINING, CONFIG, EVALUATION, PRESETS, SIZES, SAMPLERS, EXAMPLES, recipe, restore_recipe

OUT = ROOT / ".outputs"
OUT.mkdir(exist_ok=True)
if os.environ.get("IGNIS_ZERO_GPU") == "1":
    import runtime


def persist(results, seconds, previous):
    # Remove somente saídas geradas antigas; os fontes e pesos não são tocados.
    for old in OUT.iterdir():
        if old.is_file() and old.suffix in (".png", ".json", ".zip") and time.time() - old.stat().st_mtime > 86400:
            old.unlink(missing_ok=True)
    history, records, paths = list(previous or []), [], []
    for image, metadata in results:
        identifier = uuid.uuid4().hex
        metadata = {**metadata, "created_at": datetime.now(timezone.utc).isoformat(), "request_inference_seconds": seconds}
        png, info = OUT / f"{identifier}.png", OUT / f"{identifier}.json"
        embedded = PngInfo()
        embedded.add_text("ignis", json.dumps(metadata, ensure_ascii=False))
        image.save(png, pnginfo=embedded)
        info.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {"image": str(png), "json": str(info), "metadata": metadata}
        records.append(row); history.append(row); paths.extend([png, info])
    bundle = OUT / f"{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.name)
    history = history[-24:]
    gallery = [(row["image"], f"{row['metadata']['model']} · seed {row['metadata']['seed']}") for row in reversed(history)]
    return records, history, gallery, str(bundle)


def create_images(prompt, negative, seed, randomize, size, steps, guidance, scale, model, sampler, preset, count, history,
                  progress=gr.Progress(track_tqdm=True)):
    try:
        options = recipe(prompt, negative, seed, randomize, size, steps, guidance, scale, model, sampler, preset, count)
        progress(0, desc="Executando a geração")
        from runtime import generate
        results, seconds = generate(options)
        records, history, gallery, bundle = persist(results, seconds, history)
        progress(1, desc="Imagens e receitas salvas")
        return ([(r["image"], f"Seed {r['metadata']['seed']}") for r in records], options["seed"],
                f"{len(records)} imagem(ns) · {seconds}s de inferência · seed {options['seed']}",
                bundle, records[0]["image"], records[0]["json"], history, gallery, records[0]["metadata"])
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def compare_images(prompt, negative, seed, size, steps, guidance, scale, sampler, history, progress=gr.Progress(track_tqdm=True)):
    try:
        options = recipe(prompt, negative, seed, False, size, steps, guidance, scale, "IGNIS LoRA", sampler, "Sem estilo adicional")
        progress(0, desc="Gerando o par com a mesma semente")
        from runtime import compare
        results, seconds = compare(options)
        records, history, gallery, bundle = persist(results, seconds, history)
        return (results[0][0], results[1][0], (results[0][0], results[1][0]),
                f"Base à esquerda · IGNIS à direita · seed {options['seed']} · {seconds}s de inferência", bundle, history, gallery)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def edit_image(prompt, image, mask, mode, strength, seed, steps, scale, history, progress=gr.Progress(track_tqdm=True)):
    try:
        options = recipe(prompt, "", seed, False, next(iter(SIZES)), steps, 7.5, scale, "IGNIS LoRA", SAMPLERS[0], "Sem estilo adicional")
        progress(0, desc="Aplicando a edição")
        from runtime import edit
        results, seconds = edit(options, image, mask, mode, strength)
        records, history, gallery, bundle = persist(results, seconds, history)
        return results[0][0], f"{mode} · seed {int(seed)} · {seconds}s de inferência", bundle, history, gallery
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def import_recipe(raw):
    try:
        options, size = restore_recipe(raw)
        return (options["prompt"], options["negative_prompt"], options["seed"], False, size, options["steps"],
                options["guidance_scale"], options["lora_scale"], options["model"], options["sampler"], "Sem estilo adicional")
    except Exception as exc:
        raise gr.Error(f"Receita inválida: {exc}") from exc


def last_image(history):
    if not history:
        raise gr.Error("Gere uma imagem no Estúdio ou envie um arquivo para editar.")
    with Image.open(history[-1]["image"]) as image:
        return image.copy()


def gallery_selection(history, event: gr.SelectData):
    index = event.index if isinstance(event.index, int) else event.index[0]
    row = list(reversed(history))[index]
    return row["metadata"], row["image"], row["json"]


def training_summary():
    if not TRAINING:
        return "Os registros do treinamento serão apresentados após o empacotamento."
    rows = {"Atualizações": TRAINING.get("step"), "Resolução": CONFIG.get("resolution"), "Rank LoRA": CONFIG.get("rank"),
            "Parada": TRAINING.get("stop_reason"), "Loss inicial": TRAINING.get("initial_validation_loss"),
            "Melhor loss observada": TRAINING.get("best_validation_loss")}
    return "### Um experimento, com evidências\n\n| Registro | Valor |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows.items()) + "\n\nA melhor loss pode pertencer a um checkpoint anterior ao adaptador final. A loss não comprova superioridade visual."


def build_app():
    theme = gr.themes.Soft(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate", font=["Segoe UI", "Arial", "sans-serif"])
    with gr.Blocks(title="IGNIS Image Lab", analytics_enabled=False, delete_cache=(3600, 86400),
                   theme=theme, css=(ROOT / "styles.css").read_text(encoding="utf-8")) as application:
        state = gr.State([], time_to_live=3600)
        gr.HTML(f'''<div class="ignis-header"><a class="ignis-brand" href="https://huggingface.co/{MODEL_ID}" target="_blank" rel="noopener"><span class="brand-symbol">i</span><span>IGNIS <small>Image Lab</small></span></a><span class="release-label">LoRA experimental · v1</span></div><div class="ignis-hero"><div><h1>Seu próximo<br>experimento visual.</h1><p>Crie imagens, explore variações e descubra o que muda com o adaptador IGNIS.</p><div class="hero-links"><a href="https://huggingface.co/{MODEL_ID}" target="_blank" rel="noopener">Conhecer o modelo</a><a href="https://huggingface.co/{MODEL_ID}/tree/main" target="_blank" rel="noopener">Pesos e arquivos</a></div></div><div class="hero-art" aria-hidden="true"><div class="art-square one"></div><div class="art-square two"></div><div class="art-square three"></div><span>Texto. Forma. Possibilidade.</span></div></div>''')
        gr.Markdown("**Base:** Stable Diffusion 1.5 · **Especialização:** IGNIS LoRA · **ZeroGPU:** filas e quotas quando disponível.", elem_classes="context-bar")
        with gr.Tabs():
            with gr.Tab("Estúdio"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=300, elem_classes="control-panel"):
                        gr.Markdown("### Descreva sua ideia\nPrompts em inglês costumam funcionar melhor nesta base.")
                        prompt = gr.Textbox(label="Prompt", placeholder="A quiet botanical garden at sunrise, delicate watercolor…", lines=4)
                        with gr.Row():
                            preset = gr.Dropdown(list(PRESETS), value="Sem estilo adicional", label="Direção visual")
                            model = gr.Dropdown(["IGNIS LoRA", "Stable Diffusion 1.5"], value="IGNIS LoRA", label="Modelo")
                        generate = gr.Button("Gerar imagem", variant="primary", size="lg", elem_id="generate-button")
                        with gr.Accordion("Ajustar a geração", open=False):
                            negative = gr.Textbox(label="Prompt negativo")
                            with gr.Row():
                                seed = gr.Number(label="Seed", value=42, precision=0, minimum=0, maximum=2**32 - 1)
                                randomize = gr.Checkbox(value=True, label="Sortear seed")
                            size = gr.Dropdown(list(SIZES), value=next(iter(SIZES)), label="Enquadramento")
                            with gr.Row():
                                steps = gr.Slider(5, 40, value=25, step=1, label="Passos")
                                guidance = gr.Slider(1, 15, value=7.5, step=.5, label="Orientação do prompt")
                            scale = gr.Slider(0, 1.5, value=1, step=.05, label="Intensidade LoRA")
                            sampler = gr.Dropdown(SAMPLERS, value=SAMPLERS[0], label="Amostrador")
                            count = gr.Radio([1, 2], value=1, label="Imagens por solicitação")
                        gr.Markdown("#### Um ponto de partida")
                        gr.Examples(EXAMPLES, inputs=[prompt, preset], cache_examples=False)
                        with gr.Accordion("Importar uma receita", open=False):
                            recipe_file = gr.File(label="Receita JSON", file_types=[".json"], type="binary")
                    with gr.Column(scale=7, min_width=320, elem_classes="result-panel"):
                        output = gr.Gallery(label="Suas imagens", columns=1, rows=1, height=520, object_fit="contain", interactive=False)
                        status = gr.Markdown("Sua imagem aparecerá aqui. Escolha um exemplo ou escreva um prompt.")
                        with gr.Row():
                            png_download = gr.File(label="PNG com receita incorporada", interactive=False)
                            json_download = gr.File(label="Receita JSON", interactive=False)
                        bundle = gr.File(label="Pacote completo · ZIP", interactive=False)
                        with gr.Accordion("Parâmetros usados", open=False):
                            metadata = gr.JSON(label="Receita reproduzível")
            with gr.Tab("Comparar"):
                gr.Markdown("### A mesma ideia. Dois resultados.\nCompare base e adaptador com prompt, seed, amostrador e resolução idênticos.")
                compare_prompt = gr.Textbox(label="Prompt", value="A watercolor landscape with mountains and a lake", lines=2)
                with gr.Row():
                    compare_seed = gr.Number(value=42, precision=0, label="Seed", minimum=0, maximum=2**32 - 1)
                    compare_size = gr.Dropdown(list(SIZES), value=next(iter(SIZES)), label="Enquadramento")
                    compare_steps = gr.Slider(5, 40, value=25, step=1, label="Passos")
                with gr.Accordion("Ajustes da comparação", open=False):
                    compare_negative = gr.Textbox(label="Prompt negativo")
                    compare_guidance = gr.Slider(1, 15, value=7.5, step=.5, label="Orientação")
                    compare_scale = gr.Slider(0, 1.5, value=1, step=.05, label="Intensidade LoRA")
                    compare_sampler = gr.Dropdown(SAMPLERS, value=SAMPLERS[0], label="Amostrador")
                compare_button = gr.Button("Gerar comparação", variant="primary")
                slider = gr.ImageSlider(label="Deslize para comparar · base / IGNIS", type="pil")
                with gr.Accordion("Ver as duas imagens", open=False):
                    with gr.Row():
                        base_output = gr.Image(label="Stable Diffusion 1.5", type="pil", interactive=False)
                        tuned_output = gr.Image(label="IGNIS LoRA", type="pil", interactive=False)
                compare_status = gr.Markdown("A comparação visual não representa uma pontuação automática de qualidade.")
                compare_bundle = gr.File(label="Comparação e receitas", interactive=False)
            with gr.Tab("Editar"):
                gr.Markdown("### Explore uma nova versão\nAltere a imagem inteira ou envie uma máscara: branco para editar. O inpainting usa o UNet de quatro canais do SD 1.5.")
                with gr.Row():
                    edit_input = gr.Image(label="Imagem inicial", type="pil", sources=["upload", "clipboard"])
                    edit_mask = gr.Image(label="Máscara opcional", type="pil", sources=["upload", "clipboard"])
                    edit_output = gr.Image(label="Resultado", type="pil", interactive=False)
                use_last = gr.Button("Usar minha última imagem")
                edit_prompt = gr.Textbox(label="Descreva o resultado desejado", lines=2)
                with gr.Row():
                    edit_mode = gr.Dropdown([("Imagem inteira", "image-to-image"), ("Região com máscara", "inpainting")], value="image-to-image", label="Operação")
                    strength = gr.Slider(.1, .95, value=.65, step=.05, label="Intensidade da mudança")
                    edit_seed = gr.Number(value=42, precision=0, minimum=0, maximum=2**32 - 1, label="Seed")
                with gr.Row():
                    edit_steps = gr.Slider(5, 40, value=25, step=1, label="Passos")
                    edit_scale = gr.Slider(0, 1.5, value=1, step=.05, label="Intensidade LoRA")
                edit_button = gr.Button("Aplicar edição", variant="primary")
                edit_status = gr.Markdown("O resultado depende da imagem, do prompt e da intensidade.")
                edit_bundle = gr.File(label="PNG e receita da edição", interactive=False)
            with gr.Tab("Minha sessão"):
                gr.Markdown("### Seu caderno de experimentos\nAté 24 imagens desta sessão. Selecione uma para consultar parâmetros e baixar arquivos. O histórico é temporário.")
                session_gallery = gr.Gallery(label="Experimentos", columns=4, height=430, interactive=False)
                selected_metadata = gr.JSON(label="Parâmetros")
                with gr.Row():
                    selected_png = gr.File(label="PNG", interactive=False)
                    selected_json = gr.File(label="JSON", interactive=False)
            with gr.Tab("Pesquisa e documentação"):
                gr.Markdown(training_summary())
                samples = [(str(ROOT / "assets" / p["sample"]), f"Seed {p['seed']} · {p['prompt']}") for p in EVALUATION.get("pairs", []) if (ROOT / "assets" / p["sample"]).is_file()]
                gr.Gallery(value=samples, label="Avaliação publicada · base à esquerda / IGNIS à direita", columns=2, height=500, interactive=False)
                with gr.Accordion("Ler a documentação completa", open=True):
                    card = ROOT / "MODEL_CARD.md"
                    gr.Markdown(card.read_text(encoding="utf-8") if card.exists() else f"[Model card no Hugging Face](https://huggingface.co/{MODEL_ID})")
                gr.Markdown("Referências de experiência: [FLUX.1 schnell](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell) e [Stable Diffusion 3.5 Large](https://huggingface.co/spaces/stabilityai/stable-diffusion-3.5-large). Interface IGNIS própria.")
        controls = [prompt, negative, seed, randomize, size, steps, guidance, scale, model, sampler, preset]
        generate.click(create_images, controls + [count, state], [output, seed, status, bundle, png_download, json_download, state, session_gallery, metadata], concurrency_id="inference", concurrency_limit=1)
        recipe_file.upload(import_recipe, [recipe_file], controls)
        compare_button.click(compare_images, [compare_prompt, compare_negative, compare_seed, compare_size, compare_steps, compare_guidance, compare_scale, compare_sampler, state], [base_output, tuned_output, slider, compare_status, compare_bundle, state, session_gallery], concurrency_id="inference", concurrency_limit=1)
        edit_button.click(edit_image, [edit_prompt, edit_input, edit_mask, edit_mode, strength, edit_seed, edit_steps, edit_scale, state], [edit_output, edit_status, edit_bundle, state, session_gallery], concurrency_id="inference", concurrency_limit=1)
        use_last.click(last_image, [state], [edit_input], queue=False)
        session_gallery.select(gallery_selection, [state], [selected_metadata, selected_png, selected_json], queue=False)
        gr.HTML(f'<div class="ignis-footer"><span>IGNIS · Vinicius de Souza Santos</span><span>Stable Diffusion 1.5 + LoRA · versão {str(RELEASE.get("model_revision", "a configurar"))[:8]}</span></div>')
    return application


if __name__ == "__main__":
    demo = build_app()
    demo.queue(default_concurrency_limit=1, max_size=12).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
        max_file_size="15mb", allowed_paths=[str(OUT), str(ROOT / "assets")])
