"""API local do IGNIS; falhas de modelos e dependências permanecem explícitas."""
from __future__ import annotations

import io
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from filelock import Timeout
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from backend.common import MODELS, ROOT, inside, read_json, write_json
from backend.engine import engine
from backend.jobs import jobs

app = FastAPI(title="IGNIS Multimodal", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.mount("/outputs", StaticFiles(directory=ROOT / "outputs"), name="outputs")


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method != "GET" and origin and origin not in {
        "http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000", "http://localhost:8000"
    }:
        return JSONResponse(status_code=403, content={"detail": "Origem não autorizada para a API local."})
    return await call_next(request)


@app.exception_handler(Exception)
async def explicit_error(request, exc):
    code = 409 if isinstance(exc, Timeout) else 422 if isinstance(exc, ValueError) else 503
    return JSONResponse(status_code=code, content={"detail": str(exc) or type(exc).__name__})


def database():
    connection = sqlite3.connect(ROOT / "outputs/history.sqlite")
    connection.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, session TEXT, role TEXT, content TEXT, model TEXT)")
    connection.execute("CREATE TABLE IF NOT EXISTS assets (message_id INTEGER, file TEXT)")
    return connection


def history(session):
    with database() as connection:
        rows = connection.execute("SELECT id, role, content, model FROM messages WHERE session=? ORDER BY id", (session,)).fetchall()
        result = []
        for identifier, role, content, model in rows:
            assets = connection.execute("SELECT file FROM assets WHERE message_id=?", (identifier,)).fetchall()
            result.append({"role": role, "content": content, "model": model,
                           "attachments": ["/outputs/" + row[0] for row in assets]})
    return result


def upload_image(upload, max_size=20 * 1024 * 1024):
    raw = upload.file.read(max_size + 1)
    if len(raw) > max_size:
        raise HTTPException(413, "Imagem maior que 20 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        if image.width * image.height > 24_000_000:
            raise HTTPException(413, "Imagem excede 24 megapixels.")
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, "Arquivo de imagem inválido.")


@app.get("/api/health")
def health():
    return {"status": "ok", "application": "IGNIS Multimodal", "inference": "requires_installed_models"}


@app.get("/api/status")
def status():
    return engine.status()


class ModelAction(BaseModel):
    model: Literal["smolvlm", "qwen", "sd15"] = "smolvlm"


@app.post("/api/models/load")
def load_model(payload: ModelAction):
    with engine.mutex, engine.lease():
        engine.load(payload.model)
    return {"loaded": engine.loaded}


@app.post("/api/models/unload")
def unload_model():
    with engine.mutex, engine.lease():
        engine.unload()
    return {"loaded": None}


@app.get("/api/chat/{session}")
def get_chat(session: str):
    return history(session)


@app.post("/api/chat")
def chat(prompt: str = Form(..., min_length=1, max_length=16000),
         session: str = Form("default", max_length=128),
         model: Literal["smolvlm", "qwen"] = Form("smolvlm"),
         files: list[UploadFile] = File(default=[])):
    images = []
    if len(files) > 4:
        raise HTTPException(422, "Envie até quatro imagens ou um PDF com até quatro páginas.")
    for upload in files:
        if upload.content_type == "application/pdf":
            import fitz
            raw = upload.file.read(20 * 1024 * 1024 + 1)
            if len(raw) > 20 * 1024 * 1024:
                raise HTTPException(413, "PDF maior que 20 MB.")
            with fitz.open(stream=raw, filetype="pdf") as document:
                for page in document:
                    if len(images) >= 4:
                        break
                    scale = min(1.5, 1024 / max(page.rect.width, page.rect.height, 1))
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                    image.thumbnail((1024, 1024))
                    images.append(image)
        else:
            image = upload_image(upload)
            image.thumbnail((1024, 1024))
            images.append(image)
    if len(images) > 4:
        raise HTTPException(422, "Limite de quatro páginas/imagens por solicitação.")
    previous = history(session)
    if not images:
        for message in reversed(previous):
            if message.get("attachments"):
                for url in message["attachments"][:4]:
                    with Image.open(inside(ROOT / "outputs", url.removeprefix("/outputs/"))) as image:
                        images.append(image.convert("RGB"))
                break
    result = engine.vision(prompt, images, key=model, history=previous)
    with database() as connection:
        cursor = connection.execute("INSERT INTO messages(session,role,content,model) VALUES(?,?,?,?)",
                                    (session, "user", prompt, model))
        if files:
            (ROOT / "outputs/attachments").mkdir(exist_ok=True)
            for image in images:
                filename = f"attachments/{uuid.uuid4().hex}.png"
                image.save(ROOT / "outputs" / filename)
                connection.execute("INSERT INTO assets(message_id,file) VALUES(?,?)", (cursor.lastrowid, filename))
        connection.execute("INSERT INTO messages(session,role,content,model) VALUES(?,?,?,?)",
                           (session, "assistant", result["text"], model))
    return result


@app.post("/api/images")
def generate_image(prompt: str = Form(..., min_length=1, max_length=4000),
                   mode: Literal["text-to-image", "image-to-image", "inpainting"] = Form("text-to-image"),
                   seed: int = Form(42, ge=0, le=2**32 - 1),
                   width: int = Form(512, ge=256, le=768), height: int = Form(512, ge=256, le=768),
                   steps: int = Form(25, ge=5, le=80), strength: float = Form(0.65, ge=0.1, le=0.95),
                   negative_prompt: str = Form("", max_length=2000),
                   adapter: str | None = Form(None), image: UploadFile | None = File(None),
                   mask: UploadFile | None = File(None)):
    if width % 64 or height % 64:
        raise HTTPException(422, "Use dimensões múltiplas de 64.")
    return engine.generate(prompt, mode, seed, width, height, steps, strength,
                           upload_image(image) if image else None, upload_image(mask) if mask else None,
                           adapter or None, negative_prompt)


@app.get("/api/gallery")
def gallery():
    return [read_json(path) for path in sorted((ROOT / "outputs").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]]


@app.post("/api/core")
def orchestrate(prompt: str = Form(..., min_length=1, max_length=4000),
                task: Literal["auto", "vision", "text-to-image", "image-to-image", "inpainting", "describe-and-generate"] = Form("auto"),
                model: Literal["smolvlm", "qwen"] = Form("smolvlm"),
                seed: int = Form(42, ge=0, le=2**32 - 1),
                image: UploadFile | None = File(None), mask: UploadFile | None = File(None)):
    from backend.core import execute
    return execute(prompt, task, upload_image(image) if image else None,
                   upload_image(mask) if mask else None, model, seed)


@app.get("/api/lab")
def lab():
    manifest = read_json(ROOT / "datasets/manifest.json", {})
    runs = []
    for directory in sorted((ROOT / "training").iterdir()):
        if not directory.is_dir():
            continue
        metrics = directory / "metrics.jsonl"
        rows = []
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Uma escrita interrompida não deve impedir a inspeção do restante da execução.
                    continue
        runs.append({"name": directory.name, "config": read_json(directory / "config.json"),
                     "progress": read_json(directory / "progress.json"), "metrics": rows[-800:],
                     "checkpoints": [p.name for p in directory.glob("checkpoint-*") if (p / "progress.json").exists()]})
    adapters = [p.parent.relative_to(ROOT / "training").as_posix()
                for p in (ROOT / "training").rglob("pytorch_lora_weights.safetensors")]
    evaluations = []
    for report in (ROOT / "outputs/evaluations").glob("*/report.json"):
        evaluations.append({"url": "/outputs/" + report.relative_to(ROOT / "outputs").as_posix(),
                            "name": report.parent.name, "samples": ["/outputs/" + p.relative_to(ROOT / "outputs").as_posix()
                                                                     for p in report.parent.glob("pair-*.png")]})
    return {"dataset": {"train": len(manifest.get("train", [])), "validation": len(manifest.get("validation", [])),
                         "source": manifest.get("source"), "fingerprint": manifest.get("fingerprint")},
            "runs": runs, "adapters": adapters, "evaluations": evaluations}


class TrainConfig(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    rank: int = Field(8, ge=1, le=32)
    resolution: Literal[256, 384, 512] = 512
    batch_size: Literal[1] = 1
    gradient_accumulation: int = Field(4, ge=1, le=32)
    learning_rate: float = Field(0.0001, ge=0.000001, le=0.001)
    max_steps: int = Field(800, ge=300, le=10000)
    warmup_steps: int = Field(50, ge=0, le=500)
    checkpoint_every: int = Field(50, ge=10, le=500)
    validation_every: int = Field(50, ge=10, le=500)
    patience: int = Field(5, ge=1, le=20)
    min_steps: int = Field(300, ge=100, le=10000)
    seed: int = Field(42, ge=0, le=2**32 - 1)


@app.get("/api/config")
def get_config():
    return read_json(ROOT / "configs/train.json")


@app.post("/api/config")
def save_config(config: TrainConfig):
    if any(job["status"] in ("queued", "running") for job in jobs.list()):
        raise HTTPException(409, "Aguarde a operação atual antes de alterar a configuração.")
    old = read_json(ROOT / "training" / config.name / "config.json")
    values = config.model_dump()
    if config.min_steps > config.max_steps or config.warmup_steps >= config.max_steps:
        raise HTTPException(422, "min_steps e warmup_steps incompatíveis com max_steps.")
    if old and any(old.get(key) != value for key, value in values.items()):
        raise HTTPException(409, "Use outro nome de experimento para alterar uma execução existente.")
    write_json(ROOT / "configs/train.json", values)
    return values


class JobAction(BaseModel):
    action: Literal["prepare", "download", "train", "evaluate"]
    model: Literal["all", "smolvlm", "qwen", "sd15"] = "all"


@app.post("/api/jobs")
def start_job(payload: JobAction):
    try:
        if payload.action in ("train", "evaluate"):
            with engine.mutex, engine.lease():
                engine.unload()
        return jobs.submit(payload.action, payload.model)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/jobs")
def list_jobs():
    return jobs.list()


@app.get("/api/jobs/{identifier}/log")
def job_log(identifier: str):
    try:
        return {"log": jobs.log(identifier)}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/training/stop")
def stop_training():
    if not any(job["action"] == "train" and job["status"] == "running" for job in jobs.list()):
        raise HTTPException(409, "Não há treinamento iniciado pela aplicação.")
    (ROOT / "logs/stop-training").touch()
    return {"status": "stop_requested", "message": "Salvará checkpoint ao terminar a atualização atual."}


if (ROOT / "frontend/dist").exists():
    app.mount("/", StaticFiles(directory=ROOT / "frontend/dist", html=True), name="frontend")
