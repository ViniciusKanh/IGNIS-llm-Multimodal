"""Testes sem pesos: invariantes de dados, API, bloqueios e seleção de arquivos."""
import io
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend.common import ROOT, inside, model_path, read_json, write_json
from backend.engine import Engine
from scripts.download_models import select_files
from scripts.prepare_dataset import pixel_hash, split_records
from scripts.train_lora import validate_manifest


@pytest.fixture
def client():
    if os.environ.get("IGNIS_TEST_HTTP") != "1":
        pytest.skip("Testes HTTP requerem IGNIS_TEST_HTTP=1 e suporte de loopback do sistema.")
    return TestClient(app, raise_server_exceptions=False)


def test_real_route_handlers_without_socket():
    from backend.app import health, status, lab, get_config
    assert health()["status"] == "ok"
    resources = status()
    assert isinstance(resources["cuda"], bool)
    assert resources["ram_free_mb"] > 0
    assert len(resources["models"]) == 3
    assert set(lab()) >= {"dataset", "runs", "adapters", "evaluations"}
    assert get_config()["rank"] == 8


def test_missing_weights_rejected_by_real_engine():
    if (ROOT / "models/manifest.json").exists():
        pytest.skip("Contrato para instalação sem pesos.")
    with pytest.raises(FileNotFoundError, match="não instalado"):
        Engine().load("smolvlm")
    with pytest.raises(FileNotFoundError, match="não instalado"):
        Engine().generate("An oil painting of a garden")


def test_openapi_exposes_four_modules():
    paths = app.openapi()["paths"]
    for path in ("/api/chat", "/api/images", "/api/lab", "/api/status", "/api/jobs", "/api/config"):
        assert path in paths


def test_config_validation_without_http():
    from backend.app import TrainConfig
    from pydantic import ValidationError
    config = read_json(ROOT / "configs/train.json")
    for override in ({"max_steps": 2}, {"learning_rate": -1}, {"name": "../escape"}):
        with pytest.raises(ValidationError):
            TrainConfig(**{**config, **override})


@pytest.mark.parametrize("prompt,image,mask,expected", [
    ("Gere uma imagem de uma montanha", False, False, "text-to-image"),
    ("Edite a iluminação desta foto", True, False, "image-to-image"),
    ("O que aparece nesta imagem?", True, False, "vision"),
    ("Descreva a área selecionada", True, True, "inpainting"),
    ("Explique como criar uma imagem", False, False, "vision"),
])
def test_core_routes_requests(prompt, image, mask, expected):
    from backend.core import infer_task
    assert infer_task(prompt, image, mask) == expected


def test_health_and_real_resources(client):
    assert client.get("/api/health").json()["status"] == "ok"
    status = client.get("/api/status").json()
    assert isinstance(status["cuda"], bool)
    assert len(status["models"]) == 3
    assert status["ram_free_mb"] > 0


def test_lab_handles_absent_experiment(client):
    response = client.get("/api/lab")
    assert response.status_code == 200
    assert set(response.json()) >= {"dataset", "runs", "adapters", "evaluations"}


def test_missing_models_fail_explicitly(client):
    if (ROOT / "models/manifest.json").exists():
        pytest.skip("Contrato específico para instalação sem pesos.")
    response = client.post("/api/models/load", json={"model": "smolvlm"})
    assert response.status_code == 503
    assert "não instalado" in response.json()["detail"]


def test_chat_without_weights_does_not_invent_answer(client):
    if (ROOT / "models/manifest.json").exists():
        pytest.skip("Contrato específico para instalação sem pesos.")
    response = client.post("/api/chat", data={"prompt": "Descreva a imagem", "session": "contract-test"})
    assert response.status_code == 503
    assert "não instalado" in response.json()["detail"]
    assert client.get("/api/chat/contract-test").json() == []


def test_generation_does_not_create_fake_output(client):
    if (ROOT / "models/manifest.json").exists():
        pytest.skip("Contrato específico para instalação sem pesos.")
    previous = set((ROOT / "outputs").glob("*.png"))
    response = client.post("/api/images", data={"prompt": "An oil painting of a garden"})
    assert response.status_code == 503
    assert set((ROOT / "outputs").glob("*.png")) == previous


def test_invalid_resolution_rejected_before_inference(client):
    response = client.post("/api/images", data={"prompt": "test", "width": "513"})
    assert response.status_code == 422


def test_unknown_model_rejected(client):
    assert client.post("/api/models/load", json={"model": "../../other"}).status_code == 422


def test_foreign_origin_cannot_start_jobs(client):
    response = client.post("/api/jobs", json={"action": "train"}, headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 403


def test_unknown_job_has_explicit_404(client):
    assert client.get("/api/jobs/does-not-exist/log").status_code == 404


def test_api_config_rejects_tiny_run_and_bad_parameters(client):
    config = client.get("/api/config").json()
    assert client.post("/api/config", json={**config, "max_steps": 2}).status_code == 422
    assert client.post("/api/config", json={**config, "learning_rate": -1}).status_code == 422
    assert client.post("/api/config", json={**config, "name": "../escape"}).status_code == 422


@pytest.mark.parametrize("relative", ["../elsewhere", "../../models", "C:/Windows/system32"])
def test_path_cannot_escape_project(relative):
    with pytest.raises(ValueError):
        inside(ROOT / "datasets", relative)


def test_atomic_json_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    write_json(path, {"name": "validação", "steps": 0})
    assert read_json(path) == {"name": "validação", "steps": 0}
    assert len(list(tmp_path.iterdir())) == 1


def test_pixel_hash_is_content_based():
    first = Image.new("RGB", (16, 16), (100, 10, 9))
    second = first.copy()
    assert pixel_hash(first) == pixel_hash(second)
    second.putpixel((0, 0), (0, 0, 0))
    assert pixel_hash(first) != pixel_hash(second)


def test_deduplication_and_reproducible_split():
    # Imagens sintéticas servem exclusivamente como fixtures de teste, nunca como treino reportado.
    rows = [{"sha256": str(i), "file": f"{i}.png", "caption": "fixture"} for i in range(20)]
    result = split_records(rows + [rows[0]], seed=42)
    train = {r["sha256"] for r in result["train"]}
    validation = {r["sha256"] for r in result["validation"]}
    assert not train & validation
    assert len(train | validation) == 20
    assert result == split_records(list(reversed(rows)), seed=42)


def test_insufficient_dataset_rejected():
    with pytest.raises(ValueError):
        split_records([{"sha256": "same"}] * 12)


def test_cross_split_leakage_rejected():
    rows = [{"sha256": str(i)} for i in range(10)]
    with pytest.raises(ValueError, match="Vazamento"):
        validate_manifest({"train": rows, "validation": [{"sha256": "0"}, {"sha256": "new"}]})


def test_download_selection_avoids_duplicate_sd_weights():
    files = ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
             "unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.bin",
             "vae/diffusion_pytorch_model.fp16.safetensors", "v1-5-pruned.safetensors", "v1-5.ckpt",
             "tokenizer/vocab.json", "README.md", "LICENSE"]
    selected = select_files(files, "sd15")
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in selected
    assert "unet/diffusion_pytorch_model.safetensors" not in selected
    assert "v1-5-pruned.safetensors" not in selected
    assert not any(f.endswith((".bin", ".ckpt")) for f in selected)
    assert "LICENSE" in selected


def test_vision_download_keeps_processor_and_shards():
    files = ["config.json", "processor_config.json", "model-00001-of-00002.safetensors",
             "model-00002-of-00002.safetensors", "model.safetensors.index.json", "model.onnx", "onnx/model.safetensors"]
    selected = select_files(files, "qwen")
    assert "processor_config.json" in selected
    assert "model.safetensors.index.json" in selected
    assert len([n for n in selected if n.endswith(".safetensors")]) == 2


def test_gpu_lock_excludes_other_engine():
    from filelock import Timeout
    first, second = Engine(), Engine()
    with first.lease():
        with pytest.raises(Timeout):
            with second.lease():
                pass


def test_training_stop_requires_running_job(client):
    assert client.post("/api/training/stop").status_code == 409


def test_hf_release_rejects_absent_training():
    from scripts.prepare_hf_release import collect_evidence
    with pytest.raises(RuntimeError, match="faltam pesos"):
        collect_evidence("nonexistent-release-test/final")


def test_hf_release_rejects_invalid_repository_identifier():
    from scripts.prepare_hf_release import prepare
    with pytest.raises(ValueError, match="Identificador"):
        prepare(model_id="../../outside")


def test_diffusiondb_zip_rejects_duplicate_prompts_and_paths(tmp_path, monkeypatch):
    import zipfile
    import scripts.prepare_dataset as module
    monkeypatch.setattr(module, "ROOT", tmp_path)
    folder = tmp_path / "datasets/images"
    archive = tmp_path / "fixture.zip"
    metadata = {}
    with zipfile.ZipFile(archive, "w") as stream:
        for index, caption in enumerate([
            "A watercolor landscape with mountains", "A watercolor landscape with mountains",
            "An oil painting of a quiet garden", "nsfw nude photograph of a person",
        ]):
            name = f"fixture-{index}.png"
            buffer = io.BytesIO()
            Image.new("RGB", (256, 256), (index * 40, 0, 0)).save(buffer, format="PNG")
            stream.writestr(name, buffer.getvalue())
            metadata[name] = {"p": caption}
        metadata["../outside.png"] = {"p": "A valid caption attached to an invalid path"}
        stream.writestr("part-000001.json", json.dumps(metadata))
    rows = module.diffusiondb_records(archive, folder, 96, "test-revision")
    assert len(rows) == 2
    assert len({row["caption"] for row in rows}) == 2
    assert all(row["synthetic"] and row["license"] == "cc0-1.0" for row in rows)
    assert not (tmp_path / "outside.png").exists()


def test_public_dataset_falls_back_on_http_403(monkeypatch):
    import requests
    import scripts.prepare_dataset as module
    def denied(limit):
        raise requests.HTTPError("403 Forbidden")
    monkeypatch.setattr(module, "artic_dataset", denied)
    monkeypatch.setattr(module, "write_json", lambda *args: None)
    monkeypatch.setattr(module, "diffusiondb_dataset", lambda limit: {"source": "diffusiondb", "limit": limit})
    assert module.public_dataset(96) == {"source": "diffusiondb", "limit": 96}
