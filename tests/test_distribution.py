"""Contratos da distribuição Docker/GHCR, verificados sem Docker e sem GPU.

Garantem que a imagem não incorpora pesos, dados ou credenciais, que a porta fica em
loopback, que o workflow publica apenas com GITHUB_TOKEN e que o diagnóstico CUDA não
alega inferência quando ela não foi executada.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = ("models", "datasets", "training", "outputs", "logs", "configs")
# Dentro da imagem, Dockerfile/compose/workflow não são copiados (.dockerignore).
repository_files = pytest.mark.skipif(not (ROOT / "Dockerfile").exists(),
                                      reason="arquivos de distribuição ausentes (execução dentro da imagem)")


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


@repository_files
def test_dockerfile_is_multistage_non_root_and_without_weights():
    dockerfile = read("Dockerfile")
    assert dockerfile.count("\nFROM ") + dockerfile.startswith("FROM ") >= 2
    assert "AS frontend" in dockerfile and "AS runtime" in dockerfile
    assert re.search(r"^USER ignis$", dockerfile, re.M)
    assert "HEALTHCHECK" in dockerfile
    copied = re.findall(r"^COPY (?:--\S+ )*(\S+)", dockerfile, re.M)
    for source in copied:
        assert source.split("/")[0] not in {"models", "datasets", "training", "outputs", "logs", ".cache", ".venv"}, source
    assert "download.pytorch.org/whl/${TORCH_VARIANT}" in dockerfile
    assert re.search(r"ARG TORCH_VARIANT=cu128", dockerfile)
    assert "org.opencontainers.image.source=\"https://github.com/ViniciusKanh/IGNIS-llm-Multimodal\"" in dockerfile


@repository_files
def test_dockerignore_excludes_data_caches_and_secrets():
    ignored = {line.strip() for line in read(".dockerignore").splitlines() if line.strip() and not line.startswith("#")}
    required = {".git", ".venv", ".cache", "models", "datasets", "training", "outputs", "logs",
                "*.safetensors", "*.ckpt", "*.pt", "*.pth", "*.gguf", ".env", ".env.*", "*.pem", "*.key",
                "frontend/node_modules"}
    assert required <= ignored, required - ignored


@repository_files
def test_entrypoint_is_valid_posix_shell():
    entrypoint = ROOT / "docker/entrypoint.sh"
    assert entrypoint.read_bytes().startswith(b"#!/bin/sh\n")
    assert b"\r\n" not in entrypoint.read_bytes(), "CRLF quebra o entrypoint no Linux"
    subprocess.run(["sh", "-n", str(entrypoint)], check=True)
    assert 'exec python -m uvicorn backend.app:app' in entrypoint.read_text()


def load_compose(*names):
    service = {}
    for name in names:
        data = yaml.safe_load(read(name))
        for key, value in data["services"]["ignis"].items():
            service[key] = {**service.get(key, {}), **value} if isinstance(value, dict) else value
    return service


@repository_files
def test_compose_binds_loopback_and_persists_all_data():
    service = load_compose("compose.yaml")
    assert service["image"].startswith("${IGNIS_IMAGE:-ghcr.io/viniciuskanh/ignis-multimodal}")
    for port in service["ports"]:
        assert port.startswith("127.0.0.1:"), "a API não tem autenticação; publique só em loopback"
    targets = {volume.split(":")[-1] for volume in service["volumes"]}
    assert {f"/app/{name}" for name in DATA_DIRS} | {"/app/.cache"} <= targets
    assert service["init"] is True
    assert "IGNIS_ALLOWED_ORIGINS" in service["environment"]


@repository_files
def test_gpu_override_reserves_nvidia_device():
    service = load_compose("compose.yaml", "compose.gpu.yaml")
    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert devices[0]["driver"] == "nvidia" and "gpu" in devices[0]["capabilities"]
    assert service["ports"] == load_compose("compose.yaml")["ports"]


@repository_files
def test_workflow_publishes_only_with_github_token_to_ghcr():
    text = read(".github/workflows/docker-publish.yml")
    workflow = yaml.safe_load(text)
    secrets = set(re.findall(r"secrets\.([A-Z_]+)", text))
    assert secrets == {"GITHUB_TOKEN"}, secrets
    assert workflow["env"]["IMAGE_NAME"] == "ghcr.io/viniciuskanh/ignis-multimodal"
    docker = workflow["jobs"]["docker"]
    assert docker["permissions"]["packages"] == "write"
    assert workflow["permissions"] == {"contents": "read"}
    assert all(job["runs-on"] == "ubuntu-latest" for job in workflow["jobs"].values()), "somente runners gratuitos"
    publish_steps = [step for step in docker["steps"] if "docker push" in step.get("run", "")]
    assert publish_steps and all("publish" in step.get("if", "").lower() for step in publish_steps)
    triggers = workflow[True] if True in workflow else workflow["on"]
    assert "v*.*.*" in triggers["push"]["tags"]
    assert "pull_request" in triggers


def test_allowed_origins_accept_published_port_only():
    from backend.app import allowed_origins
    origins = allowed_origins("http://127.0.0.1:9000, http://localhost:9000/")
    assert {"http://127.0.0.1:9000", "http://localhost:9000", "http://127.0.0.1:8000"} <= origins
    assert "http://evil.example" not in origins
    assert allowed_origins("") == allowed_origins(" , ")


def test_health_reports_version():
    from backend.app import VERSION, health
    assert health()["version"] == VERSION


def test_gpu_check_does_not_claim_inference_without_execution(tmp_path):
    from scripts import gpu_check
    output = tmp_path / "report.json"
    code = gpu_check.main(["--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["inference_executed"] == []
    assert report["inference"] == {}
    assert report["summary"]["cuda_detected"] == report["environment"]["cuda_available"]
    if not report["environment"]["cuda_available"]:
        assert report["compute"] == {"executed": False, "reason": "CUDA indisponível"}
        assert gpu_check.main(["--require-cuda", "--output", str(output)]) == 2
    assert code == 0


def test_gpu_check_reports_missing_models_without_inference(tmp_path, monkeypatch):
    from scripts import gpu_check
    monkeypatch.setattr(gpu_check, "read_json", lambda *args, **kwargs: {})
    result = gpu_check.inference_check(["sd15", "smolvlm"], steps=5, size=256)
    for key in ("sd15", "smolvlm"):
        assert result[key] == {"executed": False, "reason": "modelo não instalado em models/"}


@repository_files
@pytest.mark.parametrize("name", ["compose.yaml", "compose.gpu.yaml", ".github/workflows/docker-publish.yml"])
def test_distribution_files_are_valid_yaml(name):
    assert yaml.safe_load(read(name))


def test_vision_processor_runtime_dependencies_installed():
    """SmolVLM2 exige num2words no processador; a ausência só aparecia na inferência real."""
    from transformers.utils import is_num2words_available, is_torchvision_available
    assert is_num2words_available(), "instale requirements.txt (num2words)"
    assert is_torchvision_available()
