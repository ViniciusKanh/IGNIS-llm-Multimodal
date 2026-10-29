"""Contratos de receitas e exportações do Space sem simular inferência."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

SPACE = Path(__file__).resolve().parents[1] / "huggingface/space"


def settings_module():
    spec = importlib.util.spec_from_file_location("ignis_space_settings_test", SPACE / "settings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipe_roundtrip_preserves_effective_prompt_and_seed():
    settings = settings_module()
    options = settings.recipe("A mountain lake", "blur", 2026, False, next(iter(settings.SIZES)),
                              25, 7.5, .8, "IGNIS LoRA", settings.SAMPLERS[0], "Aquarela")
    restored, size = settings.restore_recipe(json.dumps(options).encode())
    assert restored["prompt"] == options["prompt"]
    assert restored["seed"] == 2026
    assert restored["lora_scale"] == .8
    assert size == next(iter(settings.SIZES))


def test_recipe_rejects_unsupported_parameters():
    settings = settings_module()
    with pytest.raises(ValueError):
        settings.recipe("A lake", "", 42, False, "99999x99999", 25, 7.5, 1, "IGNIS LoRA", settings.SAMPLERS[0], "Aquarela")
    with pytest.raises(ValueError):
        settings.restore_recipe(b'{"prompt":"A lake","width":99999,"height":99999}')


def test_publication_requires_existing_model_release(tmp_path):
    from scripts.publish_space import prepare_space
    with pytest.raises(RuntimeError, match="Pacote publicado"):
        prepare_space(tmp_path, tmp_path / "space")


@pytest.mark.parametrize("exists", [True, False])
def test_publication_reuses_existing_space_before_creating(monkeypatch, tmp_path, exists):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import huggingface_hub
    from huggingface_hub.utils import HfHubHTTPError
    from requests import Response
    from scripts.publish_space import publish

    api = Mock()
    api.whoami.return_value = {"name": "ViniciusKhan"}
    if not exists:
        response = Response()
        response.status_code = 404
        api.space_info.side_effect = HfHubHTTPError("Space inexistente", response=response)
    api.get_space_runtime.return_value = SimpleNamespace(
        hardware="zero-a10g", requested_hardware="zero-a10g", stage="BUILDING")
    api.upload_folder.return_value = SimpleNamespace(oid="commit-de-teste")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    publish(tmp_path, tmp_path, "ViniciusKhan/IGNIS-Image-Lab", {"model_revision": "revisao-de-teste"})

    assert api.create_repo.call_count == (0 if exists else 1)
    api.request_space_hardware.assert_not_called()
    api.upload_folder.assert_called_once()
