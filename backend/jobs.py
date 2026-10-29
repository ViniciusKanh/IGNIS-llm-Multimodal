"""Fila persistente de comandos conhecidos, sem execução arbitrária do cliente."""
from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from backend.common import ROOT, read_json, write_json


class Jobs:
    def __init__(self):
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ignis-job")
        self.records = read_json(ROOT / "logs/jobs.json", {})
        for item in self.records.values():
            if item["status"] in ("queued", "running"):
                item.update(status="interrupted", error="Backend reiniciado; consulte o log e retome o comando.")

    def save(self):
        write_json(ROOT / "logs/jobs.json", self.records)

    def submit(self, action, model=None):
        training_name = read_json(ROOT / "configs/train.json", {}).get("name", "ignis-image-lora-v1")
        commands = {
            "prepare": ["scripts/prepare_dataset.py"],
            "train": ["scripts/train_lora.py"],
            "evaluate": ["scripts/evaluate.py", "--adapter", training_name + "/final"],
            "download": ["scripts/download_models.py", "--models", model or "all"],
        }
        if action not in commands:
            raise ValueError("Operação desconhecida.")
        with self.lock:
            if any(item["status"] in ("queued", "running") for item in self.records.values()):
                raise ValueError("Já há uma operação em execução; aguarde sua conclusão.")
            identifier = uuid.uuid4().hex
            record = {"id": identifier, "action": action, "model": model, "status": "queued",
                      "created_at": datetime.now(timezone.utc).isoformat()}
            self.records[identifier] = record
            self.save()
            self.pool.submit(self.run, identifier, commands[action])
            return record.copy()

    def run(self, identifier, command):
        with self.lock:
            self.records[identifier]["status"] = "running"
            self.save()
        try:
            with (ROOT / "logs" / f"{identifier}.log").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                code = process.wait()
            with self.lock:
                self.records[identifier].update(status="completed" if code == 0 else "failed", exit_code=code,
                                                 finished_at=datetime.now(timezone.utc).isoformat())
                self.save()
        except Exception as exc:
            with self.lock:
                self.records[identifier].update(status="failed", error=str(exc))
                self.save()

    def list(self):
        with self.lock:
            return [dict(record) for record in reversed(list(self.records.values()))]

    def log(self, identifier):
        if identifier not in self.records:
            raise FileNotFoundError("Operação inexistente.")
        path = ROOT / "logs" / f"{identifier}.log"
        if not path.exists():
            return "Aguardando início."
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 32000))
            return stream.read().decode("utf-8", errors="replace")


jobs = Jobs()
