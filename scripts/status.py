"""Exibe recursos reais e manifesto, sem carregar pesos."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.engine import engine

if __name__ == '__main__':
    print(json.dumps(engine.status(), indent=2, ensure_ascii=False))
