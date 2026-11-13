"""Diagnóstico reprodutível de CUDA e, opcionalmente, inferência real com pesos instalados.

Separa três níveis de evidência, sem simular resultados:
  1. detecção   — PyTorch enxerga uma GPU CUDA;
  2. computação — um kernel CUDA executa e confere o resultado numérico;
  3. inferência — um pipeline IGNIS real (SD 1.5 e/ou SmolVLM2) roda com os pesos instalados.

Uso (dentro do container ou no ambiente local):
  python scripts/gpu_check.py                       # detecção + computação
  python scripts/gpu_check.py --require-cuda        # falha (código 2) se não houver CUDA
  python scripts/gpu_check.py --inference sd15      # gera uma imagem pequena real
  python scripts/gpu_check.py --inference sd15 smolvlm
O relatório JSON é salvo em outputs/validation/.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.common import ROOT, read_json, write_json  # noqa: E402


def environment() -> dict:
    import torch
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "in_container": Path("/.dockerenv").exists() or os.environ.get("IGNIS_TORCH_VARIANT") is not None,
        "image_version": os.environ.get("IGNIS_VERSION"),
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
    }
    if info["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["devices"].append({"index": index, "name": props.name,
                                    "capability": f"{props.major}.{props.minor}",
                                    "total_vram_mb": round(props.total_memory / 2**20)})
        free, total = torch.cuda.mem_get_info()
        info["vram_free_mb"] = round(free / 2**20)
        info["vram_total_mb"] = round(total / 2**20)
    return info


def compute_check() -> dict:
    """Multiplicação FP16 na GPU comparada com referência FP32 na CPU."""
    import torch
    if not torch.cuda.is_available():
        return {"executed": False, "reason": "CUDA indisponível"}
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(512, 512, generator=generator)
    b = torch.randn(512, 512, generator=generator)
    start = time.perf_counter()
    result = (a.cuda().half() @ b.cuda().half()).float().cpu()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    error = (result - a @ b).abs().max().item()
    return {"executed": True, "ok": error < 1.0, "max_abs_error": round(error, 4),
            "seconds": round(elapsed, 4), "dtype": "float16"}


def inference_check(models: list[str], steps: int, size: int) -> dict:
    from backend.engine import engine
    manifest = read_json(ROOT / "models/manifest.json", {})
    results = {}
    generated = None
    for key in models:
        if manifest.get(key, {}).get("status") != "installed":
            results[key] = {"executed": False, "reason": "modelo não instalado em models/"}
            continue
        try:
            if key == "sd15":
                report = engine.generate("IGNIS GPU validation: a lighthouse on a rocky coast at sunset, oil painting",
                                         seed=2026, width=size, height=size, steps=steps)
                generated = ROOT / "outputs" / report["file"]
                results[key] = {"executed": True, "ok": generated.is_file(), "output": report["url"],
                                "seconds": report["seconds"], "steps": steps, "size": size,
                                "revision": manifest[key].get("revision")}
            else:
                from PIL import Image
                image = Image.open(generated).convert("RGB") if generated else Image.new("RGB", (256, 256), (200, 80, 40))
                start = time.perf_counter()
                answer = engine.vision("Describe this image in one sentence.", [image], key=key)
                text = answer["text"].strip()
                results[key] = {"executed": True, "ok": bool(text), "answer": text[:400],
                                "seconds": round(time.perf_counter() - start, 2),
                                "input": "imagem gerada no teste" if generated else "imagem sólida sintética",
                                "revision": manifest[key].get("revision")}
        except Exception as exc:  # o relatório registra a falha real
            results[key] = {"executed": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            engine.unload()
    try:
        import torch
        if torch.cuda.is_available():
            results["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    except Exception:
        pass
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--require-cuda", action="store_true", help="retorna código 2 sem GPU CUDA")
    parser.add_argument("--inference", nargs="*", choices=["sd15", "smolvlm", "qwen"], default=[],
                        help="executa inferência real com os modelos instalados")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--size", type=int, default=256, choices=[256, 384, 512])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = {"created_at": datetime.now(timezone.utc).isoformat(), "environment": environment()}
    report["compute"] = compute_check()
    report["inference"] = inference_check(args.inference, args.steps, args.size) if args.inference else {}
    report["summary"] = {
        "cuda_detected": report["environment"]["cuda_available"],
        "cuda_compute_ok": bool(report["compute"].get("ok")),
        "inference_executed": sorted(k for k, v in report["inference"].items() if isinstance(v, dict) and v.get("executed")),
        "inference_ok": sorted(k for k, v in report["inference"].items() if isinstance(v, dict) and v.get("ok")),
    }
    output = args.output or ROOT / "outputs/validation" / f"gpu-check-{datetime.now():%Y%m%d-%H%M%S}.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Relatório: {output}")
    if args.require_cuda and not report["summary"]["cuda_detected"]:
        return 2
    failed = [k for k, v in report["inference"].items() if isinstance(v, dict) and v.get("executed") and not v.get("ok")]
    if failed or (report["compute"].get("executed") and not report["compute"].get("ok")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
