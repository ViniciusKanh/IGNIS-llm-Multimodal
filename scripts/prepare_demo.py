"""Recorta a gravação autorizada em GIFs da aplicação para o README do GitHub."""
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image


def prepare(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("Não foi possível abrir a gravação.")
    segments = [("studio", 0, 23, 3, 1.5), ("comparacao", 26, 42, 4, 1),
                ("edicao-historico", 43, 54.5, 3, 1)]
    records = []
    try:
        for name, start, end, fps, speed in segments:
            frames = []
            for index in range(round((end - start) * fps)):
                capture.set(cv2.CAP_PROP_POS_MSEC, (start + index / fps) * 1000)
                ok, frame = capture.read()
                if not ok:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Enquadramento da aplicação, sem abas, favoritos e barra de endereço.
                image = image.crop((315, 88, 955, 685))
                frames.append(image.quantize(colors=128))
            if not frames:
                raise RuntimeError(f"Trecho vazio: {name}")
            path = destination / f"{name}.gif"
            frames[0].save(path, save_all=True, append_images=frames[1:],
                           duration=round(1000 / fps / speed), loop=0, optimize=True, disposal=2)
            records.append({"file": path.name, "source_start_seconds": start,
                            "source_end_seconds": end, "sampling_fps": fps,
                            "playback_speed": speed, "bytes": path.stat().st_size})
        capture.set(cv2.CAP_PROP_POS_MSEC, 38000)
        ok, frame = capture.read()
        if ok:
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop((315, 88, 955, 685)).save(destination / "comparacao.png")
        (destination / "manifest.json").write_text(json.dumps({"source": "Gravação do usuário: 2026-10-29 10-57-08.mp4",
            "source_duration_seconds": capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS),
            "crop_xyxy": [315, 88, 955, 685], "clips": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(records, indent=2))
    finally:
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/media"))
    args = parser.parse_args()
    prepare(args.source, args.output)
