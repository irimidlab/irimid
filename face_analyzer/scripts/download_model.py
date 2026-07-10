"""Script de download do modelo MediaPipe Face Landmarker."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "assets" / "face_landmarker.task"


def download_model(output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Baixa o modelo face_landmarker.task se ainda não existir."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Model already exists: {output_path}")
        return output_path

    print(f"Downloading model from {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, output_path)
    print(f"Model saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    path = download_model()
    if not path.exists():
        print("Failed to download model.", file=sys.stderr)
        sys.exit(1)
