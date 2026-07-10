"""Utilitários de processamento de imagem."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from numpy.typing import NDArray


def load_image(source: str | Path | bytes | BinaryIO) -> NDArray[np.uint8]:
    """Carrega imagem de arquivo, bytes ou buffer e retorna array BGR."""
    if isinstance(source, (str, Path)):
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Could not load image: {source}")
        return image

    if isinstance(source, bytes):
        buffer = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image bytes.")
        return image

    data = source.read()
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image buffer.")
    return image


def rgb_to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Converte imagem RGB para BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Converte imagem BGR para RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_image_dimensions(image: NDArray[np.uint8]) -> tuple[int, int]:
    """Retorna (largura, altura) da imagem."""
    height, width = image.shape[:2]
    return width, height


def assess_brightness(image: NDArray[np.uint8]) -> float:
    """Avalia brilho médio da imagem (0-255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def assess_sharpness(image: NDArray[np.uint8]) -> float:
    """Avalia nitidez via variância do Laplaciano."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_for_display(
    image: NDArray[np.uint8],
    max_dimension: int = 1200,
) -> NDArray[np.uint8]:
    """Redimensiona imagem mantendo proporção para exibição."""
    height, width = image.shape[:2]
    if max(height, width) <= max_dimension:
        return image.copy()
    scale = max_dimension / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def encode_image_png(image: NDArray[np.uint8]) -> bytes:
    """Codifica imagem BGR como PNG."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode PNG image.")
    return buffer.tobytes()


def save_image(image: NDArray[np.uint8], path: str | Path) -> None:
    """Salva imagem BGR em disco."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Failed to save image to: {path}")
