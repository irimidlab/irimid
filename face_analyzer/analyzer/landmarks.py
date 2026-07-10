"""Detecção de landmarks faciais com MediaPipe Face Landmarker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy.typing import NDArray

from face_analyzer.constants import (
    CHIN,
    FOREHEAD,
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_PUPIL,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_PUPIL,
)
from face_analyzer.utils.geometry import compute_midline, euclidean_distance, midpoint

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "face_landmarker.task"


@dataclass
class FaceLandmarks:
    """Container para landmarks faciais detectados."""

    points: NDArray[np.float64]
    image_width: int
    image_height: int
    confidence: float = 1.0
    z_coords: Optional[NDArray[np.float64]] = None
    _midline_top: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _midline_bottom: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _face_scale: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("Landmarks devem ter formato (N, 2).")

    @property
    def count(self) -> int:
        """Número de landmarks detectados."""
        return self.points.shape[0]

    @property
    def face_scale(self) -> float:
        """Escala facial (altura testa-queixo) para normalização."""
        if self._face_scale is None:
            self._face_scale = euclidean_distance(
                self.points[FOREHEAD], self.points[CHIN]
            )
        return self._face_scale

    @property
    def midline(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Linha média facial (topo, base)."""
        if self._midline_top is None or self._midline_bottom is None:
            self._midline_top, self._midline_bottom = compute_midline(
                self.points, FOREHEAD, CHIN
            )
        return self._midline_top, self._midline_bottom

    def get_point(self, index: int) -> NDArray[np.float64]:
        """Retorna landmark por índice."""
        return self.points[index]

    @property
    def left_pupil(self) -> NDArray[np.float64]:
        """Centro pupilar esquerdo (ou aproximação)."""
        if self.count > LEFT_PUPIL:
            return self.points[LEFT_PUPIL]
        return midpoint(
            self.points[LEFT_EYE_INNER], self.points[LEFT_EYE_OUTER]
        )

    @property
    def right_pupil(self) -> NDArray[np.float64]:
        """Centro pupilar direito (ou aproximação)."""
        if self.count > RIGHT_PUPIL:
            return self.points[RIGHT_PUPIL]
        return midpoint(
            self.points[RIGHT_EYE_INNER], self.points[RIGHT_EYE_OUTER]
        )

    def to_normalized(self) -> NDArray[np.float64]:
        """Retorna landmarks normalizados (0-1) em relação à imagem."""
        normalized = self.points.copy()
        normalized[:, 0] /= self.image_width
        normalized[:, 1] /= self.image_height
        return normalized


class FaceLandmarkDetector:
    """Detector de landmarks faciais usando MediaPipe Face Landmarker."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {resolved_path}. "
                "Download face_landmarker.task to face_analyzer/assets/."
            )

        base_options = python.BaseOptions(model_asset_path=str(resolved_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: NDArray[np.uint8]) -> Optional[FaceLandmarks]:
        """
        Detecta landmarks na imagem BGR.

        Returns:
            FaceLandmarks ou None se nenhuma face for detectada.
        """
        height, width = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = self._landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]
        points = np.array(
            [[lm.x * width, lm.y * height] for lm in face_landmarks],
            dtype=np.float64,
        )
        z_coords = np.array([lm.z for lm in face_landmarks], dtype=np.float64)

        return FaceLandmarks(
            points=points,
            image_width=width,
            image_height=height,
            confidence=1.0,
            z_coords=z_coords,
        )

    def close(self) -> None:
        """Libera recursos do detector."""
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
