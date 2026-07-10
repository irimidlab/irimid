"""Análise de perfil facial (imagem lateral opcional)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_analyzer.analyzer.landmarks import FaceLandmarkDetector, FaceLandmarks
from face_analyzer.constants import CHIN, FOREHEAD, NOSE_TIP
from face_analyzer.utils.geometry import euclidean_distance, line_angle


@dataclass
class ProfileMetrics:
    """Métricas de perfil facial."""

    available: bool = False
    facial_convexity: float = 0.0
    nose_projection: float = 0.0
    chin_projection: float = 0.0
    forehead_projection: float = 0.0
    profile_angle: float = 0.0
    message: str = "No profile image provided."

    def to_dict(self) -> dict[str, object]:
        """Converte métricas para dicionário."""
        return {
            "profile_available": self.available,
            "facial_convexity": round(self.facial_convexity, 4),
            "profile_nose_projection": round(self.nose_projection, 4),
            "profile_chin_projection": round(self.chin_projection, 4),
            "profile_forehead_projection": round(self.forehead_projection, 4),
            "profile_angle": round(self.profile_angle, 2),
            "profile_message": self.message,
        }


class ProfileAnalyzer:
    """Analisa imagem de perfil quando disponível."""

    def __init__(self, detector: FaceLandmarkDetector | None = None) -> None:
        self._detector = detector

    def analyze(
        self,
        profile_image: NDArray[np.uint8] | None,
        frontal_landmarks: FaceLandmarks | None = None,
    ) -> ProfileMetrics:
        """
        Analisa imagem lateral.

        Quando não há imagem de perfil, retorna métricas vazias com mensagem.
        """
        if profile_image is None:
            return ProfileMetrics()

        detector = self._detector or FaceLandmarkDetector()
        own_detector = self._detector is None

        try:
            landmarks = detector.detect(profile_image)
            if landmarks is None:
                return ProfileMetrics(
                    available=False,
                    message="No face detected in the profile image.",
                )

            return self._compute_profile_metrics(landmarks)
        finally:
            if own_detector:
                detector.close()

    def _compute_profile_metrics(self, landmarks: FaceLandmarks) -> ProfileMetrics:
        """Calcula métricas a partir de landmarks de perfil."""
        points = landmarks.points
        scale = landmarks.face_scale

        forehead = points[FOREHEAD]
        nose = points[NOSE_TIP]
        chin = points[CHIN]

        nose_proj = (nose[0] - forehead[0]) / scale
        chin_proj = (chin[0] - forehead[0]) / scale
        forehead_proj = 0.0

        convexity = euclidean_distance(forehead, chin)
        if convexity > 0:
            nose_deviation = abs(nose[0] - (forehead[0] + chin[0]) / 2) / convexity
        else:
            nose_deviation = 0.0

        profile_angle = line_angle(forehead, chin)

        return ProfileMetrics(
            available=True,
            facial_convexity=nose_deviation,
            nose_projection=nose_proj,
            chin_projection=chin_proj,
            forehead_projection=forehead_proj,
            profile_angle=profile_angle,
            message="Profile analysis complete.",
        )
