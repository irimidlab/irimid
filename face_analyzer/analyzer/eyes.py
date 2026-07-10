"""Análise das características oculares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from face_analyzer.analyzer.landmarks import FaceLandmarks
from face_analyzer.constants import (
    LEFT_EYE_BOTTOM,
    LEFT_EYE_CONTOUR,
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_EYE_TOP,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_CONTOUR,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_TOP,
)
from face_analyzer.utils.geometry import (
    euclidean_distance,
    line_angle,
    mirror_point_across_line,
    polygon_area,
    symmetry_score,
)


@dataclass
class EyeMetrics:
    """Métricas dos olhos."""

    canthal_tilt_left: float = 0.0
    canthal_tilt_right: float = 0.0
    ocular_inclination: float = 0.0
    palpebral_opening_left: float = 0.0
    palpebral_opening_right: float = 0.0
    palpebral_opening_avg: float = 0.0
    interpupillary_distance: float = 0.0
    sclera_area_left: float = 0.0
    sclera_area_right: float = 0.0
    ocular_symmetry_index: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Converte métricas para dicionário."""
        return {
            "canthal_tilt_left": round(self.canthal_tilt_left, 2),
            "canthal_tilt_right": round(self.canthal_tilt_right, 2),
            "ocular_inclination": round(self.ocular_inclination, 2),
            "palpebral_opening_left": round(self.palpebral_opening_left, 4),
            "palpebral_opening_right": round(self.palpebral_opening_right, 4),
            "palpebral_opening": round(self.palpebral_opening_avg, 4),
            "interpupillary_distance": round(self.interpupillary_distance, 2),
            "sclera_area_left": round(self.sclera_area_left, 2),
            "sclera_area_right": round(self.sclera_area_right, 2),
            "ocular_symmetry": round(self.ocular_symmetry_index, 2),
        }


class EyeAnalyzer:
    """Analisa características geométricas dos olhos."""

    def analyze(self, landmarks: FaceLandmarks) -> EyeMetrics:
        """Calcula métricas oculares."""
        points = landmarks.points
        scale = landmarks.face_scale
        mid_top, mid_bottom = landmarks.midline

        canthal_left = self._canthal_tilt(
            points[LEFT_EYE_INNER], points[LEFT_EYE_OUTER]
        )
        canthal_right = self._canthal_tilt(
            points[RIGHT_EYE_INNER], points[RIGHT_EYE_OUTER]
        )

        left_center = (points[LEFT_EYE_TOP] + points[LEFT_EYE_BOTTOM]) / 2
        right_center = (points[RIGHT_EYE_TOP] + points[RIGHT_EYE_BOTTOM]) / 2
        ocular_inclination = line_angle(left_center, right_center)

        palpebral_left = euclidean_distance(
            points[LEFT_EYE_TOP], points[LEFT_EYE_BOTTOM]
        ) / scale
        palpebral_right = euclidean_distance(
            points[RIGHT_EYE_TOP], points[RIGHT_EYE_BOTTOM]
        ) / scale

        interpupillary = euclidean_distance(
            landmarks.left_pupil, landmarks.right_pupil
        )

        sclera_left = self._eye_contour_area(points, LEFT_EYE_CONTOUR)
        sclera_right = self._eye_contour_area(points, RIGHT_EYE_CONTOUR)

        left_eye_center = (points[LEFT_EYE_INNER] + points[LEFT_EYE_OUTER]) / 2
        right_mirrored = mirror_point_across_line(
            (points[RIGHT_EYE_INNER] + points[RIGHT_EYE_OUTER]) / 2,
            mid_top,
            mid_bottom,
        )
        ocular_symmetry = symmetry_score(left_eye_center, right_mirrored, scale)

        return EyeMetrics(
            canthal_tilt_left=canthal_left,
            canthal_tilt_right=canthal_right,
            ocular_inclination=ocular_inclination,
            palpebral_opening_left=palpebral_left,
            palpebral_opening_right=palpebral_right,
            palpebral_opening_avg=(palpebral_left + palpebral_right) / 2,
            interpupillary_distance=interpupillary,
            sclera_area_left=sclera_left,
            sclera_area_right=sclera_right,
            ocular_symmetry_index=ocular_symmetry,
        )

    def _canthal_tilt(
        self,
        inner: np.ndarray,
        outer: np.ndarray,
    ) -> float:
        """Calcula canthal tilt (ângulo canto medial-lateral)."""
        angle = line_angle(inner, outer)
        return -angle

    def _eye_contour_area(
        self,
        points: np.ndarray,
        contour_indices: list[int],
    ) -> float:
        """Estima área aparente da região ocular."""
        contour_points = [points[i] for i in contour_indices if i < len(points)]
        return polygon_area(contour_points)
