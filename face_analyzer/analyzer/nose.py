"""Análise das características do nariz."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from face_analyzer.analyzer.landmarks import FaceLandmarks
from face_analyzer.constants import NOSE_BRIDGE, NOSE_LEFT, NOSE_RIGHT, NOSE_TIP
from face_analyzer.utils.geometry import (
    euclidean_distance,
    mirror_point_across_line,
    perpendicular_distance_to_line,
    symmetry_score,
)


@dataclass
class NoseMetrics:
    """Métricas do nariz."""

    length: float = 0.0
    width: float = 0.0
    length_ratio: float = 0.0
    width_ratio: float = 0.0
    symmetry_index: float = 0.0
    alignment_deviation: float = 0.0
    projection_estimate: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Converte métricas para dicionário."""
        return {
            "nose_length": round(self.length, 2),
            "nose_width": round(self.width, 2),
            "nose_length_ratio": round(self.length_ratio, 4),
            "nose_width_ratio": round(self.width_ratio, 4),
            "nose_symmetry": round(self.symmetry_index, 2),
            "nose_alignment": round(self.alignment_deviation, 4),
            "nose_projection": round(self.projection_estimate, 4),
        }


class NoseAnalyzer:
    """Analisa características geométricas do nariz."""

    def analyze(self, landmarks: FaceLandmarks) -> NoseMetrics:
        """Calcula métricas do nariz."""
        points = landmarks.points
        scale = landmarks.face_scale
        mid_top, mid_bottom = landmarks.midline

        length = euclidean_distance(points[NOSE_BRIDGE], points[NOSE_TIP])
        width = euclidean_distance(points[NOSE_LEFT], points[NOSE_RIGHT])

        left_nostril = points[NOSE_LEFT]
        right_nostril = points[NOSE_RIGHT]
        mirrored_right = mirror_point_across_line(right_nostril, mid_top, mid_bottom)
        symmetry = symmetry_score(left_nostril, mirrored_right, scale)

        nose_tip = points[NOSE_TIP]
        alignment = perpendicular_distance_to_line(nose_tip, mid_top, mid_bottom) / scale

        bridge = points[NOSE_BRIDGE]
        projection = (nose_tip[1] - bridge[1]) / scale

        return NoseMetrics(
            length=length,
            width=width,
            length_ratio=length / scale,
            width_ratio=width / scale,
            symmetry_index=symmetry,
            alignment_deviation=alignment,
            projection_estimate=projection,
        )
