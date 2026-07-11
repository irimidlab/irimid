"""Análise das características labiais."""

from __future__ import annotations

from dataclasses import dataclass

from face_analyzer.analyzer.landmarks import FaceLandmarks
from face_analyzer.constants import (
    LEFT_MOUTH,
    LOWER_LIP_BOTTOM,
    LOWER_LIP_CENTER,
    RIGHT_MOUTH,
    UPPER_LIP_CENTER,
    UPPER_LIP_TOP,
)
from face_analyzer.utils.geometry import (
    euclidean_distance,
    mirror_point_across_line,
    perpendicular_distance_to_line,
    symmetry_score,
)


@dataclass
class LipMetrics:
    """Métricas dos lábios."""

    width: float = 0.0
    width_ratio: float = 0.0
    symmetry_index: float = 0.0
    upper_lip_height: float = 0.0
    lower_lip_height: float = 0.0
    upper_lower_ratio: float = 0.0
    alignment_deviation: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Converte métricas para dicionário."""
        return {
            "mouth_width": round(self.width, 2),
            "mouth_width_ratio": round(self.width_ratio, 4),
            "mouth_symmetry": round(self.symmetry_index, 2),
            "upper_lip_height": round(self.upper_lip_height, 2),
            "lower_lip_height": round(self.lower_lip_height, 2),
            "upper_lower_lip_ratio": round(self.upper_lower_ratio, 4),
            "mouth_alignment": round(self.alignment_deviation, 4),
        }


class LipAnalyzer:
    """Analisa características geométricas dos lábios."""

    def analyze(self, landmarks: FaceLandmarks) -> LipMetrics:
        """Calcula métricas labiais."""
        points = landmarks.points
        scale = landmarks.face_scale
        mid_top, mid_bottom = landmarks.midline

        width = euclidean_distance(points[LEFT_MOUTH], points[RIGHT_MOUTH])
        upper_height = euclidean_distance(
            points[UPPER_LIP_TOP], points[UPPER_LIP_CENTER]
        )
        lower_height = euclidean_distance(
            points[LOWER_LIP_CENTER], points[LOWER_LIP_BOTTOM]
        )

        left_corner = points[LEFT_MOUTH]
        right_mirrored = mirror_point_across_line(
            points[RIGHT_MOUTH], mid_top, mid_bottom
        )
        symmetry = symmetry_score(left_corner, right_mirrored, scale)

        mouth_center = (points[UPPER_LIP_CENTER] + points[LOWER_LIP_CENTER]) / 2
        alignment = perpendicular_distance_to_line(
            mouth_center, mid_top, mid_bottom
        ) / scale

        return LipMetrics(
            width=width,
            width_ratio=width / scale,
            symmetry_index=symmetry,
            upper_lip_height=upper_height,
            lower_lip_height=lower_height,
            upper_lower_ratio=upper_height / lower_height if lower_height > 0 else 0,
            alignment_deviation=alignment,
        )
