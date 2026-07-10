"""Análise do queixo."""

from __future__ import annotations

from dataclasses import dataclass

from face_analyzer.analyzer.landmarks import FaceLandmarks
from face_analyzer.constants import CHIN, LEFT_JAW, LOWER_LIP_BOTTOM, NOSE_TIP, RIGHT_JAW
from face_analyzer.utils.geometry import euclidean_distance, perpendicular_distance_to_line


@dataclass
class ChinMetrics:
    """Métricas do queixo."""

    height: float = 0.0
    height_ratio: float = 0.0
    width: float = 0.0
    width_ratio: float = 0.0
    alignment_deviation: float = 0.0
    projection_estimate: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Converte métricas para dicionário."""
        return {
            "chin_height": round(self.height, 2),
            "chin_height_ratio": round(self.height_ratio, 4),
            "chin_width": round(self.width, 2),
            "chin_width_ratio": round(self.width_ratio, 4),
            "chin_alignment": round(self.alignment_deviation, 4),
            "chin_projection": round(self.projection_estimate, 4),
        }


class ChinAnalyzer:
    """Analisa características geométricas do queixo."""

    def analyze(self, landmarks: FaceLandmarks) -> ChinMetrics:
        """Calcula métricas do queixo."""
        points = landmarks.points
        scale = landmarks.face_scale
        mid_top, mid_bottom = landmarks.midline

        chin_point = points[CHIN]
        lip_bottom = points[LOWER_LIP_BOTTOM]
        height = euclidean_distance(lip_bottom, chin_point)
        width = euclidean_distance(points[LEFT_JAW], points[RIGHT_JAW]) * 0.6

        alignment = perpendicular_distance_to_line(chin_point, mid_top, mid_bottom) / scale
        nose_tip = points[NOSE_TIP]
        projection = (chin_point[1] - nose_tip[1]) / scale

        return ChinMetrics(
            height=height,
            height_ratio=height / scale,
            width=width,
            width_ratio=width / scale,
            alignment_deviation=alignment,
            projection_estimate=projection,
        )
