"""Análise da mandíbula."""

from __future__ import annotations

from dataclasses import dataclass

from face_analyzer.analyzer.landmarks import FaceLandmarks
from face_analyzer.constants import CHIN, LEFT_CHEEK, LEFT_JAW, RIGHT_CHEEK, RIGHT_JAW
from face_analyzer.utils.geometry import angle_between_points, euclidean_distance


@dataclass
class JawMetrics:
    """Métricas da mandíbula."""

    width: float = 0.0
    width_ratio: float = 0.0
    jaw_angle_left: float = 0.0
    jaw_angle_right: float = 0.0
    jaw_angle_avg: float = 0.0
    jaw_zygoma_ratio: float = 0.0
    definition_index: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Converte métricas para dicionário."""
        return {
            "jaw_width": round(self.width, 2),
            "jaw_width_ratio": round(self.width_ratio, 4),
            "jaw_angle_left": round(self.jaw_angle_left, 2),
            "jaw_angle_right": round(self.jaw_angle_right, 2),
            "jaw_angle": round(self.jaw_angle_avg, 2),
            "jaw_zygoma_ratio": round(self.jaw_zygoma_ratio, 4),
            "jaw_definition": round(self.definition_index, 2),
        }


class JawAnalyzer:
    """Analisa características geométricas da mandíbula."""

    def analyze(self, landmarks: FaceLandmarks) -> JawMetrics:
        """Calcula métricas mandibulares."""
        points = landmarks.points
        scale = landmarks.face_scale

        jaw_width = euclidean_distance(points[LEFT_JAW], points[RIGHT_JAW])
        zygoma_width = euclidean_distance(points[LEFT_CHEEK], points[RIGHT_CHEEK])

        angle_left = angle_between_points(
            points[LEFT_CHEEK], points[LEFT_JAW], points[CHIN]
        )
        angle_right = angle_between_points(
            points[RIGHT_CHEEK], points[RIGHT_JAW], points[CHIN]
        )

        jaw_height = (
            euclidean_distance(points[LEFT_JAW], points[CHIN])
            + euclidean_distance(points[RIGHT_JAW], points[CHIN])
        ) / 2
        wh_ratio = jaw_width / (jaw_height + 1e-10)
        definition = float(np.clip((wh_ratio - 0.85) / 0.55 * 100, 0, 100))

        return JawMetrics(
            width=jaw_width,
            width_ratio=jaw_width / scale,
            jaw_angle_left=angle_left,
            jaw_angle_right=angle_right,
            jaw_angle_avg=(angle_left + angle_right) / 2,
            jaw_zygoma_ratio=jaw_width / zygoma_width if zygoma_width > 0 else 0,
            definition_index=definition,
        )
