"""Análise de proporções faciais."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from face_analyzer.analyzer.landmarks import FaceLandmarks
from face_analyzer.constants import (
    ANTHROPOMETRIC_REFERENCES,
    CHIN,
    FOREHEAD,
    LEFT_CHEEK,
    LEFT_EYE_TOP,
    LEFT_JAW,
    LEFT_MOUTH,
    NOSE_LEFT,
    NOSE_RIGHT,
    NOSE_TIP,
    RIGHT_CHEEK,
    RIGHT_EYE_TOP,
    RIGHT_JAW,
    RIGHT_MOUTH,
)
from face_analyzer.utils.geometry import euclidean_distance, score_against_reference


@dataclass
class ProportionMetrics:
    """Métricas de proporção facial."""

    upper_third_ratio: float = 0.0
    middle_third_ratio: float = 0.0
    lower_third_ratio: float = 0.0
    face_width: float = 0.0
    face_height: float = 0.0
    face_width_height_ratio: float = 0.0
    nose_width: float = 0.0
    nose_width_ratio: float = 0.0
    mouth_width: float = 0.0
    mouth_width_ratio: float = 0.0
    interpupillary_distance: float = 0.0
    interpupillary_ratio: float = 0.0
    eye_spacing: float = 0.0
    eye_spacing_ratio: float = 0.0
    zygoma_width: float = 0.0
    jaw_width: float = 0.0
    jaw_zygoma_ratio: float = 0.0
    chin_height: float = 0.0
    reference_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Converte métricas para dicionário."""
        return {
            "upper_third": round(self.upper_third_ratio, 4),
            "middle_third": round(self.middle_third_ratio, 4),
            "lower_third": round(self.lower_third_ratio, 4),
            "face_width": round(self.face_width, 2),
            "face_height": round(self.face_height, 2),
            "face_width_height": round(self.face_width_height_ratio, 4),
            "nose_width": round(self.nose_width, 2),
            "nose_width_face": round(self.nose_width_ratio, 4),
            "mouth_width": round(self.mouth_width, 2),
            "mouth_width_face": round(self.mouth_width_ratio, 4),
            "interpupillary_distance": round(self.interpupillary_distance, 2),
            "interpupillary_face": round(self.interpupillary_ratio, 4),
            "eye_spacing": round(self.eye_spacing, 2),
            "eye_spacing_ratio": round(self.eye_spacing_ratio, 4),
            "zygoma_width": round(self.zygoma_width, 2),
            "jaw_width": round(self.jaw_width, 2),
            "jaw_zygoma": round(self.jaw_zygoma_ratio, 4),
            "chin_height": round(self.chin_height, 2),
            "reference_scores": {k: round(v, 2) for k, v in self.reference_scores.items()},
        }


class ProportionAnalyzer:
    """Calcula proporções faciais e compara com referências."""

    def analyze(self, landmarks: FaceLandmarks) -> ProportionMetrics:
        """Calcula todas as proporções faciais."""
        points = landmarks.points

        forehead_y = points[FOREHEAD][1]
        brow_y = (points[LEFT_EYE_TOP][1] + points[RIGHT_EYE_TOP][1]) / 2
        nose_y = points[NOSE_TIP][1]
        chin_y = points[CHIN][1]

        face_height = chin_y - forehead_y
        upper_third = (brow_y - forehead_y) / face_height if face_height > 0 else 0
        middle_third = (nose_y - brow_y) / face_height if face_height > 0 else 0
        lower_third = (chin_y - nose_y) / face_height if face_height > 0 else 0

        zygoma_width = euclidean_distance(points[LEFT_CHEEK], points[RIGHT_CHEEK])
        jaw_width = euclidean_distance(points[LEFT_JAW], points[RIGHT_JAW])
        nose_width = euclidean_distance(points[NOSE_LEFT], points[NOSE_RIGHT])
        mouth_width = euclidean_distance(points[LEFT_MOUTH], points[RIGHT_MOUTH])
        interpupillary = euclidean_distance(landmarks.left_pupil, landmarks.right_pupil)
        eye_spacing = euclidean_distance(points[133], points[362])
        chin_height = chin_y - points[14][1]

        face_width = max(zygoma_width, jaw_width)
        wh_ratio = face_width / face_height if face_height > 0 else 0

        metrics = ProportionMetrics(
            upper_third_ratio=upper_third,
            middle_third_ratio=middle_third,
            lower_third_ratio=lower_third,
            face_width=face_width,
            face_height=face_height,
            face_width_height_ratio=wh_ratio,
            nose_width=nose_width,
            nose_width_ratio=nose_width / face_width if face_width > 0 else 0,
            mouth_width=mouth_width,
            mouth_width_ratio=mouth_width / face_width if face_width > 0 else 0,
            interpupillary_distance=interpupillary,
            interpupillary_ratio=interpupillary / face_width if face_width > 0 else 0,
            eye_spacing=eye_spacing,
            eye_spacing_ratio=eye_spacing / face_width if face_width > 0 else 0,
            zygoma_width=zygoma_width,
            jaw_width=jaw_width,
            jaw_zygoma_ratio=jaw_width / zygoma_width if zygoma_width > 0 else 0,
            chin_height=chin_height,
        )

        metrics.reference_scores = self._compute_reference_scores(metrics)
        return metrics

    def _compute_reference_scores(self, metrics: ProportionMetrics) -> dict[str, float]:
        """Compara proporções com faixas antropométricas."""
        scores: dict[str, float] = {}
        mapping = {
            "upper_third_ratio": metrics.upper_third_ratio,
            "middle_third_ratio": metrics.middle_third_ratio,
            "lower_third_ratio": metrics.lower_third_ratio,
            "face_width_height": metrics.face_width_height_ratio,
            "nose_width_face": metrics.nose_width_ratio,
            "mouth_width_face": metrics.mouth_width_ratio,
            "interpupillary_face": metrics.interpupillary_ratio,
            "eye_spacing": metrics.eye_spacing_ratio,
            "jaw_zygoma": metrics.jaw_zygoma_ratio,
        }
        for key, value in mapping.items():
            if key in ANTHROPOMETRIC_REFERENCES:
                ref = ANTHROPOMETRIC_REFERENCES[key]
                scores[key] = score_against_reference(
                    value, ref["min"], ref["ideal"], ref["max"]
                )
        return scores
