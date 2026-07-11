"""Análise de simetria facial."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from face_analyzer.analyzer.landmarks import FaceLandmarks
from face_analyzer.constants import SYMMETRY_PAIRS
from face_analyzer.utils.geometry import (
    mirror_point_across_line,
    perpendicular_distance_to_line,
    symmetry_score,
)


@dataclass
class SymmetryMetrics:
    """Métricas de simetria facial."""

    overall_index: float = 0.0
    eyes_index: float = 0.0
    nose_index: float = 0.0
    mouth_index: float = 0.0
    jaw_index: float = 0.0
    midline_deviation_degrees: float = 0.0
    midline_deviation_pixels: float = 0.0
    left_right_differences: dict[str, float] = field(default_factory=dict)
    pair_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Converte métricas para dicionário."""
        return {
            "symmetry_overall": round(self.overall_index, 2),
            "symmetry_eyes": round(self.eyes_index, 2),
            "symmetry_nose": round(self.nose_index, 2),
            "symmetry_mouth": round(self.mouth_index, 2),
            "symmetry_jaw": round(self.jaw_index, 2),
            "midline_deviation_degrees": round(self.midline_deviation_degrees, 2),
            "midline_deviation_pixels": round(self.midline_deviation_pixels, 2),
            "left_right_differences": {
                k: round(v, 4) for k, v in self.left_right_differences.items()
            },
        }


class SymmetryAnalyzer:
    """Calcula índices de simetria facial."""

    EYE_PAIRS: list[tuple[int, int]] = [
        (33, 263), (133, 362), (145, 374), (159, 386),
    ]
    NOSE_PAIRS: list[tuple[int, int]] = [(129, 358), (219, 439)]
    MOUTH_PAIRS: list[tuple[int, int]] = [(61, 291), (78, 308), (13, 14)]
    JAW_PAIRS: list[tuple[int, int]] = [
        (172, 397), (58, 288), (132, 361), (93, 323),
    ]

    def analyze(self, landmarks: FaceLandmarks) -> SymmetryMetrics:
        """Calcula todas as métricas de simetria."""
        mid_top, mid_bottom = landmarks.midline
        scale = landmarks.face_scale

        pair_scores: list[float] = []
        differences: dict[str, float] = {}

        for left_idx, right_idx in SYMMETRY_PAIRS:
            if left_idx == right_idx:
                continue
            left_pt = landmarks.points[left_idx]
            right_pt = landmarks.points[right_idx]
            mirrored = mirror_point_across_line(right_pt, mid_top, mid_bottom)
            score = symmetry_score(left_pt, mirrored, scale)
            pair_scores.append(score)
            differences[f"{left_idx}_{right_idx}"] = float(
                np.linalg.norm(left_pt - mirrored) / scale
            )

        eyes_scores = self._region_scores(landmarks, self.EYE_PAIRS, mid_top, mid_bottom, scale)
        nose_scores = self._region_scores(landmarks, self.NOSE_PAIRS, mid_top, mid_bottom, scale)
        mouth_scores = self._region_scores(landmarks, self.MOUTH_PAIRS, mid_top, mid_bottom, scale)
        jaw_scores = self._region_scores(landmarks, self.JAW_PAIRS, mid_top, mid_bottom, scale)

        nose_tip = landmarks.points[1]
        midline_dev_px = perpendicular_distance_to_line(nose_tip, mid_top, mid_bottom)
        midline_dev_deg = (midline_dev_px / scale) * 10.0

        return SymmetryMetrics(
            overall_index=float(np.mean(pair_scores)) if pair_scores else 0.0,
            eyes_index=float(np.mean(eyes_scores)) if eyes_scores else 0.0,
            nose_index=float(np.mean(nose_scores)) if nose_scores else 0.0,
            mouth_index=float(np.mean(mouth_scores)) if mouth_scores else 0.0,
            jaw_index=float(np.mean(jaw_scores)) if jaw_scores else 0.0,
            midline_deviation_degrees=midline_dev_deg,
            midline_deviation_pixels=midline_dev_px,
            left_right_differences=differences,
            pair_scores=pair_scores,
        )

    def _region_scores(
        self,
        landmarks: FaceLandmarks,
        pairs: list[tuple[int, int]],
        mid_top: NDArray[np.float64],
        mid_bottom: NDArray[np.float64],
        scale: float,
    ) -> list[float]:
        """Calcula scores de simetria para uma região."""
        scores: list[float] = []
        for left_idx, right_idx in pairs:
            left_pt = landmarks.points[left_idx]
            right_pt = landmarks.points[right_idx]
            mirrored = mirror_point_across_line(right_pt, mid_top, mid_bottom)
            scores.append(symmetry_score(left_pt, mirrored, scale))
        return scores
