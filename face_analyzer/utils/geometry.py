"""Funções geométricas para análise facial."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

# Typical relative landmark mirror error (~2-4%) for frontal faces should score ~90+
SYMMETRY_DECAY_RATE: float = 100.0


def euclidean_distance(p1: NDArray[np.float64], p2: NDArray[np.float64]) -> float:
    """Calcula distância euclidiana entre dois pontos 2D."""
    return float(np.linalg.norm(p1 - p2))


def midpoint(p1: NDArray[np.float64], p2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Retorna o ponto médio entre dois pontos."""
    return (p1 + p2) / 2.0


def angle_between_points(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
) -> float:
    """Calcula ângulo em graus no vértice p2 formado por p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def line_angle(p1: NDArray[np.float64], p2: NDArray[np.float64]) -> float:
    """Retorna ângulo da linha em graus em relação ao eixo horizontal."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(math.degrees(math.atan2(dy, dx)))


def perpendicular_distance_to_line(
    point: NDArray[np.float64],
    line_p1: NDArray[np.float64],
    line_p2: NDArray[np.float64],
) -> float:
    """Distância perpendicular de um ponto a uma reta definida por dois pontos."""
    if np.allclose(line_p1, line_p2):
        return euclidean_distance(point, line_p1)
    return float(
        abs(
            (line_p2[0] - line_p1[0]) * (line_p1[1] - point[1])
            - (line_p1[0] - point[0]) * (line_p2[1] - line_p1[1])
        )
        / euclidean_distance(line_p1, line_p2)
    )


def mirror_point_across_line(
    point: NDArray[np.float64],
    line_p1: NDArray[np.float64],
    line_p2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Espelha um ponto em relação a uma reta."""
    direction = line_p2 - line_p1
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    diff = point - line_p1
    projection = np.dot(diff, direction) * direction
    return 2 * (line_p1 + projection) - point


def polygon_area(points: Sequence[NDArray[np.float64]]) -> float:
    """Calcula área de um polígono usando fórmula do cadarço (shoelace)."""
    if len(points) < 3:
        return 0.0
    coords = np.array(points)
    x = coords[:, 0]
    y = coords[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normaliza valor para faixa 0-100."""
    if max_val <= min_val:
        return 50.0
    normalized = (value - min_val) / (max_val - min_val) * 100.0
    return float(np.clip(normalized, 0.0, 100.0))


def score_against_reference(
    value: float,
    ref_min: float,
    ref_ideal: float,
    ref_max: float,
) -> float:
    """Score value (0-100) against anthropometric reference with soft falloff."""
    half_range = (ref_max - ref_min) / 2.0 + 1e-10
    deviation = abs(value - ref_ideal) / half_range

    if ref_min <= value <= ref_max:
        return float(np.clip(100.0 - deviation * 20.0, 88.0, 100.0))

    distance = min(abs(value - ref_min), abs(value - ref_max))
    range_size = ref_max - ref_min + 1e-10
    penalty = (distance / range_size) * 25.0
    return float(np.clip(88.0 - penalty, 55.0, 87.0))


def symmetry_score(
    left_point: NDArray[np.float64],
    right_mirrored: NDArray[np.float64],
    face_scale: float,
) -> float:
    """
    Symmetry score (0-100) calibrated for MediaPipe landmark precision.

    Uses a soft decay: ~3% relative error ≈ 90, ~6% ≈ 82, ~10% ≈ 75.
    """
    distance = euclidean_distance(left_point, right_mirrored)
    relative_error = distance / (face_scale + 1e-10)
    return float(np.clip(100.0 - relative_error * SYMMETRY_DECAY_RATE, 0.0, 100.0))


def compute_midline(
    landmarks: NDArray[np.float64],
    top_idx: int,
    bottom_idx: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Legacy midline — delegates to optimized facial midline."""
    return compute_facial_midline(landmarks, top_idx, bottom_idx)


def compute_facial_midline(
    landmarks: NDArray[np.float64],
    forehead_idx: int = 10,
    chin_idx: int = 152,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build a vertical midline through the facial sagittal plane.

    Uses the median x of central landmarks and bilateral midpoints for
    robustness against head tilt and landmark noise.
    """
    central_indices = [1, 2, 6, 13, 14, chin_idx]
    x_samples = [landmarks[i][0] for i in central_indices if i < len(landmarks)]

    bilateral_pairs = [(133, 362), (33, 263), (61, 291), (129, 358), (172, 397)]
    for left_idx, right_idx in bilateral_pairs:
        if left_idx < len(landmarks) and right_idx < len(landmarks):
            x_samples.append((landmarks[left_idx][0] + landmarks[right_idx][0]) / 2.0)

    mid_x = float(np.median(x_samples))
    top_y = float(landmarks[forehead_idx][1])
    bottom_y = float(landmarks[chin_idx][1])

    return (
        np.array([mid_x, top_y], dtype=np.float64),
        np.array([mid_x, bottom_y], dtype=np.float64),
    )
