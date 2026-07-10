"""Funções geométricas para análise facial."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


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
    """Pontua valor (0-100) baseado em faixa de referência antropométrica."""
    if ref_min <= value <= ref_max:
        deviation = abs(value - ref_ideal) / (ref_max - ref_min + 1e-10)
        return float(np.clip(100.0 - deviation * 50.0, 70.0, 100.0))
    distance = min(abs(value - ref_min), abs(value - ref_max))
    range_size = ref_max - ref_min + 1e-10
    penalty = (distance / range_size) * 50.0
    return float(np.clip(70.0 - penalty, 0.0, 69.0))


def symmetry_score(
    left_point: NDArray[np.float64],
    right_mirrored: NDArray[np.float64],
    face_scale: float,
) -> float:
    """Calcula score de simetria (0-100) entre ponto e seu espelho."""
    distance = euclidean_distance(left_point, right_mirrored)
    relative_error = distance / (face_scale + 1e-10)
    return float(np.clip(100.0 - relative_error * 500.0, 0.0, 100.0))


def compute_midline(
    landmarks: NDArray[np.float64],
    top_idx: int,
    bottom_idx: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Define linha média facial entre topo da testa e queixo."""
    return landmarks[top_idx], landmarks[bottom_idx]
