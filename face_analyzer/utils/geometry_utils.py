"""Advanced geometric utilities for facial analysis."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, Delaunay

from face_analyzer.utils.geometry import (
    euclidean_distance,
    mirror_point_across_line,
    perpendicular_distance_to_line,
    polygon_area,
)


def centroid(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute centroid of a set of 2D points."""
    if len(points) == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    return np.mean(points, axis=0)


def polygon_perimeter(points: Sequence[NDArray[np.float64]]) -> float:
    """Compute perimeter of a closed polygon."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points)):
        total += euclidean_distance(points[i], points[(i + 1) % len(points)])
    return total


def convex_hull_points(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ordered convex hull vertices."""
    if len(points) < 3:
        return points.copy()
    hull = ConvexHull(points)
    return points[hull.vertices]


def convex_hull_area(points: NDArray[np.float64]) -> float:
    """Compute area of the convex hull."""
    if len(points) < 3:
        return 0.0
    hull = ConvexHull(points)
    return float(hull.volume)


def delaunay_area(points: NDArray[np.float64]) -> float:
    """Compute total area covered by Delaunay triangulation."""
    if len(points) < 3:
        return 0.0
    try:
        tri = Delaunay(points)
        total = 0.0
        for simplex in tri.simplices:
            triangle = points[simplex]
            total += polygon_area([triangle[0], triangle[1], triangle[2]])
        return total
    except Exception:
        return polygon_area(points)


def rms_error(values: Sequence[float]) -> float:
    """Root mean square of a sequence."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.sqrt(np.mean(arr ** 2)))


def relative_symmetry_score(mean_error: float, scale: float) -> float:
    """Convert mean mirror error to 0-100 symmetry score (calibrated)."""
    relative = mean_error / (scale + 1e-10)
    return float(np.clip(100.0 - relative * 100.0, 0.0, 100.0))


def compute_regional_asymmetry(
    points: NDArray[np.float64],
    pairs: list[tuple[int, int]],
    mid_top: NDArray[np.float64],
    mid_bottom: NDArray[np.float64],
    scale: float,
) -> dict[str, float]:
    """Compute asymmetry metrics for a region from mirror pairs."""
    distances: list[float] = []
    for left_idx, right_idx in pairs:
        if left_idx == right_idx:
            continue
        if left_idx >= len(points) or right_idx >= len(points):
            continue
        left_pt = points[left_idx]
        right_pt = points[right_idx]
        mirrored = mirror_point_across_line(right_pt, mid_top, mid_bottom)
        distances.append(euclidean_distance(left_pt, mirrored))

    if not distances:
        return {
            "mean_distance": 0.0,
            "rms_error": 0.0,
            "symmetry_percent": 100.0,
            "score": 100.0,
        }

    mean_dist = float(np.mean(distances))
    rms = rms_error(distances)
    relative = mean_dist / (scale + 1e-10)
    symmetry_pct = float(np.clip(100.0 - relative * 80.0, 0.0, 100.0))
    score = relative_symmetry_score(mean_dist, scale)

    return {
        "mean_distance": mean_dist,
        "rms_error": rms,
        "symmetry_percent": symmetry_pct,
        "score": score,
    }


def fit_line_pca(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit a line through points using PCA; returns two endpoints spanning the data."""
    center = centroid(points)
    centered = points - center
    if len(points) < 2:
        return center, center + np.array([0.0, 1.0])

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    projections = centered @ direction
    t_min, t_max = projections.min(), projections.max()
    p1 = center + t_min * direction
    p2 = center + t_max * direction
    return p1, p2


def anatomical_midline(
    points: NDArray[np.float64],
    indices: list[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build anatomical midline from glabella, nose, philtrum, chin landmarks."""
    mid_points = np.array([points[i] for i in indices if i < len(points)], dtype=np.float64)
    return fit_line_pca(mid_points)


def axis_deviation_metrics(
    points: NDArray[np.float64],
    axis_indices: list[int],
    mid_top: NDArray[np.float64],
    mid_bottom: NDArray[np.float64],
    scale: float,
) -> dict[str, float]:
    """Compute deviation of anatomical landmarks from the facial midline."""
    deviations: list[float] = []
    lateral: list[float] = []
    for idx in axis_indices:
        if idx >= len(points):
            continue
        pt = points[idx]
        dev = perpendicular_distance_to_line(pt, mid_top, mid_bottom)
        deviations.append(dev)
        lateral.append(pt[0] - (mid_top[0] + mid_bottom[0]) / 2)

    if not deviations:
        return {"angular_deviation": 0.0, "lateral_deviation": 0.0, "mean_error": 0.0}

    axis_p1, axis_p2 = anatomical_midline(points, axis_indices)
    angular = math.degrees(
        math.atan2(axis_p2[0] - axis_p1[0], axis_p2[1] - axis_p1[1])
    ) - math.degrees(
        math.atan2(mid_bottom[0] - mid_top[0], mid_bottom[1] - mid_top[1])
    )

    return {
        "angular_deviation": float(abs(angular)),
        "lateral_deviation": float(np.mean(np.abs(lateral)) / (scale + 1e-10)),
        "mean_error": float(np.mean(deviations) / (scale + 1e-10)),
    }


def estimate_head_pose(
    points: NDArray[np.float64],
    z_coords: NDArray[np.float64] | None,
    left_eye: NDArray[np.float64],
    right_eye: NDArray[np.float64],
    nose_tip: NDArray[np.float64],
    chin: NDArray[np.float64],
    forehead: NDArray[np.float64],
) -> dict[str, float]:
    """Estimate pitch, yaw, and roll in degrees from 2D/3D landmarks."""
    eye_mid = (left_eye + right_eye) / 2
    interocular = euclidean_distance(left_eye, right_eye) + 1e-10

    roll = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    yaw_2d = math.degrees(math.atan2(nose_tip[0] - eye_mid[0], interocular * 0.5))
    if z_coords is not None and len(z_coords) > 454:
        z_left = float(np.mean([z_coords[234], z_coords[127], z_coords[33]]))
        z_right = float(np.mean([z_coords[454], z_coords[356], z_coords[263]]))
        yaw_depth = math.degrees(math.atan2(z_left - z_right, 0.05))
        yaw = (yaw_2d + yaw_depth) / 2
    else:
        cheek_left = points[234][0]
        cheek_right = points[454][0]
        face_width = abs(cheek_right - cheek_left) + 1e-10
        asymmetry = (abs(nose_tip[0] - cheek_left) - abs(cheek_right - nose_tip[0])) / face_width
        yaw = yaw_2d + asymmetry * 15.0

    nose_chin_dy = chin[1] - nose_tip[1]
    forehead_nose_dy = nose_tip[1] - forehead[1]
    pitch_ratio = forehead_nose_dy / (nose_chin_dy + 1e-10)
    pitch = math.degrees(math.atan(pitch_ratio - 1.0))

    if z_coords is not None and len(z_coords) > 152:
        pitch_z = math.degrees(math.atan2(
            float(z_coords[1] - z_coords[152]),
            float(forehead[1] - chin[1]) / (abs(forehead[1] - chin[1]) + 1e-10),
        ))
        pitch = (pitch + pitch_z) / 2

    return {"pitch": pitch, "yaw": yaw, "roll": roll}


def interpret_head_tilt(pitch: float, yaw: float, roll: float) -> dict[str, str]:
    """Provide textual interpretation for head pose angles."""
    def _interp(angle: float, name: str) -> str:
        abs_a = abs(angle)
        if abs_a < 3:
            return f"{name} = {angle:.1f}° — Head practically aligned."
        if abs_a < 10:
            return f"{name} = {angle:.1f}° — Slight {name.lower()} deviation."
        return f"{name} = {angle:.1f}° — Significant {name.lower()} deviation."

    return {
        "pitch": _interp(pitch, "Pitch"),
        "yaw": _interp(yaw, "Yaw"),
        "roll": _interp(roll, "Roll"),
    }


def circularity(area: float, perimeter: float) -> float:
    """Circularity index: 4*pi*area / perimeter^2 (1.0 = perfect circle)."""
    if perimeter <= 0:
        return 0.0
    return float(4 * math.pi * area / (perimeter ** 2))


def elongation(width: float, height: float) -> float:
    """Elongation ratio: height / width."""
    if width <= 0:
        return 0.0
    return height / width


def eye_aspect_ratio(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
    p4: NDArray[np.float64],
    p5: NDArray[np.float64],
    p6: NDArray[np.float64],
) -> float:
    """Compute Eye Aspect Ratio (EAR) from six eye landmarks."""
    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal <= 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def classify_face_shape(
    width: float,
    height: float,
    forehead_width: float,
    cheek_width: float,
    jaw_width: float,
    circularity_val: float,
    elongation_val: float,
    convexity_val: float,
) -> dict[str, float]:
    """Probabilistic classification of face shape from geometric descriptors."""
    wh = width / (height + 1e-10)
    forehead_jaw = forehead_width / (jaw_width + 1e-10)
    cheek_jaw = cheek_width / (jaw_width + 1e-10)
    cheek_forehead = cheek_width / (forehead_width + 1e-10)

    prototypes = {
        "oval": {"wh": 0.75, "circ": 0.78, "elong": 1.35, "f_j": 1.05, "c_j": 1.1, "c_f": 1.05},
        "round": {"wh": 0.90, "circ": 0.90, "elong": 1.05, "f_j": 1.0, "c_j": 1.0, "c_f": 1.0},
        "square": {"wh": 0.88, "circ": 0.82, "elong": 1.10, "f_j": 1.0, "c_j": 1.0, "c_f": 1.0},
        "rectangular": {"wh": 0.72, "circ": 0.70, "elong": 1.45, "f_j": 1.0, "c_j": 1.0, "c_f": 1.0},
        "heart": {"wh": 0.78, "circ": 0.75, "elong": 1.30, "f_j": 1.20, "c_j": 1.15, "c_f": 1.10},
        "diamond": {"wh": 0.76, "circ": 0.72, "elong": 1.32, "f_j": 0.85, "c_j": 1.25, "c_f": 1.20},
        "triangular": {"wh": 0.80, "circ": 0.74, "elong": 1.25, "f_j": 0.80, "c_j": 1.05, "c_f": 0.90},
    }

    features = {
        "wh": wh,
        "circ": circularity_val,
        "elong": elongation_val,
        "f_j": forehead_jaw,
        "c_j": cheek_jaw,
        "c_f": cheek_forehead,
    }

    scores: dict[str, float] = {}
    for shape, proto in prototypes.items():
        dist = sum((features[k] - proto[k]) ** 2 for k in features)
        scores[shape] = math.exp(-dist * 8.0)

    total = sum(scores.values()) + 1e-10
    return {k: round(v / total * 100, 2) for k, v in scores.items()}


def cupid_bow_symmetry(
    left_peak: NDArray[np.float64],
    right_peak: NDArray[np.float64],
    center: NDArray[np.float64],
    mid_top: NDArray[np.float64],
    mid_bottom: NDArray[np.float64],
    scale: float,
) -> float:
    """Symmetry of cupid's bow using mirrored peak distances to center."""
    left_dist = euclidean_distance(left_peak, center)
    right_mirrored = mirror_point_across_line(right_peak, mid_top, mid_bottom)
    right_dist = euclidean_distance(right_mirrored, center)
    diff = abs(left_dist - right_dist) / (scale + 1e-10)
    return float(np.clip(100.0 - diff * 80.0, 0.0, 100.0))
