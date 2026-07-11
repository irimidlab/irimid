"""Advanced geometric and anthropometric facial analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from face_analyzer.analyzer.landmarks import FaceLandmarks
from face_analyzer.constants import (
    ADVANCED_REGION_PAIRS,
    ADVANCED_REGION_WEIGHTS,
    ANATOMICAL_AXIS_INDICES,
    CHIN,
    FACE_OVAL,
    FOREHEAD,
    FOREHEAD_REGION,
    LEFT_BROW,
    LEFT_CHEEK,
    LEFT_EYE_BOTTOM,
    LEFT_EYE_CONTOUR,
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_EYE_TOP,
    LEFT_JAW,
    LEFT_MOUTH,
    LOWER_LIP_CENTER,
    LIPS_OUTER,
    NOSE_LEFT,
    NOSE_RIGHT,
    NOSE_TIP,
    REGION_AREA_INDICES,
    RIGHT_BROW,
    RIGHT_CHEEK,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_CONTOUR,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_TOP,
    RIGHT_JAW,
    RIGHT_MOUTH,
    UPPER_LIP_CENTER,
)
from face_analyzer.utils.geometry import (
    euclidean_distance,
    line_angle,
    mirror_point_across_line,
    midpoint,
    perpendicular_distance_to_line,
    polygon_area,
    symmetry_score,
)
from face_analyzer.utils.geometry_utils import (
    anatomical_midline,
    axis_deviation_metrics,
    centroid,
    circularity,
    classify_face_shape,
    compute_regional_asymmetry,
    convex_hull_area,
    cupid_bow_symmetry,
    delaunay_area,
    elongation,
    estimate_head_pose,
    eye_aspect_ratio,
    interpret_head_tilt,
    polygon_perimeter,
)


@dataclass
class RegionalAsymmetryResult:
    """Asymmetry metrics for a single facial region."""

    mean_distance: float = 0.0
    rms_error: float = 0.0
    symmetry_percent: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_distance": round(self.mean_distance, 4),
            "rms_error": round(self.rms_error, 4),
            "symmetry_percent": round(self.symmetry_percent, 2),
            "score": round(self.score, 2),
        }


@dataclass
class AdvancedGeometryMetrics:
    """Complete advanced geometry analysis results."""

    regional_asymmetry: dict[str, RegionalAsymmetryResult] = field(default_factory=dict)
    regional_symmetry_overall: float = 0.0
    geometric_center_x: float = 0.0
    geometric_center_y: float = 0.0
    center_midline_distance: float = 0.0
    center_offset_percent: float = 0.0
    axis_angular_deviation: float = 0.0
    axis_lateral_deviation: float = 0.0
    axis_mean_error: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    head_pitch_interpretation: str = ""
    head_yaw_interpretation: str = ""
    head_roll_interpretation: str = ""
    face_max_width: float = 0.0
    face_height: float = 0.0
    face_circularity: float = 0.0
    face_elongation: float = 0.0
    face_convexity: float = 0.0
    face_shape_probabilities: dict[str, float] = field(default_factory=dict)
    face_area_total: float = 0.0
    face_perimeter: float = 0.0
    face_area_normalized: float = 0.0
    region_areas: dict[str, float] = field(default_factory=dict)
    region_area_percentages: dict[str, float] = field(default_factory=dict)
    forehead_height: float = 0.0
    forehead_width: float = 0.0
    forehead_wh_ratio: float = 0.0
    forehead_zygoma_ratio: float = 0.0
    brow_length_left: float = 0.0
    brow_length_right: float = 0.0
    brow_thickness_avg: float = 0.0
    brow_eye_distance: float = 0.0
    brow_inter_distance: float = 0.0
    brow_tilt_left: float = 0.0
    brow_tilt_right: float = 0.0
    brow_curvature_left: float = 0.0
    brow_curvature_right: float = 0.0
    brow_symmetry_score: float = 0.0
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    eye_area_left: float = 0.0
    eye_area_right: float = 0.0
    eye_wh_ratio_left: float = 0.0
    eye_wh_ratio_right: float = 0.0
    pupil_center_distance_left: float = 0.0
    pupil_center_distance_right: float = 0.0
    eye_shape_symmetry: float = 0.0
    eye_area_diff_percent: float = 0.0
    zygoma_width: float = 0.0
    zygoma_jaw_ratio: float = 0.0
    zygoma_forehead_ratio: float = 0.0
    nose_index: float = 0.0
    nostril_area_left: float = 0.0
    nostril_area_right: float = 0.0
    nostril_symmetry: float = 0.0
    nose_width: float = 0.0
    nose_height: float = 0.0
    nose_classification: str = ""
    lip_area: float = 0.0
    cupid_bow_symmetry: float = 0.0
    mouth_relative_width: float = 0.0
    mouth_nose_distance_ratio: float = 0.0
    mouth_chin_distance_ratio: float = 0.0
    chin_wh_ratio: float = 0.0
    chin_symmetry: float = 0.0
    chin_projection_index: float = 0.0
    jaw_body_length: float = 0.0
    jaw_ramus_length_est: float = 0.0
    jaw_face_ratio: float = 0.0
    jaw_definition_geometric: float = 0.0
    anatomical_axis: tuple[np.ndarray, np.ndarray] | None = None
    geometric_center: np.ndarray | None = None

    def to_dict(self) -> dict[str, object]:
        """Nested dictionary representation."""
        return {
            "regional_asymmetry": {
                k: v.to_dict() for k, v in self.regional_asymmetry.items()
            },
            "regional_symmetry_overall": round(self.regional_symmetry_overall, 2),
            "geometric_center": {
                "x": round(self.geometric_center_x, 2),
                "y": round(self.geometric_center_y, 2),
                "midline_distance": round(self.center_midline_distance, 4),
                "offset_percent": round(self.center_offset_percent, 2),
            },
            "facial_axis_deviation": {
                "angular": round(self.axis_angular_deviation, 2),
                "lateral": round(self.axis_lateral_deviation, 4),
                "mean_error": round(self.axis_mean_error, 4),
            },
            "head_pose": {
                "pitch": round(self.head_pitch, 2),
                "yaw": round(self.head_yaw, 2),
                "roll": round(self.head_roll, 2),
                "pitch_interpretation": self.head_pitch_interpretation,
                "yaw_interpretation": self.head_yaw_interpretation,
                "roll_interpretation": self.head_roll_interpretation,
            },
            "ovality": {
                "max_width": round(self.face_max_width, 2),
                "height": round(self.face_height, 2),
                "circularity": round(self.face_circularity, 4),
                "elongation": round(self.face_elongation, 4),
                "convexity": round(self.face_convexity, 4),
                "shape_probabilities": self.face_shape_probabilities,
            },
            "face_area": {
                "total": round(self.face_area_total, 2),
                "perimeter": round(self.face_perimeter, 2),
                "normalized": round(self.face_area_normalized, 6),
            },
            "region_areas": {
                "absolute": {k: round(v, 2) for k, v in self.region_areas.items()},
                "percentages": {k: round(v, 2) for k, v in self.region_area_percentages.items()},
            },
            "forehead": {
                "height": round(self.forehead_height, 2),
                "width": round(self.forehead_width, 2),
                "wh_ratio": round(self.forehead_wh_ratio, 4),
                "zygoma_ratio": round(self.forehead_zygoma_ratio, 4),
            },
            "eyebrows": {
                "length_left": round(self.brow_length_left, 2),
                "length_right": round(self.brow_length_right, 2),
                "thickness_avg": round(self.brow_thickness_avg, 4),
                "eye_distance": round(self.brow_eye_distance, 4),
                "inter_distance": round(self.brow_inter_distance, 4),
                "tilt_left": round(self.brow_tilt_left, 2),
                "tilt_right": round(self.brow_tilt_right, 2),
                "curvature_left": round(self.brow_curvature_left, 4),
                "curvature_right": round(self.brow_curvature_right, 4),
                "symmetry_score": round(self.brow_symmetry_score, 2),
            },
            "eyes_advanced": {
                "ear_left": round(self.ear_left, 4),
                "ear_right": round(self.ear_right, 4),
                "ear_avg": round(self.ear_avg, 4),
                "area_left": round(self.eye_area_left, 2),
                "area_right": round(self.eye_area_right, 2),
                "wh_ratio_left": round(self.eye_wh_ratio_left, 4),
                "wh_ratio_right": round(self.eye_wh_ratio_right, 4),
                "pupil_center_dist_left": round(self.pupil_center_distance_left, 4),
                "pupil_center_dist_right": round(self.pupil_center_distance_right, 4),
                "shape_symmetry": round(self.eye_shape_symmetry, 2),
                "area_diff_percent": round(self.eye_area_diff_percent, 2),
            },
            "zygomas": {
                "width": round(self.zygoma_width, 2),
                "jaw_ratio": round(self.zygoma_jaw_ratio, 4),
                "forehead_ratio": round(self.zygoma_forehead_ratio, 4),
            },
            "nose_advanced": {
                "index": round(self.nose_index, 4),
                "nostril_area_left": round(self.nostril_area_left, 2),
                "nostril_area_right": round(self.nostril_area_right, 2),
                "nostril_symmetry": round(self.nostril_symmetry, 2),
                "width": round(self.nose_width, 2),
                "height": round(self.nose_height, 2),
                "classification": self.nose_classification,
            },
            "mouth_advanced": {
                "lip_area": round(self.lip_area, 2),
                "cupid_bow_symmetry": round(self.cupid_bow_symmetry, 2),
                "relative_width": round(self.mouth_relative_width, 4),
                "nose_distance_ratio": round(self.mouth_nose_distance_ratio, 4),
                "chin_distance_ratio": round(self.mouth_chin_distance_ratio, 4),
            },
            "chin_advanced": {
                "wh_ratio": round(self.chin_wh_ratio, 4),
                "symmetry": round(self.chin_symmetry, 2),
                "projection_index": round(self.chin_projection_index, 4),
            },
            "jaw_advanced": {
                "body_length": round(self.jaw_body_length, 2),
                "ramus_length_est": round(self.jaw_ramus_length_est, 2),
                "face_ratio": round(self.jaw_face_ratio, 4),
                "definition_geometric": round(self.jaw_definition_geometric, 2),
            },
        }

    def to_flat_dict(self) -> dict[str, object]:
        """Flat dictionary for export."""
        flat: dict[str, object] = {}
        for region, data in self.regional_asymmetry.items():
            for key, val in data.to_dict().items():
                flat[f"adv_asym_{region}_{key}"] = val
        flat["adv_regional_symmetry_overall"] = round(self.regional_symmetry_overall, 2)
        flat["adv_center_x"] = round(self.geometric_center_x, 2)
        flat["adv_center_y"] = round(self.geometric_center_y, 2)
        flat["adv_center_midline_distance"] = round(self.center_midline_distance, 4)
        flat["adv_center_offset_percent"] = round(self.center_offset_percent, 2)
        flat["adv_axis_angular_deviation"] = round(self.axis_angular_deviation, 2)
        flat["adv_axis_lateral_deviation"] = round(self.axis_lateral_deviation, 4)
        flat["adv_axis_mean_error"] = round(self.axis_mean_error, 4)
        flat["adv_head_pitch"] = round(self.head_pitch, 2)
        flat["adv_head_yaw"] = round(self.head_yaw, 2)
        flat["adv_head_roll"] = round(self.head_roll, 2)
        flat["adv_face_circularity"] = round(self.face_circularity, 4)
        flat["adv_face_elongation"] = round(self.face_elongation, 4)
        flat["adv_face_convexity"] = round(self.face_convexity, 4)
        flat["adv_face_area_total"] = round(self.face_area_total, 2)
        flat["adv_face_perimeter"] = round(self.face_perimeter, 2)
        flat["adv_face_area_normalized"] = round(self.face_area_normalized, 6)
        flat["adv_ear_left"] = round(self.ear_left, 4)
        flat["adv_ear_right"] = round(self.ear_right, 4)
        flat["adv_ear_avg"] = round(self.ear_avg, 4)
        flat["adv_nose_index"] = round(self.nose_index, 4)
        flat["adv_jaw_definition_geometric"] = round(self.jaw_definition_geometric, 2)
        for shape, prob in self.face_shape_probabilities.items():
            flat[f"adv_shape_{shape}"] = prob
        for region, area in self.region_areas.items():
            flat[f"adv_area_{region}"] = round(area, 2)
            flat[f"adv_area_pct_{region}"] = round(
                self.region_area_percentages.get(region, 0), 2
            )
        nested = self.to_dict()
        for section in ("forehead", "eyebrows", "zygomas", "eyes_advanced",
                        "nose_advanced", "mouth_advanced", "chin_advanced", "jaw_advanced"):
            if section in nested and isinstance(nested[section], dict):
                for k, v in nested[section].items():
                    if not isinstance(v, dict):
                        flat[f"adv_{section}_{k}"] = v
        return flat


class AdvancedGeometryAnalyzer:
    """Computes advanced geometric facial metrics from landmarks."""

    def analyze(self, landmarks: FaceLandmarks) -> AdvancedGeometryMetrics:
        """Run full advanced geometry analysis."""
        points = landmarks.points
        scale = landmarks.face_scale
        mid_top, mid_bottom = landmarks.midline
        z_coords = landmarks.z_coords

        metrics = AdvancedGeometryMetrics()
        metrics.regional_asymmetry = self._compute_regional_asymmetry(
            points, mid_top, mid_bottom, scale
        )
        metrics.regional_symmetry_overall = self._weighted_symmetry(
            metrics.regional_asymmetry
        )
        self._compute_geometric_center(metrics, points, mid_top, mid_bottom, scale)
        self._compute_axis_deviation(metrics, points, mid_top, mid_bottom, scale)
        self._compute_head_pose(metrics, landmarks, points, z_coords)
        self._compute_ovality(metrics, points, scale)
        self._compute_face_area(metrics, points, scale)
        self._compute_region_areas(metrics, points)
        self._compute_forehead(metrics, points, scale)
        self._compute_eyebrows(metrics, points, mid_top, mid_bottom, scale)
        self._compute_eyes_advanced(metrics, landmarks, points, mid_top, mid_bottom, scale)
        self._compute_zygomas(metrics, points, scale)
        self._compute_nose_advanced(metrics, points, mid_top, mid_bottom, scale)
        self._compute_mouth_advanced(metrics, points, mid_top, mid_bottom, scale)
        self._compute_chin_advanced(metrics, points, mid_top, mid_bottom, scale)
        self._compute_jaw_advanced(metrics, points, scale)
        return metrics

    def _compute_regional_asymmetry(
        self,
        points: np.ndarray,
        mid_top: np.ndarray,
        mid_bottom: np.ndarray,
        scale: float,
    ) -> dict[str, RegionalAsymmetryResult]:
        results: dict[str, RegionalAsymmetryResult] = {}
        for region, pairs in ADVANCED_REGION_PAIRS.items():
            data = compute_regional_asymmetry(points, pairs, mid_top, mid_bottom, scale)
            results[region] = RegionalAsymmetryResult(**data)
        return results

    def _weighted_symmetry(
        self,
        regions: dict[str, RegionalAsymmetryResult],
    ) -> float:
        total = 0.0
        weight_sum = 0.0
        for name, result in regions.items():
            w = ADVANCED_REGION_WEIGHTS.get(name, 1.0 / len(regions))
            total += result.score * w
            weight_sum += w
        return total / (weight_sum + 1e-10)

    def _compute_geometric_center(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        mid_top: np.ndarray,
        mid_bottom: np.ndarray,
        scale: float,
    ) -> None:
        oval_pts = np.array([points[i] for i in FACE_OVAL if i < len(points)])
        center = centroid(oval_pts)
        metrics.geometric_center = center
        metrics.geometric_center_x = float(center[0])
        metrics.geometric_center_y = float(center[1])
        metrics.center_midline_distance = perpendicular_distance_to_line(
            center, mid_top, mid_bottom
        )
        metrics.center_offset_percent = (
            metrics.center_midline_distance / (scale + 1e-10) * 100
        )

    def _compute_axis_deviation(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        mid_top: np.ndarray,
        mid_bottom: np.ndarray,
        scale: float,
    ) -> None:
        axis = anatomical_midline(points, ANATOMICAL_AXIS_INDICES)
        metrics.anatomical_axis = axis
        dev = axis_deviation_metrics(
            points, ANATOMICAL_AXIS_INDICES, mid_top, mid_bottom, scale
        )
        metrics.axis_angular_deviation = dev["angular_deviation"]
        metrics.axis_lateral_deviation = dev["lateral_deviation"]
        metrics.axis_mean_error = dev["mean_error"]

    def _compute_head_pose(
        self,
        metrics: AdvancedGeometryMetrics,
        landmarks: FaceLandmarks,
        points: np.ndarray,
        z_coords: np.ndarray | None,
    ) -> None:
        left_eye = (points[LEFT_EYE_TOP] + points[LEFT_EYE_BOTTOM]) / 2
        right_eye = (points[RIGHT_EYE_TOP] + points[RIGHT_EYE_BOTTOM]) / 2
        pose = estimate_head_pose(
            points, z_coords, left_eye, right_eye,
            points[NOSE_TIP], points[CHIN], points[FOREHEAD],
        )
        metrics.head_pitch = pose["pitch"]
        metrics.head_yaw = pose["yaw"]
        metrics.head_roll = pose["roll"]
        interp = interpret_head_tilt(pose["pitch"], pose["yaw"], pose["roll"])
        metrics.head_pitch_interpretation = interp["pitch"]
        metrics.head_yaw_interpretation = interp["yaw"]
        metrics.head_roll_interpretation = interp["roll"]

    def _compute_ovality(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        scale: float,
    ) -> None:
        oval_pts = np.array([points[i] for i in FACE_OVAL if i < len(points)])
        width = float(np.max(oval_pts[:, 0]) - np.min(oval_pts[:, 0]))
        height = float(np.max(oval_pts[:, 1]) - np.min(oval_pts[:, 1]))
        area = polygon_area(oval_pts)
        perim = polygon_perimeter(oval_pts)
        hull_area = convex_hull_area(oval_pts)

        metrics.face_max_width = width
        metrics.face_height = height
        metrics.face_circularity = circularity(area, perim)
        metrics.face_elongation = elongation(width, height)
        metrics.face_convexity = area / (hull_area + 1e-10)

        forehead_w = euclidean_distance(points[127], points[356])
        cheek_w = euclidean_distance(points[LEFT_CHEEK], points[RIGHT_CHEEK])
        jaw_w = euclidean_distance(points[LEFT_JAW], points[RIGHT_JAW])

        metrics.face_shape_probabilities = classify_face_shape(
            width, height, forehead_w, cheek_w, jaw_w,
            metrics.face_circularity, metrics.face_elongation, metrics.face_convexity,
        )

    def _compute_face_area(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        scale: float,
    ) -> None:
        oval_pts = [points[i] for i in FACE_OVAL if i < len(points)]
        area = polygon_area(oval_pts)
        perim = polygon_perimeter(oval_pts)
        metrics.face_area_total = area
        metrics.face_perimeter = perim
        metrics.face_area_normalized = area / ((scale ** 2) + 1e-10)

    def _compute_region_areas(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
    ) -> None:
        total = metrics.face_area_total + 1e-10
        for region, indices in REGION_AREA_INDICES.items():
            region_pts = [points[i] for i in indices if i < len(points)]
            if len(region_pts) >= 3:
                area = delaunay_area(np.array(region_pts))
            else:
                area = 0.0
            metrics.region_areas[region] = area
            metrics.region_area_percentages[region] = (area / total) * 100

    def _compute_forehead(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        scale: float,
    ) -> None:
        brow_y = (points[LEFT_EYE_TOP][1] + points[RIGHT_EYE_TOP][1]) / 2
        metrics.forehead_height = brow_y - points[FOREHEAD][1]
        fh_pts = [points[i] for i in FOREHEAD_REGION if i < len(points)]
        metrics.forehead_width = float(np.max([p[0] for p in fh_pts]) - np.min([p[0] for p in fh_pts]))
        metrics.forehead_wh_ratio = metrics.forehead_width / (metrics.forehead_height + 1e-10)
        zygoma_w = euclidean_distance(points[LEFT_CHEEK], points[RIGHT_CHEEK])
        metrics.forehead_zygoma_ratio = metrics.forehead_width / (zygoma_w + 1e-10)

    def _compute_eyebrows(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        mid_top: np.ndarray,
        mid_bottom: np.ndarray,
        scale: float,
    ) -> None:
        metrics.brow_length_left = sum(
            euclidean_distance(points[LEFT_BROW[i]], points[LEFT_BROW[i + 1]])
            for i in range(len(LEFT_BROW) - 1)
        )
        metrics.brow_length_right = sum(
            euclidean_distance(points[RIGHT_BROW[i]], points[RIGHT_BROW[i + 1]])
            for i in range(len(RIGHT_BROW) - 1)
        )
        left_eye_top = points[LEFT_EYE_TOP][1]
        left_brow_bottom = points[LEFT_BROW[2]][1]
        metrics.brow_thickness_avg = abs(left_brow_bottom - left_eye_top) / scale
        metrics.brow_eye_distance = abs(points[LEFT_BROW[2]][1] - points[LEFT_EYE_TOP][1]) / scale
        metrics.brow_inter_distance = euclidean_distance(points[LEFT_BROW[0]], points[RIGHT_BROW[0]]) / scale
        metrics.brow_tilt_left = line_angle(points[LEFT_BROW[0]], points[LEFT_BROW[-1]])
        metrics.brow_tilt_right = line_angle(points[RIGHT_BROW[0]], points[RIGHT_BROW[-1]])
        brow_mid_left = midpoint(points[LEFT_BROW[0]], points[LEFT_BROW[-1]])
        brow_mid_right = midpoint(points[RIGHT_BROW[0]], points[RIGHT_BROW[-1]])
        eye_mid = (points[LEFT_EYE_TOP] + points[RIGHT_EYE_TOP]) / 2
        metrics.brow_curvature_left = euclidean_distance(points[LEFT_BROW[2]], brow_mid_left) / scale
        metrics.brow_curvature_right = euclidean_distance(points[RIGHT_BROW[2]], brow_mid_right) / scale
        left_center = midpoint(points[LEFT_BROW[0]], points[LEFT_BROW[-1]])
        right_mirrored = mirror_point_across_line(brow_mid_right, mid_top, mid_bottom)
        metrics.brow_symmetry_score = symmetry_score(left_center, right_mirrored, scale)

    def _compute_eyes_advanced(
        self,
        metrics: AdvancedGeometryMetrics,
        landmarks: FaceLandmarks,
        points: np.ndarray,
        mid_top: np.ndarray,
        mid_bottom: np.ndarray,
        scale: float,
    ) -> None:
        metrics.ear_left = eye_aspect_ratio(
            points[LEFT_EYE_OUTER], points[LEFT_EYE_TOP], points[LEFT_EYE_BOTTOM],
            points[LEFT_EYE_INNER], points[159], points[145],
        )
        metrics.ear_right = eye_aspect_ratio(
            points[RIGHT_EYE_OUTER], points[RIGHT_EYE_TOP], points[RIGHT_EYE_BOTTOM],
            points[RIGHT_EYE_INNER], points[386], points[374],
        )
        metrics.ear_avg = (metrics.ear_left + metrics.ear_right) / 2

        left_contour = [points[i] for i in LEFT_EYE_CONTOUR if i < len(points)]
        right_contour = [points[i] for i in RIGHT_EYE_CONTOUR if i < len(points)]
        metrics.eye_area_left = polygon_area(left_contour)
        metrics.eye_area_right = polygon_area(right_contour)

        left_w = euclidean_distance(points[LEFT_EYE_OUTER], points[LEFT_EYE_INNER])
        left_h = euclidean_distance(points[LEFT_EYE_TOP], points[LEFT_EYE_BOTTOM])
        right_w = euclidean_distance(points[RIGHT_EYE_OUTER], points[RIGHT_EYE_INNER])
        right_h = euclidean_distance(points[RIGHT_EYE_TOP], points[RIGHT_EYE_BOTTOM])
        metrics.eye_wh_ratio_left = left_w / (left_h + 1e-10)
        metrics.eye_wh_ratio_right = right_w / (right_h + 1e-10)

        face_center = centroid(np.array([points[i] for i in FACE_OVAL if i < len(points)]))
        metrics.pupil_center_distance_left = euclidean_distance(
            landmarks.left_pupil, face_center
        ) / scale
        metrics.pupil_center_distance_right = euclidean_distance(
            landmarks.right_pupil, face_center
        ) / scale

        left_shape = metrics.eye_wh_ratio_left
        right_mirrored_ratio = metrics.eye_wh_ratio_right
        metrics.eye_shape_symmetry = float(np.clip(
            100.0 - abs(left_shape - right_mirrored_ratio) / (left_shape + 1e-10) * 40,
            0, 100,
        ))
        if metrics.eye_area_left > 0:
            metrics.eye_area_diff_percent = abs(
                metrics.eye_area_left - metrics.eye_area_right
            ) / metrics.eye_area_left * 100

    def _compute_zygomas(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        scale: float,
    ) -> None:
        metrics.zygoma_width = euclidean_distance(points[LEFT_CHEEK], points[RIGHT_CHEEK])
        jaw_w = euclidean_distance(points[LEFT_JAW], points[RIGHT_JAW])
        fh_pts = [points[i] for i in FOREHEAD_REGION if i < len(points)]
        forehead_w = float(np.max([p[0] for p in fh_pts]) - np.min([p[0] for p in fh_pts]))
        metrics.zygoma_jaw_ratio = metrics.zygoma_width / (jaw_w + 1e-10)
        metrics.zygoma_forehead_ratio = metrics.zygoma_width / (forehead_w + 1e-10)

    def _compute_nose_advanced(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        mid_top: np.ndarray,
        mid_bottom: np.ndarray,
        scale: float,
    ) -> None:
        metrics.nose_width = euclidean_distance(points[NOSE_LEFT], points[NOSE_RIGHT])
        metrics.nose_height = euclidean_distance(points[6], points[NOSE_TIP])
        metrics.nose_index = metrics.nose_width / (metrics.nose_height + 1e-10)

        nostril_left_pts = [points[129], points[49], points[219]]
        nostril_right_pts = [points[358], points[279], points[439]]
        metrics.nostril_area_left = polygon_area(
            [points[i] for i in [129, 49, 219] if i < len(points)]
        )
        metrics.nostril_area_right = polygon_area(
            [points[i] for i in [358, 279, 439] if i < len(points)]
        )
        if metrics.nostril_area_left > 0:
            area_diff = abs(metrics.nostril_area_left - metrics.nostril_area_right)
            metrics.nostril_symmetry = float(np.clip(
                100.0 - area_diff / metrics.nostril_area_left * 40, 0, 100
            ))

        if metrics.nose_index < 0.65:
            metrics.nose_classification = "Leptorrhine (narrow)"
        elif metrics.nose_index < 0.85:
            metrics.nose_classification = "Mesorrhine (medium)"
        else:
            metrics.nose_classification = "Platyrrhine (broad)"

    def _compute_mouth_advanced(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        mid_top: np.ndarray,
        mid_bottom: np.ndarray,
        scale: float,
    ) -> None:
        lip_pts = [points[i] for i in LIPS_OUTER if i < len(points)]
        metrics.lip_area = polygon_area(lip_pts)
        metrics.cupid_bow_symmetry = cupid_bow_symmetry(
            points[61], points[291], points[UPPER_LIP_CENTER],
            mid_top, mid_bottom, scale,
        )
        mouth_w = euclidean_distance(points[LEFT_MOUTH], points[RIGHT_MOUTH])
        face_w = euclidean_distance(points[LEFT_CHEEK], points[RIGHT_CHEEK])
        metrics.mouth_relative_width = mouth_w / (face_w + 1e-10)
        metrics.mouth_nose_distance_ratio = (
            euclidean_distance(points[NOSE_TIP], points[UPPER_LIP_CENTER]) / scale
        )
        metrics.mouth_chin_distance_ratio = (
            euclidean_distance(points[LOWER_LIP_CENTER], points[CHIN]) / scale
        )

    def _compute_chin_advanced(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        mid_top: np.ndarray,
        mid_bottom: np.ndarray,
        scale: float,
    ) -> None:
        chin_w = euclidean_distance(points[LEFT_JAW], points[RIGHT_JAW]) * 0.55
        chin_h = euclidean_distance(points[14], points[CHIN])
        metrics.chin_wh_ratio = chin_w / (chin_h + 1e-10)
        left_jaw = points[LEFT_JAW]
        right_mirrored = mirror_point_across_line(points[RIGHT_JAW], mid_top, mid_bottom)
        metrics.chin_symmetry = symmetry_score(left_jaw, right_mirrored, scale)
        metrics.chin_projection_index = (
            points[CHIN][1] - points[NOSE_TIP][1]
        ) / (scale + 1e-10)

    def _compute_jaw_advanced(
        self,
        metrics: AdvancedGeometryMetrics,
        points: np.ndarray,
        scale: float,
    ) -> None:
        metrics.jaw_body_length = euclidean_distance(points[LEFT_JAW], points[RIGHT_JAW])
        metrics.jaw_ramus_length_est = (
            euclidean_distance(points[LEFT_JAW], points[LEFT_CHEEK])
            + euclidean_distance(points[RIGHT_JAW], points[RIGHT_CHEEK])
        ) / 2
        metrics.jaw_face_ratio = metrics.jaw_body_length / (scale + 1e-10)
        jaw_h = (
            euclidean_distance(points[LEFT_JAW], points[CHIN])
            + euclidean_distance(points[RIGHT_JAW], points[CHIN])
        ) / 2
        metrics.jaw_definition_geometric = float(np.clip(
            (metrics.jaw_body_length / (jaw_h + 1e-10) - 0.85) / 0.55 * 100, 0, 100
        ))
