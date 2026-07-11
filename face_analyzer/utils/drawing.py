"""Funções de desenho e anotação facial."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from face_analyzer.constants import (
    CHIN,
    COLOR_ADV_AXIS,
    COLOR_ADV_CENTROID,
    COLOR_ADV_REGION,
    COLOR_ANGLE,
    COLOR_EYE_AXIS,
    COLOR_LANDMARK,
    COLOR_MIDLINE,
    COLOR_PROPORTION,
    COLOR_SYMMETRY,
    COLOR_THIRDS,
    FACE_OVAL,
    FOREHEAD,
    FOREHEAD_REGION,
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_EYE_TOP,
    LEFT_MOUTH,
    NOSE_TIP,
    REGION_AREA_INDICES,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_TOP,
    RIGHT_MOUTH,
)
from face_analyzer.utils.geometry import line_angle, midpoint


@dataclass
class DrawingLayers:
    """Controle de camadas de visualização."""

    landmarks: bool = True
    midline: bool = True
    eye_axes: bool = True
    facial_thirds: bool = True
    proportions: bool = True
    angles: bool = True
    distances: bool = True
    symmetry_lines: bool = False
    landmark_numbers: bool = False
    advanced_geometry: bool = False


@dataclass
class FaceAnnotator:
    """Desenha anotações faciais sobre a imagem."""

    layers: DrawingLayers = field(default_factory=DrawingLayers)
    landmark_radius: int = 2
    line_thickness: int = 1

    def annotate(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
        metrics: dict | None = None,
        advanced_metrics: object | None = None,
    ) -> NDArray[np.uint8]:
        """Generate annotated image with selected layers."""
        annotated = image.copy()

        if self.layers.midline:
            self._draw_midline(annotated, landmarks)

        if self.layers.facial_thirds:
            self._draw_facial_thirds(annotated, landmarks)

        if self.layers.eye_axes:
            self._draw_eye_axes(annotated, landmarks)

        if self.layers.proportions:
            self._draw_proportions(annotated, landmarks)

        if self.layers.advanced_geometry and advanced_metrics is not None:
            self._draw_advanced_geometry(annotated, landmarks, advanced_metrics)

        if self.layers.angles and metrics:
            self._draw_angles(annotated, landmarks, metrics)

        if self.layers.distances:
            self._draw_distances(annotated, landmarks)

        if self.layers.symmetry_lines:
            self._draw_symmetry_lines(annotated, landmarks)

        if self.layers.landmarks:
            self._draw_landmarks(annotated, landmarks)

        if self.layers.landmark_numbers:
            self._draw_landmark_numbers(annotated, landmarks)

        return annotated

    def _draw_advanced_geometry(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
        advanced_metrics: object,
    ) -> None:
        """Draw advanced geometry overlays: centroid, axis, regions."""
        overlay = image.copy()

        if advanced_metrics.geometric_center is not None:
            cx, cy = advanced_metrics.geometric_center.astype(int)
            cv2.circle(overlay, (cx, cy), 6, COLOR_ADV_CENTROID, -1)
            cv2.putText(
                overlay, "CM", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ADV_CENTROID, 1,
            )

        if advanced_metrics.anatomical_axis is not None:
            p1, p2 = advanced_metrics.anatomical_axis
            cv2.line(
                overlay,
                tuple(p1.astype(int)), tuple(p2.astype(int)),
                COLOR_ADV_AXIS, self.line_thickness + 1,
            )

        region_colors = {
            "forehead": (180, 120, 255),
            "eyes": (120, 200, 255),
            "nose": (255, 180, 120),
            "mouth": (120, 255, 180),
            "chin": (255, 120, 180),
        }
        for region, indices in REGION_AREA_INDICES.items():
            pts = np.array(
                [landmarks[i] for i in indices if i < len(landmarks)],
                dtype=np.int32,
            )
            if len(pts) >= 3:
                color = region_colors.get(region, COLOR_ADV_REGION)
                cv2.polylines(overlay, [pts], True, color, 1, cv2.LINE_AA)

        if hasattr(advanced_metrics, "head_roll"):
            brow_y = int((landmarks[LEFT_EYE_TOP][1] + landmarks[RIGHT_EYE_TOP][1]) / 2)
            cv2.putText(
                overlay,
                f"Roll {advanced_metrics.head_roll:.1f}",
                (10, brow_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ANGLE, 1,
            )

        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    def _draw_midline(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
    ) -> None:
        """Desenha linha média facial."""
        top = landmarks[FOREHEAD].astype(int)
        bottom = landmarks[CHIN].astype(int)
        cv2.line(image, tuple(top), tuple(bottom), COLOR_MIDLINE, self.line_thickness + 1)

    def _draw_facial_thirds(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
    ) -> None:
        """Desenha linhas dos terços faciais."""
        height = image.shape[0]
        brow_y = int((landmarks[LEFT_EYE_TOP][1] + landmarks[RIGHT_EYE_TOP][1]) / 2)
        nose_y = int(landmarks[NOSE_TIP][1])
        for y in [brow_y, nose_y]:
            cv2.line(image, (0, y), (image.shape[1], y), COLOR_THIRDS, self.line_thickness)

    def _draw_eye_axes(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
    ) -> None:
        """Desenha eixos dos olhos."""
        left_inner = landmarks[LEFT_EYE_INNER].astype(int)
        left_outer = landmarks[LEFT_EYE_OUTER].astype(int)
        right_inner = landmarks[RIGHT_EYE_INNER].astype(int)
        right_outer = landmarks[RIGHT_EYE_OUTER].astype(int)
        cv2.line(image, tuple(left_inner), tuple(left_outer), COLOR_EYE_AXIS, self.line_thickness)
        cv2.line(image, tuple(right_inner), tuple(right_outer), COLOR_EYE_AXIS, self.line_thickness)

    def _draw_proportions(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
    ) -> None:
        """Desenha linhas de proporção principais."""
        left_cheek = landmarks[234].astype(int)
        right_cheek = landmarks[454].astype(int)
        left_jaw = landmarks[172].astype(int)
        right_jaw = landmarks[397].astype(int)
        cv2.line(image, tuple(left_cheek), tuple(right_cheek), COLOR_PROPORTION, self.line_thickness)
        cv2.line(image, tuple(left_jaw), tuple(right_jaw), COLOR_PROPORTION, self.line_thickness)

    def _draw_angles(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
        metrics: dict,
    ) -> None:
        """Desenha indicação de ângulos calculados."""
        left_inner = landmarks[LEFT_EYE_INNER]
        left_outer = landmarks[LEFT_EYE_OUTER]
        angle = line_angle(left_inner, left_outer)
        mid = midpoint(left_inner, left_outer).astype(int)
        cv2.putText(
            image,
            f"{angle:.1f}°",
            (mid[0] - 20, mid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            COLOR_ANGLE,
            1,
        )

    def _draw_distances(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
    ) -> None:
        """Desenha distâncias importantes."""
        left_mouth = landmarks[LEFT_MOUTH].astype(int)
        right_mouth = landmarks[RIGHT_MOUTH].astype(int)
        cv2.line(image, tuple(left_mouth), tuple(right_mouth), COLOR_PROPORTION, self.line_thickness)

        if landmarks.shape[0] > 473:
            left_pupil = landmarks[468].astype(int)
            right_pupil = landmarks[473].astype(int)
            cv2.line(image, tuple(left_pupil), tuple(right_pupil), COLOR_PROPORTION, self.line_thickness)
            cv2.circle(image, tuple(left_pupil), 3, COLOR_ANGLE, -1)
            cv2.circle(image, tuple(right_pupil), 3, COLOR_ANGLE, -1)

    def _draw_symmetry_lines(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
    ) -> None:
        """Desenha linhas de simetria entre pares."""
        pairs = [(33, 263), (61, 291), (172, 397)]
        for left_idx, right_idx in pairs:
            left_pt = landmarks[left_idx].astype(int)
            right_pt = landmarks[right_idx].astype(int)
            cv2.line(image, tuple(left_pt), tuple(right_pt), COLOR_SYMMETRY, 1, cv2.LINE_AA)

    def _draw_landmarks(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
    ) -> None:
        """Desenha todos os landmarks."""
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), self.landmark_radius, COLOR_LANDMARK, -1)

    def _draw_landmark_numbers(
        self,
        image: NDArray[np.uint8],
        landmarks: NDArray[np.float64],
    ) -> None:
        """Desenha índices numéricos dos landmarks."""
        for idx, point in enumerate(landmarks):
            x, y = int(point[0]), int(point[1])
            if idx % 5 == 0 or idx in {1, 33, 133, 152, 263, 362, 61, 291}:
                cv2.putText(
                    image,
                    str(idx),
                    (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (255, 255, 255),
                    1,
                )
