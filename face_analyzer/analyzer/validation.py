"""Validação de qualidade da imagem para análise facial."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from face_analyzer.analyzer.landmarks import FaceLandmarks
from face_analyzer.constants import (
    CHIN,
    FOREHEAD,
    LEFT_EYE_BOTTOM,
    LEFT_EYE_TOP,
    NOSE_TIP,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_TOP,
)
from face_analyzer.utils.geometry import line_angle
from face_analyzer.utils.image import assess_brightness, assess_sharpness, get_image_dimensions


@dataclass
class ValidationResult:
    """Resultado da validação de imagem."""

    is_valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    brightness: float = 0.0
    sharpness: float = 0.0
    resolution: tuple[int, int] = (0, 0)
    head_tilt_degrees: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Converte resultado para dicionário."""
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "warnings": self.warnings,
            "brightness": self.brightness,
            "sharpness": self.sharpness,
            "resolution": self.resolution,
            "head_tilt_degrees": self.head_tilt_degrees,
        }


class ImageValidator:
    """Valida qualidade e adequação da imagem para análise."""

    MIN_RESOLUTION: int = 1000
    MIN_BRIGHTNESS: float = 40.0
    MAX_BRIGHTNESS: float = 230.0
    MIN_SHARPNESS: float = 50.0
    MAX_HEAD_TILT: float = 15.0
    MIN_FACE_MARGIN_RATIO: float = 0.05

    def validate(
        self,
        image: NDArray[np.uint8],
        landmarks: FaceLandmarks | None = None,
    ) -> ValidationResult:
        """
        Valida imagem antes da análise.

        Verifica resolução, iluminação, nitidez, visibilidade facial,
        inclinação da cabeça e obstruções parciais.
        """
        issues: list[str] = []
        warnings: list[str] = []

        width, height = get_image_dimensions(image)
        brightness = assess_brightness(image)
        sharpness = assess_sharpness(image)
        head_tilt = 0.0

        if max(width, height) < self.MIN_RESOLUTION:
            issues.append(
                f"Insufficient resolution ({width}x{height}). "
                f"Recommended minimum: {self.MIN_RESOLUTION}px on the longest side."
            )

        if brightness < self.MIN_BRIGHTNESS:
            issues.append(
                f"Insufficient lighting (average brightness: {brightness:.1f}). "
                "Use a better-lit environment."
            )
        elif brightness > self.MAX_BRIGHTNESS:
            issues.append(
                f"Overexposed image (average brightness: {brightness:.1f}). "
                "Reduce direct lighting."
            )

        if sharpness < self.MIN_SHARPNESS:
            warnings.append(
                f"Low sharpness ({sharpness:.1f}). Analysis may be less accurate."
            )

        if landmarks is None:
            issues.append("No face detected in the image.")
            return ValidationResult(
                is_valid=False,
                issues=issues,
                warnings=warnings,
                brightness=brightness,
                sharpness=sharpness,
                resolution=(width, height),
            )

        face_issues = self._validate_face_visibility(image, landmarks)
        issues.extend(face_issues)

        head_tilt = self._compute_head_tilt(landmarks)
        if abs(head_tilt) > self.MAX_HEAD_TILT:
            issues.append(
                f"Head tilted ({head_tilt:.1f}°). "
                f"Maximum acceptable: ±{self.MAX_HEAD_TILT}°."
            )

        occlusion_warnings = self._check_occlusions(landmarks)
        warnings.extend(occlusion_warnings)

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            brightness=brightness,
            sharpness=sharpness,
            resolution=(width, height),
            head_tilt_degrees=head_tilt,
        )

    def _validate_face_visibility(
        self,
        image: NDArray[np.uint8],
        landmarks: FaceLandmarks,
    ) -> list[str]:
        """Verifica se o rosto está totalmente visível na imagem."""
        issues: list[str] = []
        height, width = image.shape[:2]
        margin_x = width * self.MIN_FACE_MARGIN_RATIO
        margin_y = height * self.MIN_FACE_MARGIN_RATIO

        key_indices = [FOREHEAD, CHIN, 33, 263, 61, 291]
        for idx in key_indices:
            x, y = landmarks.points[idx]
            if x < margin_x or x > width - margin_x:
                issues.append("Face partially cropped on the sides.")
                break
            if y < margin_y or y > height - margin_y:
                issues.append("Face partially cropped at the top or bottom.")
                break

        face_height = landmarks.points[CHIN][1] - landmarks.points[FOREHEAD][1]
        if face_height < height * 0.15:
            issues.append(
                "Face too small in the image. Move closer to the camera."
            )

        return issues

    def _compute_head_tilt(self, landmarks: FaceLandmarks) -> float:
        """Calcula inclinação da cabeça via eixo interocular."""
        left_eye = (landmarks.points[LEFT_EYE_TOP] + landmarks.points[LEFT_EYE_BOTTOM]) / 2
        right_eye = (landmarks.points[RIGHT_EYE_TOP] + landmarks.points[RIGHT_EYE_BOTTOM]) / 2
        return line_angle(left_eye, right_eye)

    def _check_occlusions(self, landmarks: FaceLandmarks) -> list[str]:
        """Verifica possíveis obstruções em regiões críticas."""
        warnings: list[str] = []
        points = landmarks.points

        eye_left_span = np.linalg.norm(points[33] - points[133])
        eye_right_span = np.linalg.norm(points[263] - points[362])
        if eye_left_span < landmarks.face_scale * 0.08 or eye_right_span < landmarks.face_scale * 0.08:
            warnings.append(
                "Possible eye obstruction (ocular region appears too small)."
            )

        nose_span = np.linalg.norm(points[129] - points[358])
        if nose_span < landmarks.face_scale * 0.03:
            warnings.append("Possible nose obstruction.")

        mouth_span = np.linalg.norm(points[61] - points[291])
        if mouth_span < landmarks.face_scale * 0.05:
            warnings.append("Possible mouth obstruction.")

        return warnings
