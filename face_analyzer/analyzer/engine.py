"""Orquestrador principal da análise facial."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from face_analyzer.analyzer.advanced_geometry import (
    AdvancedGeometryAnalyzer,
    AdvancedGeometryMetrics,
)
from face_analyzer.analyzer.chin import ChinAnalyzer, ChinMetrics
from face_analyzer.analyzer.eyes import EyeAnalyzer, EyeMetrics
from face_analyzer.analyzer.harmony import HarmonyAnalyzer, HarmonyMetrics
from face_analyzer.analyzer.jaw import JawAnalyzer, JawMetrics
from face_analyzer.analyzer.landmarks import FaceLandmarkDetector, FaceLandmarks
from face_analyzer.analyzer.lips import LipAnalyzer, LipMetrics
from face_analyzer.analyzer.nose import NoseAnalyzer, NoseMetrics
from face_analyzer.analyzer.profile import ProfileAnalyzer, ProfileMetrics
from face_analyzer.analyzer.proportions import ProportionAnalyzer, ProportionMetrics
from face_analyzer.analyzer.symmetry import SymmetryAnalyzer, SymmetryMetrics
from face_analyzer.analyzer.validation import ImageValidator, ValidationResult
from face_analyzer.utils.drawing import DrawingLayers, FaceAnnotator


@dataclass
class AnalysisResult:
    """Resultado completo da análise facial."""

    timestamp: datetime
    validation: ValidationResult
    landmarks: Optional[FaceLandmarks]
    symmetry: Optional[SymmetryMetrics] = None
    proportions: Optional[ProportionMetrics] = None
    eyes: Optional[EyeMetrics] = None
    nose: Optional[NoseMetrics] = None
    lips: Optional[LipMetrics] = None
    jaw: Optional[JawMetrics] = None
    chin: Optional[ChinMetrics] = None
    profile: Optional[ProfileMetrics] = None
    harmony: Optional[HarmonyMetrics] = None
    advanced_geometry: Optional[AdvancedGeometryMetrics] = None
    annotated_image: Optional[NDArray[np.uint8]] = None
    original_image: Optional[NDArray[np.uint8]] = None
    error_message: str = ""

    def to_flat_dict(self) -> dict[str, object]:
        """Retorna todas as métricas em dicionário plano."""
        result: dict[str, object] = {
            "timestamp": self.timestamp.isoformat(),
            "validation": self.validation.to_dict(),
        }
        if self.symmetry:
            result.update(self.symmetry.to_dict())
        if self.proportions:
            result.update(self.proportions.to_dict())
        if self.eyes:
            result.update(self.eyes.to_dict())
        if self.nose:
            result.update(self.nose.to_dict())
        if self.lips:
            result.update(self.lips.to_dict())
        if self.jaw:
            result.update(self.jaw.to_dict())
        if self.chin:
            result.update(self.chin.to_dict())
        if self.profile:
            result.update(self.profile.to_dict())
        if self.harmony:
            result.update(self.harmony.to_dict())
        if self.advanced_geometry:
            result.update(self.advanced_geometry.to_flat_dict())
        return result

    @property
    def is_successful(self) -> bool:
        """Indica se a análise foi concluída com sucesso."""
        return (
            self.validation.is_valid
            and self.landmarks is not None
            and self.symmetry is not None
            and not self.error_message
        )


class FaceAnalyzer:
    """Orquestra detecção, validação e análise facial completa."""

    def __init__(self) -> None:
        self._detector = FaceLandmarkDetector()
        self._validator = ImageValidator()
        self._symmetry = SymmetryAnalyzer()
        self._proportions = ProportionAnalyzer()
        self._eyes = EyeAnalyzer()
        self._nose = NoseAnalyzer()
        self._lips = LipAnalyzer()
        self._jaw = JawAnalyzer()
        self._chin = ChinAnalyzer()
        self._profile = ProfileAnalyzer(detector=self._detector)
        self._harmony = HarmonyAnalyzer()
        self._advanced_geometry = AdvancedGeometryAnalyzer()
        self._annotator = FaceAnnotator()

    def analyze(
        self,
        image: NDArray[np.uint8],
        profile_image: NDArray[np.uint8] | None = None,
        drawing_layers: DrawingLayers | None = None,
        force_analysis: bool = False,
    ) -> AnalysisResult:
        """
        Executa análise facial completa.

        Args:
            image: Imagem frontal BGR.
            profile_image: Imagem lateral opcional.
            drawing_layers: Camadas de visualização.
            force_analysis: Se True, analisa mesmo com avisos de validação.

        Returns:
            AnalysisResult com todas as métricas e visualizações.
        """
        timestamp = datetime.now()
        original = image.copy()

        try:
            landmarks = self._detector.detect(image)
            validation = self._validator.validate(image, landmarks)

            if not validation.is_valid and not force_analysis:
                return AnalysisResult(
                    timestamp=timestamp,
                    validation=validation,
                    landmarks=landmarks,
                    original_image=original,
                    error_message="; ".join(validation.issues),
                )

            if landmarks is None:
                return AnalysisResult(
                    timestamp=timestamp,
                    validation=validation,
                    landmarks=None,
                    original_image=original,
                    error_message="No face detected.",
                )

            symmetry = self._symmetry.analyze(landmarks)
            proportions = self._proportions.analyze(landmarks)
            eyes = self._eyes.analyze(landmarks)
            nose = self._nose.analyze(landmarks)
            lips = self._lips.analyze(landmarks)
            jaw = self._jaw.analyze(landmarks)
            chin = self._chin.analyze(landmarks)
            profile = self._profile.analyze(profile_image, landmarks)
            harmony = self._harmony.analyze(
                symmetry, proportions, eyes, nose, lips, jaw, chin
            )
            advanced_geometry = self._advanced_geometry.analyze(landmarks)

            if drawing_layers:
                self._annotator.layers = drawing_layers

            metrics_dict = {
                **symmetry.to_dict(),
                **eyes.to_dict(),
            }
            annotated = self._annotator.annotate(
                image, landmarks.points, metrics_dict, advanced_geometry
            )

            return AnalysisResult(
                timestamp=timestamp,
                validation=validation,
                landmarks=landmarks,
                symmetry=symmetry,
                proportions=proportions,
                eyes=eyes,
                nose=nose,
                lips=lips,
                jaw=jaw,
                chin=chin,
                profile=profile,
                harmony=harmony,
                advanced_geometry=advanced_geometry,
                annotated_image=annotated,
                original_image=original,
            )

        except Exception as exc:
            return AnalysisResult(
                timestamp=timestamp,
                validation=ValidationResult(is_valid=False, issues=[str(exc)]),
                landmarks=None,
                original_image=original,
                error_message=str(exc),
            )

    def close(self) -> None:
        """Libera recursos."""
        self._detector.close()

    def __enter__(self) -> "FaceAnalyzer":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
