"""Índice composto de harmonia facial geométrica."""

from __future__ import annotations

from dataclasses import dataclass

from face_analyzer.analyzer.chin import ChinMetrics
from face_analyzer.analyzer.eyes import EyeMetrics
from face_analyzer.analyzer.jaw import JawMetrics
from face_analyzer.analyzer.lips import LipMetrics
from face_analyzer.analyzer.nose import NoseMetrics
from face_analyzer.analyzer.proportions import ProportionMetrics
from face_analyzer.analyzer.symmetry import SymmetryMetrics


@dataclass
class HarmonyMetrics:
    """Índice de harmonia facial composto."""

    harmony_index: float = 0.0
    symmetry_component: float = 0.0
    proportion_component: float = 0.0
    alignment_component: float = 0.0
    distribution_component: float = 0.0
    disclaimer: str = (
        "This index represents a geometric assessment based on the calculated "
        "metrics only. It is not an absolute measure of attractiveness."
    )

    def to_dict(self) -> dict[str, object]:
        """Converte métricas para dicionário."""
        return {
            "harmony_index": round(self.harmony_index, 2),
            "harmony_symmetry": round(self.symmetry_component, 2),
            "harmony_proportions": round(self.proportion_component, 2),
            "harmony_alignment": round(self.alignment_component, 2),
            "harmony_distribution": round(self.distribution_component, 2),
            "harmony_disclaimer": self.disclaimer,
        }


class HarmonyAnalyzer:
    """Calcula índice composto de harmonia facial."""

    WEIGHTS: dict[str, float] = {
        "symmetry": 0.35,
        "proportions": 0.30,
        "alignment": 0.20,
        "distribution": 0.15,
    }

    def analyze(
        self,
        symmetry: SymmetryMetrics,
        proportions: ProportionMetrics,
        eyes: EyeMetrics,
        nose: NoseMetrics,
        lips: LipMetrics,
        jaw: JawMetrics,
        chin: ChinMetrics,
    ) -> HarmonyMetrics:
        """Calcula índice de harmonia a partir de todas as métricas."""
        symmetry_component = symmetry.overall_index

        ref_scores = list(proportions.reference_scores.values())
        proportion_component = (
            sum(ref_scores) / len(ref_scores) if ref_scores else 50.0
        )

        alignment_scores = [
            100.0 - min(nose.alignment_deviation * 200, 50),
            100.0 - min(lips.alignment_deviation * 200, 50),
            100.0 - min(chin.alignment_deviation * 200, 50),
            eyes.ocular_symmetry_index,
        ]
        alignment_component = sum(alignment_scores) / len(alignment_scores)

        third_balance = 100.0 - abs(proportions.upper_third_ratio - 0.33) * 300
        third_balance -= abs(proportions.middle_third_ratio - 0.34) * 300
        third_balance -= abs(proportions.lower_third_ratio - 0.33) * 300
        distribution_component = max(0.0, min(100.0, third_balance / 3 + 50))

        harmony_index = (
            symmetry_component * self.WEIGHTS["symmetry"]
            + proportion_component * self.WEIGHTS["proportions"]
            + alignment_component * self.WEIGHTS["alignment"]
            + distribution_component * self.WEIGHTS["distribution"]
        )

        return HarmonyMetrics(
            harmony_index=harmony_index,
            symmetry_component=symmetry_component,
            proportion_component=proportion_component,
            alignment_component=alignment_component,
            distribution_component=distribution_component,
        )
