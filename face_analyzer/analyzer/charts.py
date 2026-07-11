"""Chart generation for facial analysis."""

from __future__ import annotations

import io

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from face_analyzer.analyzer.engine import AnalysisResult

matplotlib.use("Agg")


class ChartGenerator:
    """Generates matplotlib charts for dashboard and reports."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid") -> None:
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("ggplot")

    def symmetry_chart(self, result: AnalysisResult) -> bytes:
        """Bar chart of symmetry by region."""
        if not result.symmetry:
            return self._empty_chart("No symmetry data available")

        sym = result.symmetry
        labels = ["Overall", "Eyes", "Nose", "Mouth", "Jaw"]
        values = [
            sym.overall_index,
            sym.eyes_index,
            sym.nose_index,
            sym.mouth_index,
            sym.jaw_index,
        ]
        return self._bar_chart(labels, values, "Symmetry Indices", "Index (0-100)", "#4C78A8")

    def proportions_chart(self, result: AnalysisResult) -> bytes:
        """Facial thirds chart."""
        if not result.proportions:
            return self._empty_chart("No proportion data available")

        prop = result.proportions
        labels = ["Upper Third", "Middle Third", "Lower Third"]
        values = [
            prop.upper_third_ratio * 100,
            prop.middle_third_ratio * 100,
            prop.lower_third_ratio * 100,
        ]
        return self._bar_chart(labels, values, "Facial Thirds Proportion", "% of Height", "#F58518")

    def harmony_chart(self, result: AnalysisResult) -> bytes:
        """Harmony components chart."""
        if not result.harmony:
            return self._empty_chart("No harmony data available")

        har = result.harmony
        labels = ["Symmetry", "Proportions", "Alignment", "Distribution", "Harmony"]
        values = [
            har.symmetry_component,
            har.proportion_component,
            har.alignment_component,
            har.distribution_component,
            har.harmony_index,
        ]
        return self._bar_chart(labels, values, "Harmony Components", "Index (0-100)", "#54A24B")

    def indices_overview_chart(self, result: AnalysisResult) -> bytes:
        """Overview of main indices."""
        indices: dict[str, float] = {}
        if result.symmetry:
            indices["Overall Symmetry"] = result.symmetry.overall_index
        if result.harmony:
            indices["Harmony"] = result.harmony.harmony_index
        if result.eyes:
            indices["Ocular Symmetry"] = result.eyes.ocular_symmetry_index
        if result.nose:
            indices["Nasal Symmetry"] = result.nose.symmetry_index
        if result.lips:
            indices["Lip Symmetry"] = result.lips.symmetry_index
        if result.jaw:
            indices["Jaw Definition"] = result.jaw.definition_index

        if not indices:
            return self._empty_chart("No indices available")

        return self._bar_chart(
            list(indices.keys()),
            list(indices.values()),
            "Indices Overview",
            "Index (0-100)",
            "#B279A2",
        )

    def history_comparison_chart(
        self,
        history: list[dict[str, object]],
    ) -> bytes:
        """Compare index evolution over history."""
        if len(history) < 2:
            return self._empty_chart("Insufficient history for comparison")

        df = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(10, 5))

        legend_labels = {
            "symmetry_overall": "Overall Symmetry",
            "harmony_index": "Harmony Index",
        }
        for col, color in [
            ("symmetry_overall", "#4C78A8"),
            ("harmony_index", "#54A24B"),
        ]:
            if col in df.columns:
                ax.plot(
                    df.index,
                    df[col],
                    marker="o",
                    label=legend_labels.get(col, col),
                    color=color,
                )

        ax.set_title("Analysis Evolution")
        ax.set_xlabel("Analysis #")
        ax.set_ylabel("Index")
        ax.legend()
        ax.set_ylim(0, 100)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def reference_comparison_chart(self, result: AnalysisResult) -> bytes:
        """Compare scores against anthropometric references."""
        if not result.proportions or not result.proportions.reference_scores:
            return self._empty_chart("No references available")

        scores = result.proportions.reference_scores
        labels = [k.replace("_", " ").title() for k in scores.keys()]
        values = list(scores.values())
        return self._bar_chart(
            labels, values, "Reference Comparison", "Score (0-100)", "#E45756"
        )

    def advanced_regional_symmetry_chart(self, result: AnalysisResult) -> bytes:
        """Bar chart of advanced regional asymmetry scores."""
        if not result.advanced_geometry:
            return self._empty_chart("No advanced geometry data")

        adv = result.advanced_geometry
        labels = [k.replace("_", " ").title() for k in adv.regional_asymmetry]
        values = [v.score for v in adv.regional_asymmetry.values()]
        values.append(adv.regional_symmetry_overall)
        labels.append("Overall")
        return self._bar_chart(
            labels, values, "Regional Asymmetry Index", "Score (0-100)", "#E45756"
        )

    def advanced_radar_chart(self, result: AnalysisResult) -> bytes:
        """Radar chart of regional symmetry scores."""
        if not result.advanced_geometry:
            return self._empty_chart("No advanced geometry data")

        import numpy as np

        adv = result.advanced_geometry
        labels = list(adv.regional_asymmetry.keys())
        values = [adv.regional_asymmetry[k].score for k in labels]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles_plot = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles_plot, values_plot, "o-", linewidth=2, color="#4C78A8")
        ax.fill(angles_plot, values_plot, alpha=0.25, color="#4C78A8")
        ax.set_thetagrids(
            np.degrees(angles),
            [l.title() for l in labels],
        )
        ax.set_ylim(0, 100)
        ax.set_title("Regional Symmetry Radar", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def advanced_face_shape_chart(self, result: AnalysisResult) -> bytes:
        """Bar chart of face shape probabilities."""
        if not result.advanced_geometry or not result.advanced_geometry.face_shape_probabilities:
            return self._empty_chart("No face shape data")

        probs = result.advanced_geometry.face_shape_probabilities
        labels = [k.title() for k in probs]
        values = list(probs.values())
        return self._bar_chart(
            labels, values, "Face Shape Classification", "Probability (%)", "#B279A2"
        )

    def advanced_region_area_chart(self, result: AnalysisResult) -> bytes:
        """Bar chart of regional area percentages."""
        if not result.advanced_geometry:
            return self._empty_chart("No region area data")

        pct = result.advanced_geometry.region_area_percentages
        labels = [k.title() for k in pct]
        values = list(pct.values())
        return self._bar_chart(
            labels, values, "Regional Area Distribution", "% of Face", "#F58518"
        )

    def _bar_chart(
        self,
        labels: list[str],
        values: list[float],
        title: str,
        ylabel: str,
        color: str,
    ) -> bytes:
        """Create a generic bar chart."""
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, values, color=color, alpha=0.85)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(values) * 1.1, 100))
        plt.xticks(rotation=30, ha="right")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        return self._fig_to_bytes(fig)

    def _empty_chart(self, message: str) -> bytes:
        """Generate empty chart with message."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
        ax.axis("off")
        return self._fig_to_bytes(fig)

    def _fig_to_bytes(self, fig: plt.Figure) -> bytes:
        """Convert matplotlib figure to PNG bytes."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return buffer.read()
