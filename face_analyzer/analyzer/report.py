"""Geração de relatório PDF."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from face_analyzer.analyzer.charts import ChartGenerator
from face_analyzer.analyzer.engine import AnalysisResult
from face_analyzer.constants import METRIC_DESCRIPTIONS


class ReportGenerator:
    """Gera relatório PDF completo da análise facial."""

    def __init__(self, output_dir: str | Path = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._charts = ChartGenerator()
        self._styles = getSampleStyleSheet()
        self._title_style = ParagraphStyle(
            "CustomTitle",
            parent=self._styles["Heading1"],
            fontSize=22,
            spaceAfter=20,
            textColor=colors.HexColor("#1a1a2e"),
        )
        self._section_style = ParagraphStyle(
            "Section",
            parent=self._styles["Heading2"],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor("#16213e"),
        )

    def generate(
        self,
        result: AnalysisResult,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Gera relatório PDF e retorna caminho do arquivo.

        Args:
            result: Resultado da análise.
            filename: Nome do arquivo (gerado automaticamente se None).
        """
        if filename is None:
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"irimidlab_report_{timestamp}.pdf"

        filepath = self.output_dir / filename
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        story = self._build_story(result)
        doc.build(story)
        return filepath

    def generate_bytes(self, result: AnalysisResult) -> bytes:
        """Gera relatório PDF em memória."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        story = self._build_story(result)
        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    def _build_story(self, result: AnalysisResult) -> list:
        """Monta conteúdo do relatório."""
        story: list = []

        story.append(Paragraph("IrimidLab", self._title_style))
        story.append(Paragraph(
            "Geometric and Anthropometric Facial Analysis Report",
            self._styles["Heading3"],
        ))
        story.append(Paragraph(
            f"Analysis date: {result.timestamp.strftime('%m/%d/%Y %H:%M:%S')}",
            self._styles["Normal"],
        ))
        story.append(Spacer(1, 0.5 * cm))

        story.extend(self._validation_section(result))
        story.extend(self._images_section(result))
        story.extend(self._metrics_section(result))
        story.extend(self._charts_section(result))
        if result.advanced_geometry:
            story.extend(self._advanced_geometry_section(result))
        story.extend(self._descriptions_section())

        if result.harmony:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(
                f"<i>{result.harmony.disclaimer}</i>",
                self._styles["Normal"],
            ))

        return story

    def _validation_section(self, result: AnalysisResult) -> list:
        """Seção de validação da imagem."""
        elements: list = []
        elements.append(Paragraph("Image Quality", self._section_style))

        val = result.validation
        status = "Passed" if val.is_valid else "Failed"
        data = [
            ["Status", status],
            ["Resolution", f"{val.resolution[0]} x {val.resolution[1]}"],
            ["Brightness", f"{val.brightness:.1f}"],
            ["Sharpness", f"{val.sharpness:.1f}"],
            ["Head tilt", f"{val.head_tilt_degrees:.1f}°"],
        ]

        if val.issues:
            data.append(["Issues", "; ".join(val.issues)])
        if val.warnings:
            data.append(["Warnings", "; ".join(val.warnings)])

        table = Table(data, colWidths=[5 * cm, 12 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8e8e8")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.5 * cm))
        return elements

    def _images_section(self, result: AnalysisResult) -> list:
        """Seção com imagens original e anotada."""
        elements: list = []
        elements.append(Paragraph("Images", self._section_style))

        for label, image in [
            ("Original", result.original_image),
            ("Annotated", result.annotated_image),
        ]:
            if image is not None:
                img_bytes = self._cv2_to_reportlab_image(image, max_width=14 * cm)
                elements.append(Paragraph(label, self._styles["Heading4"]))
                elements.append(RLImage(io.BytesIO(img_bytes), width=14 * cm, height=10 * cm))
                elements.append(Spacer(1, 0.3 * cm))

        return elements

    def _metrics_section(self, result: AnalysisResult) -> list:
        """Seção com todas as métricas."""
        elements: list = []
        elements.append(Paragraph("Complete Metrics", self._section_style))

        flat = result.to_flat_dict()
        skip_keys = {"timestamp", "validation", "reference_scores", "left_right_differences",
                     "harmony_disclaimer", "profile_message", "profile_available"}

        data = [["Metric", "Value"]]
        for key, value in sorted(flat.items()):
            if key in skip_keys or isinstance(value, (dict, list)):
                continue
            data.append([key.replace("_", " ").title(), str(value)])

        if len(data) > 1:
            table = Table(data, colWidths=[8 * cm, 9 * cm])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ]))
            elements.append(table)

        elements.append(Spacer(1, 0.5 * cm))
        return elements

    def _charts_section(self, result: AnalysisResult) -> list:
        """Seção com gráficos."""
        elements: list = []
        elements.append(Paragraph("Charts", self._section_style))

        for title, chart_bytes in [
            ("Symmetry", self._charts.symmetry_chart(result)),
            ("Proportions", self._charts.proportions_chart(result)),
            ("Harmony", self._charts.harmony_chart(result)),
        ]:
            elements.append(Paragraph(title, self._styles["Heading4"]))
            elements.append(RLImage(io.BytesIO(chart_bytes), width=14 * cm, height=7 * cm))
            elements.append(Spacer(1, 0.3 * cm))

        return elements

    def _advanced_geometry_section(self, result: AnalysisResult) -> list:
        """Advanced geometry metrics and charts section."""
        elements: list = []
        if not result.advanced_geometry:
            return elements

        adv = result.advanced_geometry
        elements.append(Paragraph("Advanced Geometry", self._section_style))

        summary = [
            ["Regional Symmetry Overall", f"{adv.regional_symmetry_overall:.1f}"],
            ["Center Offset", f"{adv.center_offset_percent:.1f}%"],
            ["Axis Mean Error", f"{adv.axis_mean_error:.4f}"],
            ["Head Pitch", f"{adv.head_pitch:.1f}°"],
            ["Head Yaw", f"{adv.head_yaw:.1f}°"],
            ["Head Roll", f"{adv.head_roll:.1f}°"],
            ["Face Circularity", f"{adv.face_circularity:.4f}"],
            ["EAR Average", f"{adv.ear_avg:.4f}"],
            ["Nose Classification", adv.nose_classification],
        ]
        table = Table(summary, colWidths=[8 * cm, 9 * cm])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.3 * cm))

        for title, chart_bytes in [
            ("Regional Asymmetry", self._charts.advanced_regional_symmetry_chart(result)),
            ("Face Shape", self._charts.advanced_face_shape_chart(result)),
            ("Regional Radar", self._charts.advanced_radar_chart(result)),
        ]:
            elements.append(Paragraph(title, self._styles["Heading4"]))
            elements.append(RLImage(io.BytesIO(chart_bytes), width=14 * cm, height=7 * cm))
            elements.append(Spacer(1, 0.2 * cm))

        flat = adv.to_flat_dict()
        data = [["Metric", "Value"]]
        for key, value in sorted(flat.items()):
            if not isinstance(value, (dict, list)):
                data.append([key.replace("_", " ").title(), str(value)])

        if len(data) > 1:
            elements.append(Paragraph("Advanced Metrics Detail", self._styles["Heading4"]))
            detail_table = Table(data, colWidths=[8 * cm, 9 * cm])
            detail_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]))
            elements.append(detail_table)

        elements.append(Spacer(1, 0.5 * cm))
        return elements

    def _descriptions_section(self) -> list:
        """Seção com explicação das métricas."""
        elements: list = []
        elements.append(Paragraph("Metrics Glossary", self._section_style))

        for key, description in sorted(METRIC_DESCRIPTIONS.items()):
            elements.append(Paragraph(
                f"<b>{key.replace('_', ' ').title()}</b>: {description}",
                self._styles["Normal"],
            ))
            elements.append(Spacer(1, 0.1 * cm))

        return elements

    def _cv2_to_reportlab_image(
        self,
        image,
        max_width: float,
    ) -> bytes:
        """Converte imagem OpenCV BGR para bytes PNG."""
        import cv2

        height, width = image.shape[:2]
        scale = max_width / width
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(image, new_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        from PIL import Image

        pil_image = Image.fromarray(rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.read()
