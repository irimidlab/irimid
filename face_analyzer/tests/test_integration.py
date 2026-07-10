"""Teste de integração com detecção real de face."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from face_analyzer.analyzer.engine import FaceAnalyzer
from face_analyzer.analyzer.report import ReportGenerator
from face_analyzer.utils.drawing import DrawingLayers


def create_test_face_image() -> np.ndarray:
    """Cria imagem sintética com rosto estilo retrato para testes básicos."""
    image = np.full((1200, 1000, 3), 220, dtype=np.uint8)
    center = (500, 550)

    cv2.ellipse(image, center, (220, 280), 0, 0, 360, (200, 180, 170), -1)
    cv2.circle(image, (420, 480), 25, (80, 80, 80), -1)
    cv2.circle(image, (580, 480), 25, (80, 80, 80), -1)
    cv2.ellipse(image, (500, 580), (40, 25), 0, 0, 360, (170, 150, 140), -1)
    cv2.ellipse(image, (500, 680), (70, 30), 0, 0, 180, (150, 100, 100), 2)

    return image


def main() -> None:
    """Executa teste de integração end-to-end."""
    print("IrimidLab - Teste de Integracao")
    print("=" * 40)

    analyzer = FaceAnalyzer()
    try:
        image = create_test_face_image()
        result = analyzer.analyze(
            image,
            drawing_layers=DrawingLayers(),
            force_analysis=False,
        )

        print(f"Validacao: {'OK' if result.validation.is_valid else 'FALHOU'}")
        if result.validation.issues:
            for issue in result.validation.issues:
                print(f"  - {issue}")

        if result.landmarks:
            print(f"Landmarks detectados: {result.landmarks.count}")
        else:
            print("Landmarks: nenhum (esperado em imagem sintetica)")

        sample_path = Path(__file__).parent / "sample_face.jpg"
        if sample_path.exists():
            real_image = cv2.imread(str(sample_path))
            real_result = analyzer.analyze(real_image, force_analysis=True)
            if real_result.is_successful:
                print("Analise com foto real: OK")
                print(f"  Simetria: {real_result.symmetry.overall_index:.1f}")
                print(f"  Harmonia: {real_result.harmony.harmony_index:.1f}")

                report_gen = ReportGenerator()
                pdf_path = report_gen.generate(real_result)
                print(f"  PDF gerado: {pdf_path}")
            else:
                print(f"Analise com foto real falhou: {real_result.error_message}")
        else:
            print("sample_face.jpg nao encontrado - pulando teste com foto real")

        print("=" * 40)
        print("Integracao concluida.")
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
