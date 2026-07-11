"""Script de teste dos módulos do IrimidLab."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_geometry() -> None:
    """Testa funções geométricas."""
    from face_analyzer.utils.geometry import (
        angle_between_points,
        euclidean_distance,
        midpoint,
        normalize_value,
        score_against_reference,
        symmetry_score,
    )

    p1 = np.array([0.0, 0.0])
    p2 = np.array([3.0, 4.0])
    assert euclidean_distance(p1, p2) == 5.0

    mid = midpoint(p1, p2)
    assert np.allclose(mid, [1.5, 2.0])

    p3 = np.array([3.0, 0.0])
    angle = angle_between_points(p1, p2, p3)
    assert 0 < angle < 180

    assert 0 <= normalize_value(50, 0, 100) <= 100
    assert score_against_reference(0.33, 0.28, 0.33, 0.38) > 85
    assert 0 <= symmetry_score(p1, p1, 10.0) <= 100
    print("[OK] geometry.py")


def test_image_utils() -> None:
    """Testa utilitários de imagem."""
    from face_analyzer.utils.image import assess_brightness, assess_sharpness

    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image[400:600, 400:600] = 200
    assert assess_brightness(image) > 0
    assert assess_sharpness(image) >= 0
    print("[OK] image.py")


def test_landmarks_structure() -> None:
    """Testa estrutura de landmarks."""
    from face_analyzer.analyzer.landmarks import FaceLandmarks

    points = np.random.rand(478, 2) * 1000
    landmarks = FaceLandmarks(points=points, image_width=1000, image_height=1000)
    assert landmarks.count == 478
    assert landmarks.face_scale > 0
    assert len(landmarks.midline) == 2
    print("[OK] landmarks.py")


def test_symmetry() -> None:
    """Testa análise de simetria."""
    from face_analyzer.analyzer.landmarks import FaceLandmarks
    from face_analyzer.analyzer.symmetry import SymmetryAnalyzer

    points = np.zeros((478, 2))
    for i in range(478):
        points[i] = [500 + (i % 50), 200 + (i % 30)]

    points[33] = [400, 300]
    points[263] = [600, 300]
    points[133] = [450, 300]
    points[362] = [550, 300]
    points[1] = [500, 400]
    points[10] = [500, 100]
    points[152] = [500, 700]

    landmarks = FaceLandmarks(points=points, image_width=1000, image_height=1000)
    analyzer = SymmetryAnalyzer()
    metrics = analyzer.analyze(landmarks)
    assert 0 <= metrics.overall_index <= 100
    assert metrics.to_dict()["symmetry_overall"] >= 0
    print("[OK] symmetry.py")


def test_proportions() -> None:
    """Testa análise de proporções."""
    from face_analyzer.analyzer.landmarks import FaceLandmarks
    from face_analyzer.analyzer.proportions import ProportionAnalyzer

    points = np.zeros((478, 2))
    points[10] = [500, 100]
    points[152] = [500, 700]
    points[1] = [500, 400]
    points[234] = [300, 350]
    points[454] = [700, 350]
    points[172] = [350, 600]
    points[397] = [650, 600]
    points[129] = [470, 420]
    points[358] = [530, 420]
    points[61] = [420, 500]
    points[291] = [580, 500]
    points[133] = [450, 280]
    points[362] = [550, 280]
    points[159] = [430, 270]
    points[386] = [570, 270]
    points[14] = [500, 520]

    landmarks = FaceLandmarks(points=points, image_width=1000, image_height=1000)
    analyzer = ProportionAnalyzer()
    metrics = analyzer.analyze(landmarks)
    assert metrics.face_height > 0
    assert abs(metrics.upper_third_ratio + metrics.middle_third_ratio + metrics.lower_third_ratio - 1.0) < 0.01
    print("[OK] proportions.py")


def test_harmony() -> None:
    """Testa índice de harmonia."""
    from face_analyzer.analyzer.chin import ChinMetrics
    from face_analyzer.analyzer.eyes import EyeMetrics
    from face_analyzer.analyzer.harmony import HarmonyAnalyzer
    from face_analyzer.analyzer.jaw import JawMetrics
    from face_analyzer.analyzer.lips import LipMetrics
    from face_analyzer.analyzer.nose import NoseMetrics
    from face_analyzer.analyzer.proportions import ProportionMetrics
    from face_analyzer.analyzer.symmetry import SymmetryMetrics

    analyzer = HarmonyAnalyzer()
    result = analyzer.analyze(
        SymmetryMetrics(overall_index=80),
        ProportionMetrics(reference_scores={"a": 75}),
        EyeMetrics(ocular_symmetry_index=85),
        NoseMetrics(alignment_deviation=0.01),
        LipMetrics(alignment_deviation=0.01),
        JawMetrics(),
        ChinMetrics(alignment_deviation=0.01),
    )
    assert 0 <= result.harmony_index <= 100
    print("[OK] harmony.py")


def test_validation() -> None:
    """Testa validação de imagem."""
    from face_analyzer.analyzer.validation import ImageValidator

    image = np.zeros((500, 500, 3), dtype=np.uint8)
    validator = ImageValidator()
    result = validator.validate(image, None)
    assert not result.is_valid
    assert len(result.issues) > 0
    print("[OK] validation.py")


def test_drawing_layers() -> None:
    """Testa anotação de imagem."""
    from face_analyzer.utils.drawing import DrawingLayers, FaceAnnotator

    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    points = np.zeros((478, 2))
    points[10] = [500, 100]
    points[152] = [500, 700]
    points[33] = [400, 300]
    points[263] = [600, 300]

    annotator = FaceAnnotator(layers=DrawingLayers())
    annotated = annotator.annotate(image, points)
    assert annotated.shape == image.shape
    print("[OK] drawing.py")


def test_export() -> None:
    """Testa exportação de dados."""
    from datetime import datetime

    from face_analyzer.analyzer.engine import AnalysisResult
    from face_analyzer.analyzer.validation import ValidationResult
    from face_analyzer.utils.export import DataExporter

    result = AnalysisResult(
        timestamp=datetime.now(),
        validation=ValidationResult(is_valid=True),
        landmarks=None,
    )
    json_str = DataExporter.to_json(result)
    assert "timestamp" in json_str
    csv_str = DataExporter.to_csv(result)
    assert "metric" in csv_str
    print("[OK] export.py")


def test_advanced_geometry() -> None:
    """Test advanced geometry analyzer."""
    from face_analyzer.analyzer.advanced_geometry import AdvancedGeometryAnalyzer
    from face_analyzer.analyzer.landmarks import FaceLandmarks
    from face_analyzer.constants import FACE_OVAL

    points = np.zeros((478, 2))
    points[10] = [500, 100]
    points[152] = [500, 700]
    points[1] = [500, 400]
    points[2] = [500, 450]
    points[6] = [500, 300]
    points[33] = [400, 300]
    points[263] = [600, 300]
    points[133] = [450, 300]
    points[362] = [550, 300]
    points[145] = [400, 310]
    points[374] = [600, 310]
    points[159] = [395, 290]
    points[386] = [605, 290]
    points[234] = [300, 350]
    points[454] = [700, 350]
    points[172] = [350, 600]
    points[397] = [650, 600]
    points[129] = [470, 420]
    points[358] = [530, 420]
    points[61] = [420, 500]
    points[291] = [580, 500]
    points[14] = [500, 520]
    points[70] = [400, 250]
    points[300] = [600, 250]
    for idx in FACE_OVAL:
        if idx < len(points) and np.allclose(points[idx], 0):
            points[idx] = [500 + (idx % 20) - 10, 200 + (idx % 30) * 15]

    landmarks = FaceLandmarks(
        points=points, image_width=1000, image_height=1000,
        z_coords=np.zeros(478),
    )
    analyzer = AdvancedGeometryAnalyzer()
    result = analyzer.analyze(landmarks)
    assert 0 <= result.regional_symmetry_overall <= 100
    assert len(result.regional_asymmetry) == 7
    assert len(result.face_shape_probabilities) == 7
    assert result.ear_avg >= 0
    assert len(result.to_flat_dict()) > 50
    print("[OK] advanced_geometry.py")


def main() -> None:
    """Executa todos os testes."""
    print("IrimidLab - Testes de Módulos\n" + "=" * 40)
    tests = [
        test_geometry,
        test_image_utils,
        test_landmarks_structure,
        test_symmetry,
        test_proportions,
        test_harmony,
        test_validation,
        test_drawing_layers,
        test_export,
        test_advanced_geometry,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
            failed += 1

    print("=" * 40)
    if failed == 0:
        print("Todos os testes passaram!")
    else:
        print(f"{failed} teste(s) falharam.")
        sys.exit(1)


if __name__ == "__main__":
    main()
