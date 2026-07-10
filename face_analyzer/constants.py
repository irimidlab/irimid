"""Índices de landmarks do MediaPipe Face Mesh e referências antropométricas."""

from __future__ import annotations

from typing import Final

# --- Landmarks principais ---
NOSE_TIP: Final[int] = 1
NOSE_BRIDGE: Final[int] = 6
CHIN: Final[int] = 152
FOREHEAD: Final[int] = 10

# Olhos
LEFT_EYE_OUTER: Final[int] = 33
LEFT_EYE_INNER: Final[int] = 133
LEFT_EYE_TOP: Final[int] = 159
LEFT_EYE_BOTTOM: Final[int] = 145
LEFT_PUPIL: Final[int] = 468

RIGHT_EYE_OUTER: Final[int] = 263
RIGHT_EYE_INNER: Final[int] = 362
RIGHT_EYE_TOP: Final[int] = 386
RIGHT_EYE_BOTTOM: Final[int] = 374
RIGHT_PUPIL: Final[int] = 473

# Contorno dos olhos
LEFT_EYE_CONTOUR: Final[list[int]] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_CONTOUR: Final[list[int]] = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
]

# Boca
LEFT_MOUTH: Final[int] = 61
RIGHT_MOUTH: Final[int] = 291
UPPER_LIP_TOP: Final[int] = 0
LOWER_LIP_BOTTOM: Final[int] = 17
UPPER_LIP_CENTER: Final[int] = 13
LOWER_LIP_CENTER: Final[int] = 14

# Nariz
NOSE_LEFT: Final[int] = 129
NOSE_RIGHT: Final[int] = 358
NOSE_BOTTOM: Final[int] = 2

# Mandíbula e zigomas
LEFT_JAW: Final[int] = 172
RIGHT_JAW: Final[int] = 397
LEFT_CHEEK: Final[int] = 234
RIGHT_CHEEK: Final[int] = 454
LEFT_TEMPLE: Final[int] = 127
RIGHT_TEMPLE: Final[int] = 356

# Contorno facial
FACE_OVAL: Final[list[int]] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Regiões para simetria
SYMMETRY_PAIRS: Final[list[tuple[int, int]]] = [
    (33, 263), (133, 362), (61, 291), (172, 397), (234, 454),
    (127, 356), (129, 358), (145, 374), (159, 386), (21, 251),
    (54, 284), (58, 288), (93, 323), (132, 361), (136, 365),
    (148, 377), (149, 378), (150, 379), (152, 152), (176, 400),
]

# Referências antropométricas (faixas ideais em proporção)
ANTHROPOMETRIC_REFERENCES: Final[dict[str, dict[str, float]]] = {
    "upper_third_ratio": {"min": 0.28, "ideal": 0.33, "max": 0.38},
    "middle_third_ratio": {"min": 0.30, "ideal": 0.34, "max": 0.38},
    "lower_third_ratio": {"min": 0.30, "ideal": 0.33, "max": 0.38},
    "face_width_height": {"min": 0.70, "ideal": 0.80, "max": 0.90},
    "nose_width_face": {"min": 0.22, "ideal": 0.25, "max": 0.30},
    "mouth_width_face": {"min": 0.40, "ideal": 0.50, "max": 0.55},
    "interpupillary_face": {"min": 0.42, "ideal": 0.46, "max": 0.50},
    "eye_spacing": {"min": 0.28, "ideal": 0.32, "max": 0.36},
    "jaw_zygoma": {"min": 0.75, "ideal": 0.85, "max": 0.95},
    "upper_lower_lip": {"min": 0.40, "ideal": 0.45, "max": 0.55},
}

# Metric descriptions for reports
METRIC_DESCRIPTIONS: Final[dict[str, str]] = {
    "symmetry_overall": (
        "Overall symmetry index (0-100). Compares mirrored landmark positions "
        "relative to the facial midline."
    ),
    "symmetry_eyes": "Symmetry between the left and right eyes.",
    "symmetry_nose": "Nose symmetry relative to the midline.",
    "symmetry_mouth": "Mouth and lip corner symmetry.",
    "symmetry_jaw": "Jaw and lower contour symmetry.",
    "midline_deviation": "Angular midline deviation in degrees.",
    "upper_third": "Upper third (forehead) proportion relative to face height.",
    "middle_third": "Middle third (eyes/nose) proportion.",
    "lower_third": "Lower third (mouth/jaw) proportion.",
    "face_width_height": "Face width-to-height ratio.",
    "canthal_tilt_left": "Medial-to-lateral cant of the left eye (degrees).",
    "canthal_tilt_right": "Medial-to-lateral cant of the right eye (degrees).",
    "palpebral_opening": "Average vertical palpebral opening (normalized pixels).",
    "interpupillary_distance": "Distance between pupil centers.",
    "nose_length": "Nose length (bridge to tip).",
    "nose_width": "Nose width at the base.",
    "mouth_width": "Mouth width between lip corners.",
    "upper_lower_lip_ratio": "Upper-to-lower lip height ratio.",
    "jaw_width": "Mandibular width.",
    "jaw_angle": "Estimated mandibular angle in degrees.",
    "chin_height": "Chin height.",
    "chin_width": "Chin width.",
    "harmony_index": (
        "Composite geometric harmony index (0-100). Assessment based on "
        "symmetry, proportions, and alignments — not an absolute measure of attractiveness."
    ),
    "adv_regional_symmetry_overall": "Weighted overall regional asymmetry score (0-100).",
    "adv_geometric_center_offset": "Percent displacement of facial centroid from midline.",
    "adv_facial_axis_deviation": "Mean deviation of anatomical axis landmarks from midline.",
    "adv_head_pitch": "Head pitch angle in degrees (nodding).",
    "adv_head_yaw": "Head yaw angle in degrees (turning).",
    "adv_head_roll": "Head roll angle in degrees (tilting).",
    "adv_face_circularity": "Face circularity index (1.0 = circle).",
    "adv_eye_aspect_ratio": "Eye Aspect Ratio (EAR) for eye openness.",
    "adv_face_shape": "Probabilistic face shape classification.",
}

# Section labels for the dashboard
SECTION_LABELS: Final[dict[str, str]] = {
    "symmetry": "Symmetry",
    "proportions": "Proportions",
    "eyes": "Eyes",
    "nose": "Nose",
    "lips": "Lips",
    "jaw": "Jaw",
    "chin": "Chin",
    "profile": "Profile",
    "harmony": "Harmony",
    "advanced_geometry": "Advanced Geometry",
}

# Cores para visualização (BGR)
COLOR_MIDLINE: Final[tuple[int, int, int]] = (0, 255, 255)
COLOR_EYE_AXIS: Final[tuple[int, int, int]] = (255, 0, 255)
COLOR_THIRDS: Final[tuple[int, int, int]] = (0, 200, 255)
COLOR_PROPORTION: Final[tuple[int, int, int]] = (255, 200, 0)
COLOR_ANGLE: Final[tuple[int, int, int]] = (0, 255, 0)
COLOR_LANDMARK: Final[tuple[int, int, int]] = (0, 150, 255)
COLOR_SYMMETRY: Final[tuple[int, int, int]] = (200, 100, 255)

# --- Advanced geometry landmarks ---
GLABELLA: Final[int] = 10
FILTRUM: Final[int] = 2
LEFT_BROW: Final[list[int]] = [70, 63, 105, 66, 107]
RIGHT_BROW: Final[list[int]] = [336, 296, 334, 293, 300]
LIPS_OUTER: Final[list[int]] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95,
]
NOSE_REGION: Final[list[int]] = [6, 197, 195, 5, 4, 1, 19, 94, 2, 129, 358, 327, 326]
FOREHEAD_REGION: Final[list[int]] = [10, 151, 9, 8, 168, 107, 55, 65, 52, 53, 46, 124, 35, 143]
CHIN_REGION: Final[list[int]] = [152, 175, 199, 200, 18, 313, 406, 335, 148, 176, 149]
JAW_REGION: Final[list[int]] = [172, 136, 150, 149, 176, 148, 397, 365, 379, 378, 400, 377, 288, 361]

ANATOMICAL_AXIS_INDICES: Final[list[int]] = [GLABELLA, NOSE_TIP, FILTRUM, CHIN]

ADVANCED_REGION_PAIRS: Final[dict[str, list[tuple[int, int]]]] = {
    "forehead": [(21, 251), (54, 284), (67, 109), (103, 332), (127, 356), (162, 389)],
    "eyebrows": [(70, 300), (63, 293), (105, 334), (66, 296), (107, 336)],
    "eyes": [(33, 263), (133, 362), (145, 374), (159, 386), (133, 362)],
    "nose": [(129, 358), (219, 439), (49, 279)],
    "mouth": [(61, 291), (78, 308), (95, 325), (88, 318), (178, 402), (87, 317)],
    "jaw": [(172, 397), (58, 288), (132, 361), (93, 323), (172, 397)],
    "chin": [(176, 400), (148, 377), (149, 378), (150, 379)],
}

REGION_AREA_INDICES: Final[dict[str, list[int]]] = {
    "forehead": FOREHEAD_REGION,
    "eyes": LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR,
    "nose": NOSE_REGION,
    "mouth": LIPS_OUTER,
    "chin": CHIN_REGION,
}

ADVANCED_REGION_WEIGHTS: Final[dict[str, float]] = {
    "forehead": 0.12,
    "eyebrows": 0.10,
    "eyes": 0.18,
    "nose": 0.15,
    "mouth": 0.15,
    "jaw": 0.15,
    "chin": 0.15,
}

COLOR_ADV_CENTROID: Final[tuple[int, int, int]] = (0, 0, 255)
COLOR_ADV_AXIS: Final[tuple[int, int, int]] = (255, 100, 0)
COLOR_ADV_REGION: Final[tuple[int, int, int]] = (100, 200, 100)
