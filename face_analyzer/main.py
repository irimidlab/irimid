"""IrimidLab - Streamlit dashboard for facial analysis."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from face_analyzer import __app_name__, __version__
from face_analyzer.analyzer.charts import ChartGenerator
from face_analyzer.analyzer.engine import FaceAnalyzer
from face_analyzer.analyzer.report import ReportGenerator
from face_analyzer.constants import METRIC_DESCRIPTIONS, SECTION_LABELS
from face_analyzer.utils.drawing import DrawingLayers
from face_analyzer.utils.export import DataExporter, HistoryManager
from face_analyzer.utils.image import bgr_to_rgb, encode_image_png, load_image

st.set_page_config(
    page_title=f"{__app_name__} - Facial Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .issue-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize session state."""
    defaults = {
        "result": None,
        "analyzer": None,
        "history": HistoryManager(),
        "charts": ChartGenerator(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_analyzer() -> FaceAnalyzer:
    """Return singleton analyzer instance."""
    if st.session_state.analyzer is None:
        st.session_state.analyzer = FaceAnalyzer()
    return st.session_state.analyzer


def format_metric_name(key: str) -> str:
    """Format metric key for display."""
    return key.replace("_", " ").title()


def render_sidebar() -> DrawingLayers:
    """Render sidebar controls."""
    st.sidebar.markdown(f"## {__app_name__}")
    st.sidebar.markdown(f"v{__version__}")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Image Upload")
    frontal_file = st.sidebar.file_uploader(
        "Frontal Photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Front-facing photo with a visible face, minimum 1000px.",
    )
    profile_file = st.sidebar.file_uploader(
        "Profile Photo (optional)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Visualization Layers")
    layers = DrawingLayers(
        landmarks=st.sidebar.checkbox("Landmarks", value=True),
        landmark_numbers=st.sidebar.checkbox("Landmark Numbers", value=False),
        midline=st.sidebar.checkbox("Midline", value=True),
        eye_axes=st.sidebar.checkbox("Eye Axes", value=True),
        facial_thirds=st.sidebar.checkbox("Facial Thirds", value=True),
        proportions=st.sidebar.checkbox("Proportion Lines", value=True),
        angles=st.sidebar.checkbox("Angles", value=True),
        distances=st.sidebar.checkbox("Distances", value=True),
        symmetry_lines=st.sidebar.checkbox("Symmetry Lines", value=False),
        advanced_geometry=st.sidebar.checkbox("Advanced Geometry", value=False),
    )

    st.sidebar.markdown("---")
    force = st.sidebar.checkbox(
        "Force analysis (skip validation)",
        value=False,
        help="Run analysis even when quality issues are detected.",
    )

    if st.sidebar.button("Analyze", type="primary", use_container_width=True):
        if frontal_file is None:
            st.sidebar.error("Please upload a frontal photo.")
        else:
            run_analysis(frontal_file, profile_file, layers, force)

    if st.session_state.result and st.session_state.result.is_successful:
        if st.sidebar.button("Save to History", use_container_width=True):
            st.session_state.history.add(st.session_state.result)
            st.sidebar.success("Analysis saved to history!")

    return layers


def run_analysis(frontal_file, profile_file, layers: DrawingLayers, force: bool) -> None:
    """Run facial analysis."""
    with st.spinner("Analyzing image..."):
        try:
            frontal_image = load_image(frontal_file.read())
            profile_image = None
            if profile_file:
                profile_image = load_image(profile_file.read())

            analyzer = get_analyzer()
            result = analyzer.analyze(
                frontal_image,
                profile_image=profile_image,
                drawing_layers=layers,
                force_analysis=force,
            )
            st.session_state.result = result

            if result.is_successful:
                st.sidebar.success("Analysis complete!")
            else:
                st.sidebar.warning("Analysis completed with warnings.")

        except Exception as exc:
            st.sidebar.error(f"Error: {exc}")


def render_header() -> None:
    """Render application header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">IrimidLab</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Geometric and Anthropometric Facial Analysis</p>',
            unsafe_allow_html=True,
        )
    with col2:
        if st.session_state.result and st.session_state.result.harmony:
            har = st.session_state.result.harmony.harmony_index
            st.metric("Harmony Index", f"{har:.1f}")


def render_validation_status() -> None:
    """Display validation status."""
    result = st.session_state.result
    if result is None:
        return

    val = result.validation
    if val.is_valid:
        st.markdown(
            '<div class="success-box">Image validated successfully.</div>',
            unsafe_allow_html=True,
        )
    else:
        for issue in val.issues:
            st.markdown(
                f'<div class="error-box">{issue}</div>',
                unsafe_allow_html=True,
            )

    for warning in val.warnings:
        st.markdown(
            f'<div class="issue-box">{warning}</div>',
            unsafe_allow_html=True,
        )


def render_metrics_summary() -> None:
    """Display summary metric cards."""
    result = st.session_state.result
    if not result or not result.is_successful:
        return

    cols = st.columns(5)
    metrics = [
        ("Symmetry", result.symmetry.overall_index if result.symmetry else 0),
        ("Harmony", result.harmony.harmony_index if result.harmony else 0),
        ("Eyes", result.eyes.ocular_symmetry_index if result.eyes else 0),
        ("Nose", result.nose.symmetry_index if result.nose else 0),
        ("Jaw", result.jaw.definition_index if result.jaw else 0),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, f"{value:.1f}")


def tab_original() -> None:
    """Tab 1: Original image."""
    result = st.session_state.result
    if result is None:
        st.info("Upload a frontal photo and click **Analyze**.")
        return

    render_validation_status()

    if result.original_image is not None:
        st.image(
            bgr_to_rgb(result.original_image),
            caption="Original Image",
            use_container_width=True,
        )

        col1, col2, col3 = st.columns(3)
        val = result.validation
        col1.metric("Resolution", f"{val.resolution[0]}×{val.resolution[1]}")
        col2.metric("Brightness", f"{val.brightness:.0f}")
        col3.metric("Head Tilt", f"{val.head_tilt_degrees:.1f}°")


def tab_annotated() -> None:
    """Tab 2: Annotated image."""
    result = st.session_state.result
    if result is None or not result.is_successful:
        st.info("No analysis available.")
        return

    if result.annotated_image is not None:
        st.image(
            bgr_to_rgb(result.annotated_image),
            caption="Annotated Image",
            use_container_width=True,
        )

        png_bytes = encode_image_png(result.annotated_image)
        st.download_button(
            "Download Annotated Image (PNG)",
            data=png_bytes,
            file_name="irimidlab_annotated.png",
            mime="image/png",
        )


def tab_metrics() -> None:
    """Tab 3: Full metrics."""
    result = st.session_state.result
    if result is None or not result.is_successful:
        st.info("No analysis available.")
        return

    render_metrics_summary()
    st.markdown("---")

    sections = [
        ("symmetry", result.symmetry),
        ("proportions", result.proportions),
        ("eyes", result.eyes),
        ("nose", result.nose),
        ("lips", result.lips),
        ("jaw", result.jaw),
        ("chin", result.chin),
        ("profile", result.profile),
        ("harmony", result.harmony),
    ]

    for section_key, metrics_obj in sections:
        if metrics_obj is None:
            continue
        title = SECTION_LABELS.get(section_key, section_key.title())
        with st.expander(title, expanded=section_key in ("symmetry", "harmony")):
            data = metrics_obj.to_dict()
            df = pd.DataFrame([
                {"Metric": format_metric_name(k), "Value": v}
                for k, v in data.items()
                if not isinstance(v, (dict, list))
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

            nested = {k: v for k, v in data.items() if isinstance(v, dict)}
            for nk, nv in nested.items():
                st.markdown(f"**{format_metric_name(nk)}**")
                st.json(nv)


def tab_charts() -> None:
    """Tab 4: Charts."""
    result = st.session_state.result
    if result is None or not result.is_successful:
        st.info("No analysis available.")
        return

    charts = st.session_state.charts

    col1, col2 = st.columns(2)
    with col1:
        st.image(charts.symmetry_chart(result), caption="Symmetry by Region")
        st.image(charts.harmony_chart(result), caption="Harmony Components")
    with col2:
        st.image(charts.proportions_chart(result), caption="Facial Thirds")
        st.image(charts.indices_overview_chart(result), caption="Indices Overview")

    st.image(
        charts.reference_comparison_chart(result),
        caption="Anthropometric Reference Comparison",
    )

    history = st.session_state.history.get_all()
    if len(history) >= 2:
        st.markdown("### Temporal Evolution")
        st.image(charts.history_comparison_chart(history))


def tab_report() -> None:
    """Tab 5: Report."""
    result = st.session_state.result
    if result is None:
        st.info("No analysis available.")
        return

    st.markdown("### Generate Report")

    if result.harmony:
        st.info(result.harmony.disclaimer)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Generate PDF", type="primary", use_container_width=True):
            if result.is_successful:
                report_gen = ReportGenerator()
                pdf_bytes = report_gen.generate_bytes(result)
                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"irimidlab_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
            else:
                st.error("Incomplete analysis. Please fix image quality issues.")

    with col2:
        if result.is_successful:
            st.download_button(
                "Export JSON",
                data=DataExporter.to_json(result),
                file_name="irimidlab_analysis.json",
                mime="application/json",
                use_container_width=True,
            )

    with col3:
        if result.is_successful:
            st.download_button(
                "Export CSV",
                data=DataExporter.to_csv(result),
                file_name="irimidlab_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col4:
        if result.is_successful:
            st.download_button(
                "Export Excel",
                data=DataExporter.to_excel(result),
                file_name="irimidlab_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("### Report Preview")

    if result.is_successful:
        st.markdown(f"**Date:** {result.timestamp.strftime('%m/%d/%Y %H:%M:%S')}")

        with st.expander("Metrics Glossary"):
            for key, desc in sorted(METRIC_DESCRIPTIONS.items()):
                st.markdown(f"**{format_metric_name(key)}**: {desc}")


def tab_advanced_geometry() -> None:
    """Tab: Advanced geometry metrics and charts."""
    result = st.session_state.result
    if result is None or not result.is_successful or not result.advanced_geometry:
        st.info("No advanced geometry analysis available.")
        return

    adv = result.advanced_geometry
    charts = st.session_state.charts

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regional Symmetry", f"{adv.regional_symmetry_overall:.1f}")
    col2.metric("Center Offset", f"{adv.center_offset_percent:.1f}%")
    col3.metric("Face Circularity", f"{adv.face_circularity:.2f}")
    col4.metric("EAR (avg)", f"{adv.ear_avg:.3f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(
            charts.advanced_regional_symmetry_chart(result),
            caption="Regional Asymmetry Scores",
        )
        st.image(
            charts.advanced_face_shape_chart(result),
            caption="Face Shape Probabilities",
        )
    with col_b:
        st.image(
            charts.advanced_radar_chart(result),
            caption="Regional Symmetry Radar",
        )
        st.image(
            charts.advanced_region_area_chart(result),
            caption="Regional Area Distribution",
        )

    st.markdown("### Head Pose")
    pose_cols = st.columns(3)
    pose_cols[0].info(adv.head_pitch_interpretation)
    pose_cols[1].info(adv.head_yaw_interpretation)
    pose_cols[2].info(adv.head_roll_interpretation)

    st.markdown("### Complete Metrics Table")
    flat = adv.to_flat_dict()
    df = pd.DataFrame([
        {"Metric": format_metric_name(k), "Value": v}
        for k, v in sorted(flat.items())
        if not isinstance(v, (dict, list))
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Nested Detail View"):
        st.json(adv.to_dict())


def main() -> None:
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_header()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Original Image",
        "Annotated Image",
        "Metrics",
        "Charts",
        "Advanced Geometry",
        "Report",
    ])

    with tab1:
        tab_original()
    with tab2:
        tab_annotated()
    with tab3:
        tab_metrics()
    with tab4:
        tab_charts()
    with tab5:
        tab_advanced_geometry()
    with tab6:
        tab_report()


if __name__ == "__main__":
    main()
