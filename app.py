# ============================================================
# app.py
# Cardiac DSES Digital Twin - FINAL 28-CLASS ARCHITECTURE
#
# This app uses the final multiclass architecture:
#
#   Patient at rest
#        ↓
#   Digital Twin
#        ↓
#   28 Patient-DSES values
#        +
#   Conditional Expected-DSES RF
#        ↓
#   28 Expected-DSES values
#        ↓
#   28 DSES residuals
#        +
#   5 global stress features
#        +
#   5 Digital-Twin physiological features
#        +
#   compact clinical feature layer
#        ↓
#   Final 28-class ExtraTrees classifier
#        ↓
#   28 disease probabilities
#
# IMPORTANT:
# - This replaces the older 50/50 DT + ML ranking.
# - This replaces the older Student-t DSES-distance ranking.
# - Disease is the prediction target, not a final classifier input.
# - Patient is assumed to be at rest.
# - Smoking is fixed to NO.
#
# STYLING-ONLY REVISION: all model / loading / prediction logic below is
# unchanged. Only the visual presentation (CSS, colours, tabs, charts)
# has been modified.
# ============================================================

from pathlib import Path
import sys
import tempfile
import urllib.request
import joblib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. ROBUST PATH
# ============================================================

BASE_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)


# ============================================================
# 2. ROBUST MODEL / ARTIFACT PATHS
# ============================================================

# Python modules and trained artifacts may be stored either beside
# app.py or inside the repository's pkl+joblib folder.
SEARCH_DIRS = [
    BASE_DIR,
    BASE_DIR / "pkl+joblib",
]

# GitHub Release assets used when the large models are not stored locally.
RELEASE_BASE = (
    "https://github.com/qp361516-stack/turbo-broccoli/"
    "releases/download/v1.0/"
)
RELEASE_ASSETS = {
    "final_28class_dses_classifier.joblib": (
        RELEASE_BASE + "final_28class_dses_classifier.joblib"
    ),
    "disease_dses_rf.joblib": (
        RELEASE_BASE + "disease_dses_rf.joblib"
    ),
}
DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "cardiac_dses_models"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Accept the exact classifier filename currently present in your repository,
# as well as the standard filename and the previous duplicate-suffix variants.
CLASSIFIER_CANDIDATES = [
    "final_28class_dses_classifier(1) (1).joblib",
    "final_28class_dses_classifier(1).joblib",
    "final_28class_dses_classifier.joblib",
]
FEATURE_COLUMN_CANDIDATES = [
    "final_28class_feature_columns.joblib",
]

def resolve_file(filename):
    """Return the first existing file in supported runtime or download directories."""
    for directory in SEARCH_DIRS:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    cached = DOWNLOAD_DIR / filename
    if cached.exists():
        return cached
    return None

def ensure_release_asset(filename):
    """Download a known Release asset only when it is not already local."""
    existing = resolve_file(filename)
    if existing is not None:
        return existing, None
    url = RELEASE_ASSETS.get(filename)
    if not url:
        return None, RuntimeError(f"No Release URL configured for {filename}")
    target = DOWNLOAD_DIR / filename
    try:
        urllib.request.urlretrieve(url, target)
        if not target.exists():
            raise RuntimeError("Download completed but the model file was not created.")
        return target, None
    except Exception as exc:
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        return None, exc

REQUIRED_MODULES = [
    "digital_twin.py",
    "ml_final_28class_FIXED.py",
]

# These are mandatory. Feature columns are optional because they can be
# reconstructed from sklearn's feature_names_in_ on the saved classifier.
REQUIRED_ARTIFACTS = [
    "disease_dses_rf.joblib",
    "disease_dses_model_columns.joblib",
    "disease_dses_model_medians.joblib",
]

missing = []
resolved_modules = {}
resolved_artifacts = {}

for filename in REQUIRED_MODULES:
    path = resolve_file(filename)
    if path is None:
        missing.append(filename)
    else:
        resolved_modules[filename] = path

# Resolve/download the large Expected-DSES RF first.
rf_path, rf_error = ensure_release_asset("disease_dses_rf.joblib")
if rf_path is not None:
    resolved_artifacts["disease_dses_rf.joblib"] = rf_path
elif rf_error is not None:
    missing.append(f"pkl+joblib/disease_dses_rf.joblib (Release download failed: {rf_error})")

for filename in [
    "disease_dses_model_columns.joblib",
    "disease_dses_model_medians.joblib",
]:
    path = resolve_file(filename)
    if path is None:
        missing.append(f"pkl+joblib/{filename}")
    else:
        resolved_artifacts[filename] = path

# Resolve the classifier from any supported local filename first; if none is
# available, fall back to the standard GitHub Release asset.
classifier_path = None
for candidate in CLASSIFIER_CANDIDATES:
    path = resolve_file(candidate)
    if path is not None:
        classifier_path = path
        break
if classifier_path is None:
    classifier_path, classifier_error = ensure_release_asset("final_28class_dses_classifier.joblib")
    if classifier_path is None:
        missing.append(f"pkl+joblib/final_28class_dses_classifier.joblib (Release download failed: {classifier_error})")
resolved_artifacts["final_28class_dses_classifier.joblib"] = classifier_path

# Feature-column file is optional; we will reconstruct it from the classifier
# below when necessary.
feature_cols_path = None
for candidate in FEATURE_COLUMN_CANDIDATES:
    path = resolve_file(candidate)
    if path is not None:
        feature_cols_path = path
        break
if feature_cols_path is not None:
    resolved_artifacts["final_28class_feature_columns.joblib"] = feature_cols_path


# ============================================================
# 3. IMPORT MODEL MODULES
# ============================================================

if not missing:
    try:
        # Add the directories containing the runtime Python modules.
        # This supports both repository layouts:
        #   /app.py + /digital_twin.py + /ml_final_28class_FIXED.py
        # and
        #   /app.py + /pkl+joblib/{digital_twin.py, ml_final_28class_FIXED.py}
        for module_path in resolved_modules.values():
            if module_path.parent.exists():
                sys.path.insert(0, str(module_path.parent))

        from digital_twin import (
            DISEASE_REACTION_MAP,
            build_or_load_references,
            compute_patient_dses_scores,
        )

        import ml_final_28class_FIXED as _ml_runtime

        # Point the runtime module at the actual artifact locations.
        _ml_runtime.RF_PATH = resolved_artifacts[
            "disease_dses_rf.joblib"
        ]
        _ml_runtime.RF_COLS = resolved_artifacts[
            "disease_dses_model_columns.joblib"
        ]
        _ml_runtime.RF_MED = resolved_artifacts[
            "disease_dses_model_medians.joblib"
        ]
        _ml_runtime.CLF_PATH = resolved_artifacts[
            "final_28class_dses_classifier.joblib"
        ]

        # The feature-column artifact is optional. If it is missing, recover
        # the exact training feature names from the saved sklearn classifier.
        if "final_28class_feature_columns.joblib" in resolved_artifacts:
            _ml_runtime.FEATURE_COLS_PATH = resolved_artifacts[
                "final_28class_feature_columns.joblib"
            ]
        else:
            clf = joblib.load(_ml_runtime.CLF_PATH)
            feature_names = getattr(clf, "feature_names_in_", None)
            if feature_names is None:
                raise ValueError(
                    "final_28class_feature_columns.joblib is missing and the saved "
                    "classifier does not expose sklearn feature_names_in_. "
                    "Please upload the matching feature-column artifact."
                )
            generated = DOWNLOAD_DIR / "final_28class_feature_columns.joblib"
            joblib.dump(list(feature_names), generated)
            _ml_runtime.FEATURE_COLS_PATH = generated
            resolved_artifacts["final_28class_feature_columns.joblib"] = generated

        build_features = _ml_runtime.build_features
        predict_patient = _ml_runtime.predict_patient

    except Exception as exc:
        st.set_page_config(
            page_title="Cardiac DSES Digital Twin",
            layout="wide",
        )
        st.title("Cardiac DSES Digital Twin")
        st.error(
            "The 28-class DSES architecture could not be loaded."
        )
        st.exception(exc)
        st.stop()
else:
    st.set_page_config(
        page_title="Cardiac DSES Digital Twin",
        layout="wide",
    )
    st.title("Cardiac DSES Digital Twin")

    st.error(
        "Required 28-class DSES architecture files are missing."
    )

    st.write(
        "Expected Python modules and trained model artifacts either "
        "beside app.py or inside `pkl+joblib/`:"
    )

    for filename in missing:
        st.write(f"- `{filename}`")

    st.stop()


# ============================================================
# 4. PAGE CONFIG  (+ VISUAL THEME)
# ============================================================

# set_page_config may already have been called above in an error
# branch. In normal execution, it has not been called yet.
try:
    st.set_page_config(
        page_title="Cardiac DSES Digital Twin",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# ---------- Theme detection ----------
def _detect_dark_theme():
    """True when Streamlit is rendering in dark mode."""
    try:
        return str(st.context.theme.type).lower() == "dark"
    except Exception:
        pass
    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        return False


IS_DARK = _detect_dark_theme()

# ---------- Colour palette ----------
# A restrained clinical dashboard palette with readable light/dark themes.
NAVY = "#17324D"
RED = "#D6534A"
TEAL = "#138577"
BLUE = "#397BA5"
GREY = "#758396"

if IS_DARK:
    TEXT = "#EAF0F7"
    MUTED = "#AAB7C8"
    CARD_BG = "#172333"
    CARD_BORDER = "#2B3B50"
    GRID = "#2B3B50"
    BAR = "#83A9C7"
    TAB_BG = "#1A293B"
    TAB_SEL = "#24566B"
    HERO_BG = "#102033"
    PAGE_BG = "#0D1724"
else:
    TEXT = "#203247"
    MUTED = "#68798B"
    CARD_BG = "#FFFFFF"
    CARD_BORDER = "#DCE5ED"
    GRID = "#E8EEF3"
    BAR = NAVY
    TAB_BG = "#EEF3F7"
    TAB_SEL = NAVY
    HERO_BG = "#17324D"
    PAGE_BG = "#F5F8FB"

CUSTOM_CSS = """
<style>
/* ---------- page foundation ---------- */
.block-container {
    max-width: 1580px;
    padding-top: 1.45rem;
    padding-bottom: 3rem;
}
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: __PAGE_BG__;
}
[data-testid="stHeader"] {
    background: transparent;
}
footer, #MainMenu { visibility: hidden; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: __HERO_BG__;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #EEF3F8 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.16);
    margin: 0.8rem 0;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    background: rgba(255,255,255,0.035);
    margin-bottom: 0.65rem;
}
/* Keep fields bright and legible against the dark sidebar. */
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] input {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #203247 !important;
}
[data-testid="stSidebar"] input {
    -webkit-text-fill-color: #203247 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: #E9EFF4 !important;
    color: #203247 !important;
    border-color: #D8E2EB !important;
}

/* ---------- main title banner ---------- */
.hero {
    background: linear-gradient(120deg, __HERO_BG__ 0%, #204B62 100%);
    border-left: 5px solid __TEAL__;
    border-radius: 14px;
    padding: 24px 30px;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 24px rgba(20, 43, 63, 0.10);
}
.hero h1 {
    color: #FFFFFF !important;
    margin: 0;
    padding: 0;
    font-size: clamp(1.55rem, 2.2vw, 2rem);
    font-weight: 750;
    letter-spacing: -0.025em;
    line-height: 1.25;
}
.hero p {
    color: #D9E6EF !important;
    margin: 8px 0 0 0;
    font-size: 0.98rem;
}
.hero .hero-meta {
    color: #B7CBD9 !important;
    font-size: 0.82rem;
    margin-top: 10px;
    letter-spacing: 0.015em;
}

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {
    background: __CARD_BG__;
    border: 1px solid __CARD_BORDER__;
    border-top: 3px solid __TEAL__;
    border-radius: 12px;
    padding: 15px 17px;
    min-height: 108px;
    box-shadow: 0 3px 12px rgba(30, 55, 75, 0.035);
}
[data-testid="stMetricLabel"] p {
    color: __MUTED__ !important;
    font-weight: 650;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.045em;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: __TEXT__ !important;
    font-weight: 750;
    font-size: 1.3rem;
    white-space: normal;
    overflow-wrap: anywhere;
}

/* ---------- section typography ---------- */
h1, h2, h3 { letter-spacing: -0.015em; }
h2 {
    font-weight: 720;
    border-bottom: 1px solid __CARD_BORDER__;
    padding-bottom: 0.5rem;
    margin-top: 1.3rem;
}
h3 { font-weight: 680; }
[data-testid="stCaptionContainer"] p {
    color: __MUTED__;
}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: __TAB_BG__;
    border: 1px solid __CARD_BORDER__;
    border-radius: 12px;
    padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    padding: 9px 16px;
    min-height: 42px;
}
.stTabs [data-baseweb="tab"] p {
    color: __TEXT__ !important;
    font-weight: 650;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: __TAB_SEL__ !important;
}
.stTabs [aria-selected="true"] p {
    color: #FFFFFF !important;
}

/* ---------- controls, alerts, tables, expanders ---------- */
.stButton > button {
    border-radius: 9px;
    min-height: 44px;
    font-weight: 700;
    letter-spacing: 0.01em;
    transition: transform 120ms ease, box-shadow 120ms ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 14px rgba(25, 65, 85, 0.14);
}
[data-testid="stDataFrame"] {
    border: 1px solid __CARD_BORDER__;
    border-radius: 10px;
    overflow: hidden;
    background: __CARD_BG__;
}
[data-testid="stAlert"] {
    border-radius: 10px;
}
[data-testid="stExpander"] {
    border: 1px solid __CARD_BORDER__;
    border-radius: 10px;
    background: __CARD_BG__;
}
hr { border-color: __CARD_BORDER__; }
</style>
"""

for _token, _value in {
    "__HERO_BG__": HERO_BG,
    "__RED__": RED,
    "__TEAL__": TEAL,
    "__PAGE_BG__": PAGE_BG,
    "__CARD_BG__": CARD_BG,
    "__CARD_BORDER__": CARD_BORDER,
    "__MUTED__": MUTED,
    "__TEXT__": TEXT,
    "__TAB_BG__": TAB_BG,
    "__TAB_SEL__": TAB_SEL,
}.items():
    CUSTOM_CSS = CUSTOM_CSS.replace(_token, _value)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def style_fig(fig, height=None):
    """Apply the shared chart look. Background and text colours are paired."""
    fig.update_layout(
        template="plotly_dark" if IS_DARK else "plotly_white",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=TEXT),
        title=dict(font=dict(size=17, color=TEXT)),
        margin=dict(l=16, r=24, t=58, b=28),
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT),
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=TEXT),
        title_font=dict(color=TEXT),
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=TEXT),
        title_font=dict(color=TEXT),
    )
    fig.update_traces(
        textfont=dict(color=TEXT),
        selector=dict(type="bar"),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _blend(stops, t):
    """Interpolate an (r, g, b) colour along evenly spaced hex stops."""
    t = min(max(float(t), 0.0), 1.0)
    n = len(stops) - 1
    pos = t * n
    i = min(int(pos), n - 1)
    f = pos - i
    a = _hex_to_rgb(stops[i])
    b = _hex_to_rgb(stops[i + 1])
    return tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


def gradient_css(values, stops, vmin, vmax):
    """Per-cell CSS background gradient (replaces matplotlib-based Styler.background_gradient)."""
    out = []
    for v in values:
        if pd.isna(v) or vmax == vmin:
            out.append("")
            continue
        r, g, b = _blend(stops, (v - vmin) / (vmax - vmin))
        text = "#1F2A44" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#FFFFFF"
        out.append(f"background-color: rgb({r},{g},{b}); color: {text}")
    return out


st.markdown(
    """
    <div class="hero">
        <h1>Cardio-Thermodynamic Digital Twin</h1>
        <p>Disease-Specific Entropy Signature (DSES) &middot; 28-class model</p>
        <div class="hero-meta">
            Resting patient &middot; Smoking fixed to NO &middot; Research prototype
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Runtime file check", expanded=False):
    st.write("Digital Twin:", str(resolved_modules["digital_twin.py"]))
    st.write("Final ML module:", str(resolved_modules["ml_final_28class_FIXED.py"]))
    for name, path in resolved_artifacts.items():
        st.write(f"{name}:", str(path))


# ============================================================
# 5. REFERENCES / DISEASES
# ============================================================

@st.cache_resource(show_spinner=False)
def get_references():
    # If the Digital Twin module lives in pkl+joblib, its default
    # reference workbook/cache are resolved relative to that folder.
    dt_module_dir = resolved_modules["digital_twin.py"].parent

    workbook_candidates = [
        dt_module_dir / "Cardiac_Disease_Synchronized_Dataset_Patient_1_Literature_Conditioned_CLEAN (4).xlsx",
        dt_module_dir / "Cardiac_Disease_Synchronized_Dataset_Patient_1_Literature_Conditioned_CLEAN (4)_2.xlsx",
        BASE_DIR / "Cardiac_Disease_Synchronized_Dataset_Patient_1_Literature_Conditioned_CLEAN (4).xlsx",
        BASE_DIR / "Cardiac_Disease_Synchronized_Dataset_Patient_1_Literature_Conditioned_CLEAN (4)_2.xlsx",
    ]

    workbook = next((p for p in workbook_candidates if p.exists()), None)

    cache = dt_module_dir / "dt_reference_values.pkl"

    if workbook is not None:
        return build_or_load_references(
            workbook_path=str(workbook),
            cache_path=str(cache),
        )

    return build_or_load_references()


@st.cache_resource(show_spinner=False)
def get_diseases():
    return [
        disease
        for disease in DISEASE_REACTION_MAP.keys()
        if str(disease).strip().lower() != "healthy"
    ]


try:
    refs = get_references()
    diseases = get_diseases()
except Exception as exc:
    st.error(
        f"Could not load Digital Twin reference data: {exc}"
    )
    st.stop()


# ============================================================
# 6. SIDEBAR - PATIENT INPUT
# ============================================================

with st.sidebar:

    st.header("Patient Profile")
    st.caption("Enter resting measurements, then run the model.")

    with st.expander("Demographics", expanded=True):
        age = st.number_input(
            "Age (years)",
            min_value=18.0,
            max_value=120.0,
            value=55.0,
            step=1.0,
        )

        sex = st.selectbox(
            "Biological Sex",
            ["Female", "Male"],
        )

        height = st.number_input(
            "Height (cm)",
            min_value=80.0,
            max_value=230.0,
            value=170.0,
            step=0.1,
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=250.0,
            value=75.0,
            step=0.1,
        )

    with st.expander("Hemodynamics", expanded=True):
        hr = st.number_input(
            "Resting Heart Rate (bpm)",
            min_value=30.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
        )

        sbp = st.number_input(
            "Systolic BP (mmHg)",
            min_value=70.0,
            max_value=250.0,
            value=120.0,
            step=1.0,
        )

        dbp = st.number_input(
            "Diastolic BP (mmHg)",
            min_value=40.0,
            max_value=160.0,
            value=80.0,
            step=1.0,
        )

        edv = st.number_input(
            "End-Diastolic Volume (mL)",
            min_value=20.0,
            max_value=500.0,
            value=150.0,
            step=1.0,
        )

        esv = st.number_input(
            "End-Systolic Volume (mL)",
            min_value=5.0,
            max_value=400.0,
            value=60.0,
            step=1.0,
        )

        pawp = st.number_input(
            "PAWP (mmHg)",
            min_value=0.0,
            max_value=40.0,
            value=12.0,
            step=0.1,
        )

        myocardial_mass = st.number_input(
            "Myocardial Mass (g)",
            min_value=50.0,
            max_value=500.0,
            value=150.0,
            step=1.0,
        )

    with st.expander("Laboratory & Physiology", expanded=False):
        core_temp = st.number_input(
            "Core Temperature (°C)",
            min_value=34.0,
            max_value=41.0,
            value=37.0,
            step=0.01,
        )

        spo2 = st.number_input(
            "SpO₂ (%)",
            min_value=70.0,
            max_value=100.0,
            value=98.0,
            step=0.1,
        )

        ph = st.number_input(
            "Arterial pH",
            min_value=6.8,
            max_value=7.8,
            value=7.40,
            step=0.001,
        )

        fbs = st.number_input(
            "Fasting Blood Sugar (mg/dL)",
            min_value=50.0,
            max_value=500.0,
            value=90.0,
            step=1.0,
        )

        chol = st.number_input(
            "Serum Cholesterol (mg/dL)",
            min_value=80.0,
            max_value=500.0,
            value=200.0,
            step=1.0,
        )

    with st.expander("Clinical Phenotype", expanded=True):
        chest_pain = st.selectbox(
            "Chest Pain Type",
            [
                "asymptomatic",
                "typical angina",
                "atypical angina",
                "non-anginal pain",
            ],
        )

        rest_ecg = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "Left ventricular hypertrophy",
                "ST-T wave abnormality",
            ],
        )

    st.divider()
    run_button = st.button(
        "Run 28-Class Model",
        type="primary",
        use_container_width=True,
    )

# ============================================================
# 7. PATIENT DICTIONARY
# ============================================================

patient = {
    "age": age,
    "sex": sex,
    "height_cm": height,
    "weight_kg": weight,
    "hr": hr,
    "sbp": sbp,
    "dbp": dbp,
    "edv": edv,
    "esv": esv,
    "pawp": pawp,
    "myocardial_mass_g": myocardial_mass,
    "temp_c": core_temp,
    "spo2": spo2,
    "ph": ph,
    "fbs": fbs,
    "chol": chol,
    "chest_pain": chest_pain,
    "rest_ecg": rest_ecg,
}


# ============================================================
# 8. RUN MODEL
# ============================================================

if run_button:

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if edv <= esv:
        st.error("EDV must be greater than ESV.")
        st.stop()

    if sbp <= dbp:
        st.error("Systolic BP must be greater than DBP.")
        st.stop()

    if height <= 0:
        st.error("Height must be greater than zero.")
        st.stop()

    if weight <= 0:
        st.error("Weight must be greater than zero.")
        st.stop()

    # --------------------------------------------------------
    # Run Digital Twin + final classifier
    # --------------------------------------------------------

    with st.spinner(
        "Running Digital Twin and final 28-class classifier..."
    ):

        try:
            dt_result = compute_patient_dses_scores(
                patient,
                refs,
            )

            final_predictions = predict_patient(
                patient,
                refs,
            )

            # Build the same final feature vector used during
            # final-classifier development so the intermediate
            # DSES/stress/physiology features can be displayed.
            model_input = pd.DataFrame([
                {
                    "Age_years": age,
                    "Biological_Sex": sex,
                    "Height_cm": height,
                    "Weight_kg": weight,
                    "Heart_Rate_bpm": hr,
                    "Systolic_BP_mmHg": sbp,
                    "Diastolic_BP_mmHg": dbp,
                    "EDV_mL": edv,
                    "ESV_mL": esv,
                    "PAWP_mmHg": pawp,
                    "Core_Temp_C": core_temp,
                    "SpO2_pct": spo2,
                    "pH": ph,
                    "Fasting_Blood_Sugar": fbs,
                    "Serum_Cholesterol": chol,
                    "Myocardial_Mass_g": myocardial_mass,
                    "Chest_Pain_Type": chest_pain,
                    "Resting_ECG": rest_ecg,
                }
            ])

            final_features = build_features(
                model_input,
                refs,
            )

        except Exception as exc:
            st.error(
                "Model execution failed."
            )
            st.exception(exc)
            st.stop()

    # Store results so they survive any later widget interaction.
    st.session_state["result"] = {
        "patient": dict(patient),
        "dt_result": dt_result,
        "final_predictions": final_predictions,
        "final_features": final_features,
    }


# ============================================================
# 9. RENDER RESULTS (persisted in session_state)
# ============================================================

if "result" in st.session_state:

    _res = st.session_state["result"]
    dt_result = _res["dt_result"]
    final_predictions = _res["final_predictions"]
    final_features = _res["final_features"]

    if _res["patient"] != patient:
        st.warning(
            "Sidebar inputs have changed since the last run. The results "
            "below are for the previous inputs - click **RUN 28-CLASS "
            "MODEL** to update them."
        )

    # ========================================================
    # TOP PREDICTION
    # ========================================================

    top_row = final_predictions.iloc[0]

    top_disease = str(
        top_row["Disease"]
    )

    top_probability = float(
        top_row["Probability"]
    )

    common = dt_result["common"]

    def _detect_atp_scale(c):
        """
        Factor that converts the Digital Twin's ATP-rate values to mol/s
        (the unit used in the paper, Eq. 9). The module's keys are named
        "(mol/min)", so we check the value against W_chem * 0.60 / 30500
        and only divide by 60 if the value really is per-minute.
        """
        try:
            raw = float(c["ATP Production (mol/min)"])
            expected = float(c["Chemical Power (W)"]) * 0.60 / 30500.0
            if raw > 0 and expected > 0:
                if abs(np.log(raw / expected)) <= abs(np.log(raw / 60.0 / expected)):
                    return 1.0
                return 1.0 / 60.0
        except (KeyError, TypeError, ValueError):
            pass
        return 1.0

    atp_scale = _detect_atp_scale(common)

    # ========================================================
    # TOP METRICS
    # ========================================================

    st.header("Prediction Overview")
    st.caption("Top-ranked output and key resting physiological estimates.")

    metric_row_1 = st.columns([1.5, 1, 1], gap="medium")
    metric_row_1[0].metric("Predicted Disease", top_disease)
    metric_row_1[1].metric("Model Probability", f"{top_probability:.2%}")
    metric_row_1[2].metric("Ejection Fraction", f"{common['EF'] * 100:.2f}%")

    metric_row_2 = st.columns(2, gap="medium")
    metric_row_2[0].metric("Stroke Volume", f"{common['SV']:.2f} mL")
    metric_row_2[1].metric("Mean Arterial Pressure", f"{common['MAP']:.2f} mmHg")

    st.caption(
        "Research prototype: model probabilities are not clinically calibrated "
        "diagnostic probabilities and should not be used alone for clinical decisions."
    )

    # ========================================================
    # TABS (visual grouping only - content is unchanged)
    # ========================================================

    tab_pred, tab_dses, tab_stress, tab_react, tab_info = st.tabs(
        [
            "Overview",
            "DSES Analysis",
            "Stress & Physiology",
            "Reaction Details",
            "Model Info",
        ]
    )

    # ========================================================
    # TAB 1 - PREDICTION
    # ========================================================

    with tab_pred:

        # ----------------------------------------------------
        # ARCHITECTURE PIPELINE
        # ----------------------------------------------------

        st.header("28-Class DSES Architecture")
        st.caption("Workflow from resting patient inputs to the final disease-probability ranking.")

        pipeline = go.Figure(
            go.Sankey(
                node=dict(
                    label=[
                        "Patient at Rest",
                        "Digital Twin",
                        "28 Patient DSES",
                        "Expected-DSES RF",
                        "28 Expected DSES",
                        "28 DSES Residuals",
                        "5 Global Stresses",
                        "5 DT Physiology",
                        "Clinical Layer",
                        "Final 28-Class Classifier",
                        "28 Disease Probabilities",
                    ],
                    color=[
                        BAR, BLUE, TEAL, BLUE, TEAL, TEAL,
                        TEAL, TEAL, TEAL, RED, RED,
                    ],
                    pad=18,
                    thickness=18,
                ),
                link=dict(
                    source=[
                        0, 1, 2, 3, 4,
                        5, 6, 7, 8, 9
                    ],
                    target=[
                        1, 2, 3, 4, 5,
                        6, 7, 8, 9, 10
                    ],
                    value=[1] * 10,
                    color="rgba(46,134,193,0.28)",
                ),
            )
        )

        pipeline.update_layout(
            title="Final Cardiac DSES Prediction Pipeline",
        )

        style_fig(pipeline, height=400)

        st.plotly_chart(
            pipeline,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # RANKING TABLE + TOP-10 CHART
        # ----------------------------------------------------

        rank_col, chart_col = st.columns([1.05, 1.0], gap="large")

        with rank_col:
            st.subheader("28-Class Disease Ranking")

            ranking = final_predictions.copy()
            ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
            ranking["Probability"] = ranking["Probability"] * 100.0

            st.dataframe(
                ranking[["Rank", "Disease", "Probability"]],
                column_config={
                    "Rank": st.column_config.NumberColumn(
                        "Rank",
                        width="small",
                    ),
                    "Disease": st.column_config.TextColumn(
                        "Disease",
                        width="large",
                    ),
                    "Probability": st.column_config.ProgressColumn(
                        "Probability",
                        format="%.2f%%",
                        min_value=0.0,
                        max_value=100.0,
                    ),
                },
                use_container_width=True,
                hide_index=True,
                height=520,
            )

        with chart_col:
            st.subheader("Top 10 Predictions")

            top10 = ranking.head(10).iloc[::-1]
            top10_colors = [BAR] * (len(top10) - 1) + [RED]

            fig_top10 = go.Figure(
                go.Bar(
                    x=top10["Probability"],
                    y=top10["Disease"],
                    orientation="h",
                    marker=dict(color=top10_colors),
                    text=[f"{value:.2f}%" for value in top10["Probability"]],
                    textposition="outside",
                    cliponaxis=False,
                )
            )

            fig_top10.update_layout(
                title="Highest-Ranked Disease Predictions",
                xaxis_title="Model Probability (%)",
                yaxis_title="",
                showlegend=False,
            )
            style_fig(fig_top10, height=520)
            fig_top10.update_xaxes(range=[0, max(1, float(top10["Probability"].max()) * 1.2)])

            st.plotly_chart(
                fig_top10,
                use_container_width=True,
            )

    # ========================================================
    # PATIENT DSES / EXPECTED DSES / RESIDUALS
    # ========================================================

    with tab_dses:

        st.header(
            "DSES Features Used by the Final Classifier"
        )

        dses_rows = []

        for disease in diseases:

            p_col = f"P_{disease}"
            e_col = f"E_{disease}"
            r_col = f"R_{disease}"

            dses_rows.append(
                {
                    "Disease": disease,
                    "Patient DSES": (
                        float(
                            final_features.iloc[0][p_col]
                        )
                        if p_col in final_features.columns
                        else np.nan
                    ),
                    "Expected DSES": (
                        float(
                            final_features.iloc[0][e_col]
                        )
                        if e_col in final_features.columns
                        else np.nan
                    ),
                    "DSES Residual": (
                        float(
                            final_features.iloc[0][r_col]
                        )
                        if r_col in final_features.columns
                        else np.nan
                    ),
                }
            )

        dses_table = pd.DataFrame(
            dses_rows
        )

        dses_table = dses_table.merge(
            final_predictions,
            on="Disease",
            how="left",
        )

        dses_table["Probability"] = (
            dses_table["Probability"] * 100.0
        )

        dses_table = (
            dses_table
            .sort_values(
                "Probability",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        dses_table.insert(
            0,
            "Rank",
            np.arange(
                1,
                len(dses_table) + 1,
            ),
        )

        # Residual column gets a diverging colour map; probability a green ramp.
        _res_vals = dses_table["DSES Residual"].dropna()
        _res_lim = max(
            float(_res_vals.abs().max()) if len(_res_vals) else 0.0,
            1e-9,
        )
        _prob_max = max(float(dses_table["Probability"].max()), 1e-9)

        styled_dses = (
            dses_table.style
            .format(
                {
                    "Patient DSES": "{:.3f}",
                    "Expected DSES": "{:.3f}",
                    "DSES Residual": "{:.3f}",
                    "Probability": "{:.2f}%",
                }
            )
            .apply(
                lambda s: gradient_css(
                    s, ["#2E86C1", "#FFFFFF", "#C0392B"],
                    -_res_lim, _res_lim,
                ),
                subset=["DSES Residual"],
            )
            .apply(
                lambda s: gradient_css(
                    s, ["#FFFFFF", "#A9DFBF", "#16A085"],
                    0.0, _prob_max,
                ),
                subset=["Probability"],
            )
        )

        st.dataframe(
            styled_dses,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # PATIENT VS EXPECTED DSES
        # ====================================================

        fig_dses = go.Figure()

        selected_dses = (
            dses_table
            .head(10)
            .iloc[::-1]
        )

        fig_dses.add_trace(
            go.Bar(
                x=selected_dses["Patient DSES"],
                y=selected_dses["Disease"],
                orientation="h",
                name="Patient DSES",
                marker=dict(color=RED),
            )
        )

        fig_dses.add_trace(
            go.Bar(
                x=selected_dses["Expected DSES"],
                y=selected_dses["Disease"],
                orientation="h",
                name="Expected DSES",
                marker=dict(color=BAR),
            )
        )

        fig_dses.update_layout(
            title="Patient DSES vs Expected DSES - Top 10",
            xaxis_title="DSES",
            yaxis_title="Disease",
            barmode="group",
        )

        style_fig(fig_dses, height=520)

        st.plotly_chart(
            fig_dses,
            use_container_width=True,
        )

        # ====================================================
        # DSES RESIDUAL CHART
        # ====================================================

        residual_plot = (
            dses_table
            .head(15)
            .iloc[::-1]
        )

        residual_colors = [
            RED if v >= 0 else BLUE
            for v in residual_plot["DSES Residual"]
        ]

        fig_residual = go.Figure(
            go.Bar(
                x=residual_plot["DSES Residual"],
                y=residual_plot["Disease"],
                orientation="h",
                marker=dict(color=residual_colors),
            )
        )

        fig_residual.add_vline(
            x=0,
            line_dash="dash",
            line_color=GREY,
        )

        fig_residual.update_layout(
            title="DSES Residuals - Top 15 Predicted Diseases",
            xaxis_title="Patient DSES - Expected DSES",
            yaxis_title="Disease",
        )

        style_fig(fig_residual, height=620)

        st.plotly_chart(
            fig_residual,
            use_container_width=True,
        )

    # ========================================================
    # GLOBAL STRESS + PHYSIOLOGY
    # ========================================================

    with tab_stress:

        st.header("Global Digital-Twin Stress Features")

        stress_names = [
            "Metabolic Stress",
            "Mechanical Stress",
            "Thermodynamic Stress",
            "ATP Stress",
            "Entropy Stress",
        ]

        stress_values = []

        for name in stress_names:
            column = f"G_{name}"

            stress_values.append(
                float(
                    final_features.iloc[0].get(
                        column,
                        np.nan,
                    )
                )
            )

        stress_df = pd.DataFrame(
            {
                "Stress Component": stress_names,
                "Value": stress_values,
            }
        )

        st.dataframe(
            stress_df.style.format(
                {"Value": "{:.4f}"}
            ).apply(
                lambda s: gradient_css(
                    s, ["#FFF5EB", "#FDBB84", "#C0392B"],
                    float(s.min()), float(s.max()),
                ),
                subset=["Value"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Bars above the 1.0 reference are red, others teal.
        stress_colors = [
            RED if (not np.isnan(v) and v > 1.0) else TEAL
            for v in stress_df["Value"]
        ]

        fig_stress = go.Figure(
            go.Bar(
                x=stress_df["Stress Component"],
                y=stress_df["Value"],
                marker=dict(color=stress_colors),
                text=[f"{v:.3f}" for v in stress_df["Value"]],
                textposition="outside",
            )
        )

        fig_stress.add_hline(
            y=1.0,
            line_dash="dash",
            line_color=GREY,
            annotation_text="Reference = 1.0",
        )

        fig_stress.update_layout(
            title="Global Stress Profile",
            xaxis_title="Stress",
            yaxis_title="Normalized Value",
        )

        style_fig(fig_stress, height=460)

        st.plotly_chart(
            fig_stress,
            use_container_width=True,
        )

        # ====================================================
        # DIGITAL TWIN PHYSIOLOGY
        # ====================================================

        st.header("Digital Twin Physiological State")

        physiology_rows = [
            (
                "Stroke Volume",
                common["SV"],
                "mL",
            ),
            (
                "Ejection Fraction",
                common["EF"] * 100.0,
                "%",
            ),
            (
                "Mean Arterial Pressure",
                common["MAP"],
                "mmHg",
            ),
            (
                "Cardiac Output",
                common["CO"],
                "mL/min proxy",
            ),
            (
                "Rate-Pressure Product",
                common["RPP"],
                "mmHg·bpm",
            ),
            (
                "LV Stroke Work",
                common["LVSW"],
                "J/beat",
            ),
            (
                "MVO₂",
                common["MVO2"],
                "DT units",
            ),
            (
                "Chemical Power",
                common["Chemical Power (W)"],
                "W",
            ),
            (
                "Mechanical Power",
                common["Mechanical Power (W)"],
                "W",
            ),
            (
                "Heat Production",
                common["Heat Production (W)"],
                "W",
            ),
            (
                "ATP Production",
                common["ATP Production (mol/min)"] * atp_scale,
                "mol/s",
            ),
            (
                "ATP Utilization",
                common["ATP Utilization (mol/min)"] * atp_scale,
                "mol/s",
            ),
            (
                "ATP Utilization Fraction",
                common["ATP Utilization Fraction"],
                "ratio",
            ),
            (
                "ATP Balance",
                common["ATP Balance (mol/min)"] * atp_scale,
                "mol/s",
            ),
        ]

        physiology_df = pd.DataFrame(
            physiology_rows,
            columns=[
                "Variable",
                "Value",
                "Unit",
            ],
        )

        st.dataframe(
            physiology_df.style.format(
                {"Value": "{:.6g}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # REACTION-LEVEL DETAIL FOR TOP DISEASE
    # ========================================================

    with tab_react:

        st.header(
            f"Reaction-Level Detail — {top_disease}"
        )

        top_reaction_details = (
            dt_result
            .get("reaction_details", {})
            .get(top_disease, {})
        )

        reaction_rows = []

        for reaction, detail in top_reaction_details.items():

            row = {
                "Biochemical Reaction": reaction
            }

            if isinstance(detail, dict):
                row.update(detail)

            reaction_rows.append(row)

        if reaction_rows:

            reaction_df = pd.DataFrame(
                reaction_rows
            )

            st.dataframe(
                reaction_df,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No reaction-level details were available "
                "for the top predicted disease."
            )

        # ====================================================
        # FULL DIGITAL-TWIN CALCULATION DETAILS
        # ====================================================

        with st.expander(
            "Show Full Digital-Twin Calculations"
        ):

            calculation = []

            for key, value in common.items():

                if isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                ):

                    _is_atp_rate = "mol/min" in str(key)

                    calculation.append(
                        {
                            "Variable": str(key).replace("mol/min", "mol/s"),
                            "Value": float(value) * (
                                atp_scale if _is_atp_rate else 1.0
                            ),
                        }
                    )

            calculation_df = pd.DataFrame(
                calculation
            )

            if not calculation_df.empty:
                st.dataframe(
                    calculation_df.style.format(
                        {"Value": "{:.8g}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    # ========================================================
    # MODEL FEATURE COUNT
    # ========================================================

    with tab_info:

        st.header("Final Classifier Feature Summary")

        feature_summary = pd.DataFrame(
            {
                "Feature Group": [
                    "Patient DSES",
                    "Expected DSES",
                    "DSES residuals",
                    "Global stress summaries",
                    "DT physiological features",
                    "Compact clinical layer",
                ],
                "Features": [
                    28,
                    28,
                    28,
                    5,
                    5,
                    12,
                ],
                "Total": [
                    "",
                    "",
                    "",
                    "",
                    "",
                    106,
                ],
            }
        )

        st.dataframe(
            feature_summary,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The compact clinical layer contains the raw patient "
            "fields used by the final model after categorical encoding."
        )

        # ====================================================
        # MODEL INTERPRETATION NOTICE
        # ====================================================

        st.info(
            "The displayed probabilities come from the trained final "
            "28-class classifier. They are model probabilities, not "
            "clinically calibrated diagnostic probabilities. In the "
            "accompanying paper the classifier reaches 97.20% Top-1 "
            "scenario-recovery consistency on a 1,000-case in-silico "
            "benchmark (28 literature-informed scenarios with noisy "
            "realizations). That figure measures internal model "
            "self-consistency, not clinical diagnostic accuracy, and the "
            "model has not been validated on real patient cohorts."
        )

else:

    st.info(
        "Complete the patient inputs in the sidebar, then select "
        "**Run 28-Class Model** to view the prediction and supporting analyses."
    )
