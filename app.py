# ============================================================
# app.py
# Cardio-Thermodynamic Digital Twin - FINAL 28-CLASS ARCHITECTURE
# Reported benchmark: 97.20% Top-1 scenario-recovery consistency on an in-silico
# benchmark; this is not clinical diagnostic accuracy.
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

# Python modules and trained artifacts may be stored beside app.py or in
# either spelling of the artifact folder used in prior repository revisions.
# The current repository uses pkl+joblibs; pkl+joblib remains supported for
# older deployments.
SEARCH_DIRS = [
    BASE_DIR,
    BASE_DIR / "pkl+joblibs",
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
    missing.append(
        f"disease_dses_rf.joblib (checked app root and both artifact-folder names; "
        f"Release download failed: {rf_error})"
    )

for filename in [
    "disease_dses_model_columns.joblib",
    "disease_dses_model_medians.joblib",
]:
    path = resolve_file(filename)
    if path is None:
        missing.append(
            f"{filename} (checked app root, pkl+joblibs/, and pkl+joblib/)"
        )
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
        missing.append(
            "final_28class_dses_classifier.joblib (checked app root and both "
            f"artifact-folder names; Release download failed: {classifier_error})"
        )
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
        #   /app.py + /pkl+joblibs/ or /pkl+joblib/{digital_twin.py, ml_final_28class_FIXED.py}
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
            page_title="Cardio-Thermodynamic Digital Twin",
            layout="wide",
        )
        st.title("Cardio-Thermodynamic Digital Twin")
        st.error(
            "The 28-class model runtime could not be loaded. "
            "Check the traceback and confirm that the Python modules and model artifacts "
            "belong to the same model version."
        )
        st.exception(exc)
        st.stop()
else:
    st.set_page_config(
        page_title="Cardio-Thermodynamic Digital Twin",
        layout="wide",
    )
    st.title("Cardio-Thermodynamic Digital Twin")

    st.error(
        "Required 28-class model files are missing. The app has stopped before "
        "running a prediction so it does not silently use an incomplete model."
    )

    st.write(
        "Expected Python modules and trained model artifacts either "
        "beside app.py or inside `pkl+joblibs/` or `pkl+joblib/`:"
    )

    for filename in missing:
        st.write(f"- `{filename}`")

    st.stop()


# ============================================================
# 4. PAGE CONFIG
# ============================================================

# set_page_config may already have been called above in an error
# branch. In normal execution, it has not been called yet.
try:
    st.set_page_config(
        page_title="Cardio-Thermodynamic Digital Twin",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# ============================================================
# 4A. PROFESSIONAL APPLICATION THEME
# ============================================================

st.markdown(
    """
    <style>
      :root {
        --cardio-navy: #10243A;
        --cardio-blue: #176B87;
        --cardio-teal: #1F9D8B;
        --cardio-muted: #627386;
        --cardio-border: #DCE5ED;
        --cardio-panel: #F5F8FB;
      }

      .stApp {
        background:
          radial-gradient(ellipse at top right, rgba(31,157,139,.07), transparent 34%),
          linear-gradient(180deg, #F8FAFC 0%, #F3F7FA 100%);
      }

      [data-testid="stHeader"] {
        background: rgba(248,250,252,.86);
      }

      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10243A 0%, #173650 100%);
        border-right: 1px solid rgba(255,255,255,.10);
      }

      [data-testid="stSidebar"] * {
        color: #F4F8FB;
      }

      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] input {
        background-color: rgba(255,255,255,.97);
        color: #10243A;
        border-radius: 9px;
      }

      [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #10243A;
      }

      [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,.20);
      }

      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 {
        letter-spacing: .02em;
      }

      .hero {
        padding: 1.35rem 1.55rem;
        margin: .25rem 0 1.25rem 0;
        border-radius: 18px;
        color: white;
        background: linear-gradient(120deg, #10243A 0%, #176B87 72%, #1F9D8B 100%);
        box-shadow: 0 10px 28px rgba(16,36,58,.14);
      }

      .hero-eyebrow {
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        opacity: .82;
        margin-bottom: .35rem;
      }

      .hero h1 {
        color: white;
        font-size: clamp(1.7rem, 3vw, 2.45rem);
        line-height: 1.15;
        margin: 0 0 .55rem 0;
        font-weight: 750;
      }

      .hero p {
        color: rgba(255,255,255,.88);
        margin: 0;
        font-size: .98rem;
      }

      h2, h3 {
        color: var(--cardio-navy);
        letter-spacing: -.015em;
      }

      [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--cardio-border);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        box-shadow: 0 3px 12px rgba(16,36,58,.045);
      }

      [data-testid="stMetricLabel"] {
        color: var(--cardio-muted);
        font-weight: 650;
      }

      [data-testid="stMetricValue"] {
        color: var(--cardio-navy);
        font-weight: 750;
      }

      [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
      }

      div[data-testid="stDataFrame"] {
        border: 1px solid var(--cardio-border);
        border-radius: 12px;
        overflow: hidden;
        background: white;
      }

      .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.8rem;
        transition: transform .15s ease, box-shadow .15s ease;
      }

      .stButton > button[kind="primary"] {
        background: linear-gradient(110deg, #176B87, #1F9D8B);
        border: 0;
        color: white;
        box-shadow: 0 5px 14px rgba(23,107,135,.22);
      }

      .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(16,36,58,.14);
      }

      [data-testid="stAlert"] {
        border-radius: 12px;
      }

      .section-note {
        color: var(--cardio-muted);
        font-size: .92rem;
        margin-top: -.55rem;
        margin-bottom: .85rem;
      }

      .app-footer {
        color: #718096;
        font-size: .78rem;
        border-top: 1px solid var(--cardio-border);
        margin-top: 2rem;
        padding-top: 1rem;
      }

      @media (prefers-reduced-motion: reduce) {
        * { transition: none !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="hero-eyebrow">Computational Cardiology · Research Prototype</div>
      <h1>Cardio-Thermodynamic Digital Twin</h1>
      <p>Disease-Specific Entropy Signature (DSES) analysis and 28-class model inference</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Model configuration: 28 candidate disease classes · Resting-state inputs · "
    "Smoking fixed to NO"
)

st.subheader("Reported In-Silico Benchmark")
st.markdown(
    '<p class="section-note">Metrics reported in the extended report for 1,000 noisy '
    'realizations from 28 literature-informed scenario families.</p>',
    unsafe_allow_html=True,
)

benchmark_columns = st.columns(4)
benchmark_columns[0].metric("Top-1 Scenario-Recovery Consistency", "97.20%")
benchmark_columns[1].metric("Balanced Accuracy", "97.18%")
benchmark_columns[2].metric("Macro-F1", "97.18%")
benchmark_columns[3].metric("Weighted F1", "97.19%")
st.info(
    "These are benchmark results reported for the constructed in-silico scenarios. "
    "The 97.20% result represents 972 correct scenario recoveries out of 1,000 "
    "noisy realizations. These metrics measure recovery of scenario families, not "
    "clinical diagnostic accuracy or validated probabilities for individual patients. "
    "Clinical validation on independently labelled patient cohorts remains necessary."
)
st.caption(
    "Reported 95% binomial confidence interval for Top-1 scenario-recovery consistency: "
    "approximately 96.2%–98.2%. This reflects the constructed benchmark, not a "
    "clinical patient-population validation interval."
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
    # Resolve the reference workbook relative to whichever supported folder
    # contains digital_twin.py, then fall back to BASE_DIR.
    dt_module_dir = resolved_modules["digital_twin.py"].parent

    workbook_name = "Cardiac_Disease_Synchronized_Dataset_Patient_1_Literature_Conditioned_CLEAN.xlsx"
    workbook_candidates = [
        # Canonical repository filename first.
        dt_module_dir / workbook_name,
        BASE_DIR / workbook_name,
        # Retain support for older downloaded copies with duplicate suffixes.
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

    st.header("Patient Input")

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

    st.divider()

    st.subheader("Hemodynamics")

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

    st.divider()

    st.subheader("Clinical Phenotype")

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

    run_button = st.button(
        "RUN 28-CLASS MODEL",
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
            if not isinstance(dt_result, dict) or "common" not in dt_result:
                raise ValueError(
                    "The Digital Twin returned an unexpected result structure; "
                    "the expected 'common' physiological state is missing."
                )

            final_predictions = predict_patient(
                patient,
                refs,
            )

            if not isinstance(final_predictions, pd.DataFrame):
                final_predictions = pd.DataFrame(final_predictions)
            required_prediction_columns = {"Disease", "Probability"}
            missing_prediction_columns = required_prediction_columns.difference(
                final_predictions.columns
            )
            if missing_prediction_columns:
                raise ValueError(
                    "The prediction function returned an unexpected schema. Missing: "
                    + ", ".join(sorted(missing_prediction_columns))
                )
            if final_predictions.empty:
                raise ValueError("The classifier returned no disease predictions.")

            final_predictions = final_predictions.copy()
            final_predictions["Probability"] = pd.to_numeric(
                final_predictions["Probability"], errors="coerce"
            )
            if (
                final_predictions["Probability"].isna().any()
                or not np.isfinite(final_predictions["Probability"].to_numpy()).all()
            ):
                raise ValueError(
                    "The classifier returned missing or non-finite probabilities."
                )
            final_predictions = final_predictions.sort_values(
                "Probability", ascending=False
            ).reset_index(drop=True)

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
            if not isinstance(final_features, pd.DataFrame):
                final_features = pd.DataFrame(final_features)
            if final_features.empty:
                raise ValueError("Feature construction returned an empty feature matrix.")
            if len(final_features) != 1:
                raise ValueError(
                    "Feature construction should return exactly one row for the "
                    "single patient entered."
                )

        except Exception as exc:
            st.error(
                "Model execution failed."
            )
            st.exception(exc)
            st.stop()

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

    # ========================================================
    # TOP METRICS
    # ========================================================

    st.header("Prediction Summary")
    st.markdown(
        '<p class="section-note">The leading class and derived cardiovascular metrics for the entered resting-state profile.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Predicted Disease",
        top_disease,
    )

    c2.metric(
        "Model Probability",
        f"{top_probability:.2%}",
    )

    c3.metric(
        "Stroke Volume",
        f"{common['SV']:.2f} mL",
    )

    c4.metric(
        "Ejection Fraction",
        f"{common['EF'] * 100:.2f}%",
    )

    c5.metric(
        "MAP",
        f"{common['MAP']:.2f} mmHg",
    )

    st.success(
        f"Top 28-class prediction: **{top_disease}** "
        f"with model probability **{top_probability:.2%}**."
    )

    # ========================================================
    # ARCHITECTURE PIPELINE
    # ========================================================

    st.header("Model Architecture")
    st.markdown(
        '<p class="section-note">Overview of the feature-construction and classification sequence.</p>',
        unsafe_allow_html=True,
    )

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
                ]
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
            ),
        )
    )

    pipeline.update_layout(
        title="Final Cardiac DSES Prediction Pipeline",
        height=400,
    )

    st.plotly_chart(
        pipeline,
        use_container_width=True,
    )

    # ========================================================
    # 28-CLASS RANKING TABLE
    # ========================================================

    st.header("Disease Ranking")
    st.markdown(
        '<p class="section-note">All candidate classes ordered by the final classifier output.</p>',
        unsafe_allow_html=True,
    )

    ranking = final_predictions.copy()

    ranking.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(ranking) + 1,
        ),
    )

    ranking["Probability"] = (
        ranking["Probability"] * 100.0
    )

    st.dataframe(
        ranking[
            [
                "Rank",
                "Disease",
                "Probability",
            ]
        ].style.format(
            {
                "Probability": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # TOP 10 CHART
    # ========================================================

    top10 = (
        ranking
        .head(10)
        .iloc[::-1]
    )

    fig_top10 = go.Figure(
        go.Bar(
            x=top10["Probability"],
            y=top10["Disease"],
            orientation="h",
        )
    )

    fig_top10.update_layout(
        title="Top 10 Final 28-Class Predictions",
        xaxis_title="Classifier Probability (%)",
        yaxis_title="Disease",
        height=520,
    )

    st.plotly_chart(
        fig_top10,
        use_container_width=True,
    )

    # ========================================================
    # PATIENT DSES / EXPECTED DSES / RESIDUALS
    # ========================================================

    st.header("DSES Feature Analysis")
    st.markdown(
        '<p class="section-note">Patient DSES, expected DSES, and residual values for each candidate disease.</p>',
        unsafe_allow_html=True,
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

    st.dataframe(
        dses_table.style.format(
            {
                "Patient DSES": "{:.3f}",
                "Expected DSES": "{:.3f}",
                "DSES Residual": "{:.3f}",
                "Probability": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # PATIENT VS EXPECTED DSES
    # ========================================================

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
        )
    )

    fig_dses.add_trace(
        go.Bar(
            x=selected_dses["Expected DSES"],
            y=selected_dses["Disease"],
            orientation="h",
            name="Expected DSES",
        )
    )

    fig_dses.update_layout(
        title="Patient DSES vs Expected DSES - Top 10",
        xaxis_title="DSES",
        yaxis_title="Disease",
        barmode="group",
        height=520,
    )

    st.plotly_chart(
        fig_dses,
        use_container_width=True,
    )

    # ========================================================
    # DSES RESIDUAL CHART
    # ========================================================

    residual_plot = (
        dses_table
        .head(15)
        .iloc[::-1]
    )

    fig_residual = go.Figure(
        go.Bar(
            x=residual_plot["DSES Residual"],
            y=residual_plot["Disease"],
            orientation="h",
        )
    )

    fig_residual.add_vline(
        x=0,
        line_dash="dash",
    )

    fig_residual.update_layout(
        title="DSES Residuals - Top 15 Predicted Diseases",
        xaxis_title="Patient DSES - Expected DSES",
        yaxis_title="Disease",
        height=620,
    )

    st.plotly_chart(
        fig_residual,
        use_container_width=True,
    )

    # ========================================================
    # GLOBAL STRESS FEATURES
    # ========================================================

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
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_stress = go.Figure(
        go.Bar(
            x=stress_df["Stress Component"],
            y=stress_df["Value"],
        )
    )

    fig_stress.add_hline(
        y=1.0,
        line_dash="dash",
        annotation_text="Reference = 1.0",
    )

    fig_stress.update_layout(
        title="Global Stress Profile",
        xaxis_title="Stress",
        yaxis_title="Normalized Value",
    )

    st.plotly_chart(
        fig_stress,
        use_container_width=True,
    )

    # ========================================================
    # DIGITAL TWIN PHYSIOLOGY
    # ========================================================

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
            common["ATP Production (mol/min)"],
            "mol/min",
        ),
        (
            "ATP Utilization",
            common["ATP Utilization (mol/min)"],
            "mol/min",
        ),
        (
            "ATP Utilization Fraction",
            common["ATP Utilization Fraction"],
            "ratio",
        ),
        (
            "ATP Balance",
            common["ATP Balance (mol/min)"],
            "mol/min",
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
            {"Value": "{:.6f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # REACTION-LEVEL DETAIL FOR TOP DISEASE
    # ========================================================

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

    # ========================================================
    # FULL DIGITAL-TWIN CALCULATION DETAILS
    # ========================================================

    with st.expander(
        "Show Full Digital-Twin Calculations"
    ):

        calculation = []

        for key, value in common.items():

            if isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):

                calculation.append(
                    {
                        "Variable": key,
                        "Value": float(value),
                    }
                )

        calculation_df = pd.DataFrame(
            calculation
        )

        if not calculation_df.empty:
            st.dataframe(
                calculation_df.style.format(
                    {"Value": "{:.8f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # MODEL FEATURE COUNT
    # ========================================================

    with st.expander(
        "Show Final Classifier Feature Summary"
    ):

        feature_summary = pd.DataFrame(
            {
                "Feature Group": [
                    "Patient DSES",
                    "Expected DSES",
                    "DSES residuals",
                    "Global stress summaries",
                    "DT physiological features",
                    "Compact clinical layer",
                    "Total final input features",
                ],
                "Feature Count": [
                    28,
                    28,
                    28,
                    5,
                    5,
                    12,
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

    # ========================================================
    # MODEL INTERPRETATION NOTICE
    # ========================================================

    st.info(
        "The displayed probabilities come from the trained final "
        "28-class classifier. They are model probabilities, not "
        "clinically calibrated diagnostic probabilities."
    )

else:

    st.info(
        "Enter the patient's resting parameters in the sidebar "
        "and click **RUN 28-CLASS MODEL**."
    )


st.markdown(
    """
    <div class="app-footer">
      <strong>Research-use prototype</strong> · Outputs are model-generated estimates,
      not a confirmed diagnosis. The reported in-silico benchmark metrics are not
      clinical diagnostic accuracy. Validate on independently labelled patient cohorts
      before making any clinical claims or decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
