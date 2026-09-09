"""
Predictive Maintenance Intelligence Platform - Streamlit Web Application
Enterprise-grade telemetry ingestion, failure risk prediction, physics-based root cause diagnostics, and maintenance dispatch.
"""

import io
import json
import math
import os
import random
import re
import joblib
import numpy as np
import pandas as pd
pd.set_option("styler.render.max_elements", 5000000)
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import streamlit as st

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
BENCHMARK_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")
UPLOADED_SAVE_PATH = os.path.join(DATA_DIR, "latest_uploaded.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="AI Predictive Maintenance Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .active-badge-uploaded {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid #38bdf8;
        color: #38bdf8;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }
    .active-badge-benchmark {
        background: linear-gradient(90deg, rgba(148, 163, 184, 0.1) 0%, rgba(71, 85, 105, 0.15) 100%);
        border: 1px solid #64748b;
        color: #94a3b8;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }
    .telemetry-calibration-badge {
        background: rgba(56, 189, 248, 0.08);
        border: 1px dashed rgba(56, 189, 248, 0.4);
        color: #7dd3fc;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 0.82rem;
        margin-bottom: 15px;
        line-height: 1.4;
    }
    .kpi-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        margin-bottom: 10px;
    }
    .kpi-card h4 {
        margin: 0;
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .kpi-card .kpi-value {
        margin: 8px 0 2px 0;
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .kpi-card .kpi-sub {
        margin: 0;
        font-size: 0.75rem;
        color: #64748b;
    }
    .sensor-stat-hint {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: -8px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# GLOBAL SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "dataset_source" not in st.session_state:
    st.session_state["dataset_source"] = "benchmark"  # 'benchmark' or 'uploaded'
if "uploaded_df" not in st.session_state:
    if os.path.exists(UPLOADED_SAVE_PATH):
        try:
            st.session_state["uploaded_df"] = pd.read_csv(UPLOADED_SAVE_PATH)
            st.session_state["uploaded_filename"] = "latest_uploaded.csv"
            st.session_state["dataset_source"] = "uploaded"
        except Exception:
            st.session_state["uploaded_df"] = None
            st.session_state["uploaded_filename"] = ""
    else:
        st.session_state["uploaded_df"] = None
        st.session_state["uploaded_filename"] = ""

# Manual schema overrides in session state
if "custom_col_map" not in st.session_state:
    st.session_state["custom_col_map"] = {}


@st.cache_resource
def load_trained_model():
    """Load the trained XGBoost bundle and optimal threshold metadata."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(THRESHOLD_PATH):
        return None, None, None
    bundle = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        meta = json.load(f)
    return bundle["model"], bundle["feature_cols"], meta


def normalize_str(s: str) -> str:
    """Normalize string for fuzzy column header matching."""
    return re.sub(r"[^a-zA-Z0-9]", "", str(s)).lower()


def auto_map_columns(df: pd.DataFrame) -> dict:
    """
    Intelligently map DataFrame columns to standard required telemetry schema.
    Handles case variations, punctuation, brackets, units, and synonyms.
    """
    norm_to_orig = {normalize_str(c): c for c in df.columns}

    synonyms = {
        "air_temp": [
            "airtemperaturek", "airtemperature", "airtempk", "airtemp",
            "ambienttemp", "ambienttemperature", "airtemperaturekelvin",
            "tempair", "tambient", "inlettemp", "tempambient", "airt"
        ],
        "proc_temp": [
            "processtemperaturek", "processtemperature", "processtempk",
            "processtemp", "temperature", "proctemp", "machinetemp",
            "tempprocess", "tprocess", "operatingtemp", "proct"
        ],
        "rot_speed": [
            "rotationalspeedrpm", "rotationalspeed", "speedrpm", "speed",
            "rpm", "spindlespeed", "rotationspeed", "rotationspeedrpm",
            "motorrpm", "spindlespeedrpm", "rotational"
        ],
        "torque": [
            "torquenm", "torque", "torquenewtonmeters", "loadtorque",
            "spindletorque", "motortorque", "torquen", "torq"
        ],
        "tool_wear": [
            "toolwearmin", "toolwear", "wearmin", "wear", "toolwearminutes",
            "accumulatedtoolwear", "cuttingswear", "tooltime", "weartime"
        ],
        "type": [
            "type", "producttype", "variant", "qualitytype", "machinetype",
            "productgrade", "qualityvariant", "grade", "modeltype"
        ],
        "unit_id": [
            "productid", "udi", "unitid", "machineid", "serialnumber",
            "id", "unitnumber", "assetid", "tag", "rowid", "deviceid"
        ],
        "target": [
            "machinefailure", "failure", "is_failure", "target", "label",
            "failed", "breakdown", "fault", "is_fault", "actual_failure"
        ],
    }

    mapping = {}

    for field, syn_list in synonyms.items():
        matched = None
        for syn in syn_list:
            if syn in norm_to_orig:
                matched = norm_to_orig[syn]
                break

        if matched is None:
            for norm_col, orig_col in norm_to_orig.items():
                if any(syn in norm_col for syn in syn_list if len(syn) >= 4):
                    matched = orig_col
                    break

        mapping[field] = matched

    return mapping


def diagnose_failure_mode(row: pd.Series) -> tuple[str, str, str]:
    """
    Physical domain rule engine diagnosing specific failure root cause and prescriptive actions.
    Based on industrial machining mechanics (Matzka, 2020).
    """
    torque = float(row.get("Torque [Nm]", 40.0))
    speed = float(row.get("Rotational speed [rpm]", 1500.0))
    air_temp = float(row.get("Air temperature [K]", 300.0))
    proc_temp = float(row.get("Process temperature [K]", 310.0))
    wear = float(row.get("Tool wear [min]", 100.0))
    m_type = str(row.get("Type", "M")).upper()

    power_w = torque * (speed * 2 * math.pi / 60)
    temp_diff = proc_temp - air_temp
    wear_torque_product = wear * torque

    osf_limit = 11000 if m_type == "L" else 12000 if m_type == "M" else 13000

    is_osf = wear_torque_product > osf_limit
    is_hdf = (temp_diff < 8.6) and (speed < 1380)
    is_pwf = (power_w < 3500) or (power_w > 9000)
    is_twf = wear >= 200

    if is_osf:
        return (
            "Overstrain Failure (OSF)",
            "High wear combined with excessive torque load exceeded structural yield envelope.",
            "P1 Urgent: Reduce feed torque, replace worn cutter head, and inspect drive train.",
        )
    elif is_hdf:
        return (
            "Heat Dissipation Failure (HDF)",
            "Insufficient temperature delta (<8.6K) at low spindle speed indicates thermal accumulation.",
            "P1 Urgent: Flush coolant heat exchanger, verify thermal dissipation fans and coolant delivery.",
        )
    elif is_pwf:
        return (
            "Power Delivery Failure (PWF)",
            f"Mechanical power delivery ({power_w:.0f} W) outside safe operational range [3500W - 9000W].",
            "P1 Urgent: Inspect drive motor inverter, check VFD power supply, and audit electrical line.",
        )
    elif is_twf:
        return (
            "Tool Wear Degradation (TWF)",
            f"Machining tool wear ({wear:.0f} min) reached end-of-life threshold (>=200 min).",
            "P2 High: Schedule immediate insert replacement prior to next production cycle.",
        )
    else:
        return (
            "Complex Stress / Degraded Operation",
            "Multi-sensor stress deviation approaching operational safety threshold.",
            "P3 Routine: Perform standard predictive telemetry inspection during shift turnover.",
        )


def prepare_features(df_clean: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Engineer features consistently matching the training pipeline."""
    df = df_clean.copy()

    # Derived physics features
    df["power_w"] = df["Torque [Nm]"] * (
        df["Rotational speed [rpm]"] * 2 * math.pi / 60
    )
    df["temp_diff_k"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["wear_per_torque"] = df["Tool wear [min]"] / (df["Torque [Nm]"] + 1e-3)

    # One-hot encode Type column
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], prefix="type")

    # Drop leakage / identifier columns if present
    drop_cols = [
        c for c in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure", "target"]
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    # Column name sanitize to match XGBoost format: () instead of []
    df.columns = (
        df.columns.str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", "lt", regex=False)
    )

    # Ensure all expected feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_cols].astype(float)


# Load model components
model, feature_cols, meta = load_trained_model()

# App Header
st.markdown('<div class="main-header">⚙️ Industrial Predictive Maintenance Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI Telemetry Diagnostics, Physics-Based Root Cause Analysis & Automated Maintenance Dispatch</div>',
    unsafe_allow_html=True,
)

if model is None:
    st.error("❌ Trained model artifact (`xgb_model.joblib`) not found! Please run training script first.")
    st.stop()

threshold = meta.get("threshold", 0.846)

# -------------------------------------------------------------
# STEP 1: SIDEBAR CONTROLS & UPLOAD HANDLING (RUNS FIRST)
# -------------------------------------------------------------
st.sidebar.title("🗄️ Telemetry Dataset Manager")

sidebar_upload = st.sidebar.file_uploader(
    "Upload New Telemetry CSV",
    type=["csv"],
    key="sidebar_uploader",
    help="Upload any factory sensor CSV to immediately score and analyze it across the platform.",
)

if sidebar_upload is not None and (
    st.session_state.get("uploaded_filename") != sidebar_upload.name
    or st.session_state.get("dataset_source") != "uploaded"
):
    try:
        new_df = pd.read_csv(sidebar_upload)
        st.session_state["uploaded_df"] = new_df
        st.session_state["uploaded_filename"] = sidebar_upload.name
        st.session_state["dataset_source"] = "uploaded"
        st.session_state["custom_col_map"] = {}  # reset column mapping for new dataset
        new_df.to_csv(UPLOADED_SAVE_PATH, index=False)
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to parse CSV: {e}")

dataset_options = ["Standard Benchmark (AI4I 2020 - 10k Units)"]
if st.session_state["uploaded_df"] is not None:
    dataset_options.append(f"📁 Custom Uploaded: {st.session_state['uploaded_filename']}")

current_idx = 1 if (st.session_state["dataset_source"] == "uploaded" and st.session_state["uploaded_df"] is not None) else 0

selected_source = st.sidebar.radio(
    "Active Telemetry Source",
    options=dataset_options,
    index=current_idx,
)

if "Custom Uploaded" in selected_source and st.session_state["uploaded_df"] is not None:
    if st.session_state["dataset_source"] != "uploaded":
        st.session_state["dataset_source"] = "uploaded"
        st.rerun()
else:
    if st.session_state["dataset_source"] != "benchmark":
        st.session_state["dataset_source"] = "benchmark"
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "📂 Fleet Telemetry & Batch Scoring",
        "⚡ Live Machine Telemetry",
        "📊 Model Analytics & Specs",
        "💡 System Architecture",
    ],
    key="main_navigation_radio",
)

# -------------------------------------------------------------
# STEP 2: LOAD ACTIVE DATASET & RESOLVE COLUMN MAPPINGS
# -------------------------------------------------------------
if st.session_state["dataset_source"] == "uploaded" and st.session_state["uploaded_df"] is not None:
    df_active = st.session_state["uploaded_df"]
    active_dataset_label = f"Custom Uploaded: {st.session_state['uploaded_filename']} ({len(df_active):,} records)"
    is_custom_dataset = True
else:
    if os.path.exists(BENCHMARK_PATH):
        df_active = pd.read_csv(BENCHMARK_PATH)
        active_dataset_label = f"Benchmark Fleet Telemetry (AI4I 2020 - 10,000 Units)"
        is_custom_dataset = False
    else:
        st.error("Benchmark dataset not found.")
        st.stop()

# Auto-map columns
detected_col_map = auto_map_columns(df_active)

# Merge with any user custom overrides stored in session state
active_col_map = {**detected_col_map, **st.session_state.get("custom_col_map", {})}

air_col = active_col_map.get("air_temp") or df_active.columns[0]
proc_col = active_col_map.get("proc_temp") or df_active.columns[0]
speed_col = active_col_map.get("rot_speed") or df_active.columns[0]
torque_col = active_col_map.get("torque") or df_active.columns[0]
wear_col = active_col_map.get("tool_wear") or df_active.columns[0]
type_col = active_col_map.get("type")
id_col = active_col_map.get("unit_id")
target_col = active_col_map.get("target")

# Standardize DataFrame with canonical names
df_clean = pd.DataFrame()
df_clean["Air temperature [K]"] = pd.to_numeric(df_active[air_col], errors="coerce").fillna(300.0)
df_clean["Process temperature [K]"] = pd.to_numeric(df_active[proc_col], errors="coerce").fillna(310.0)
df_clean["Rotational speed [rpm]"] = pd.to_numeric(df_active[speed_col], errors="coerce").fillna(1500.0)
df_clean["Torque [Nm]"] = pd.to_numeric(df_active[torque_col], errors="coerce").fillna(40.0)
df_clean["Tool wear [min]"] = pd.to_numeric(df_active[wear_col], errors="coerce").fillna(0.0)

if type_col and type_col in df_active.columns and type_col != "(Auto Default: 'M')":
    df_clean["Type"] = df_active[type_col].astype(str).str.strip().str.upper().apply(
        lambda x: x if x in ["L", "M", "H"] else "M"
    )
else:
    df_clean["Type"] = "M"

if id_col and id_col in df_active.columns and id_col != "(Auto Generated / Row Index)":
    unit_ids = df_active[id_col].astype(str)
else:
    unit_ids = pd.Series([f"UNIT-{i+1:05d}" for i in range(len(df_active))], name="Unit Identifier")

has_ground_truth = bool(target_col and target_col in df_active.columns and target_col != "(None / Unlabeled)")
if has_ground_truth:
    df_clean["Machine failure"] = pd.to_numeric(df_active[target_col], errors="coerce").fillna(0).astype(int)

# -------------------------------------------------------------
# STEP 3: EXECUTE INFERENCE & DIAGNOSTICS ON ACTIVE DATASET
# -------------------------------------------------------------
X_fleet = prepare_features(df_clean, feature_cols)
fleet_failure_probs = model.predict_proba(X_fleet)[:, 1]
fleet_is_flagged = fleet_failure_probs >= threshold
fleet_is_warning = (fleet_failure_probs >= 0.35) & (~fleet_is_flagged)

fleet_diagnostics = [diagnose_failure_mode(row) for _, row in df_clean.iterrows()]
fleet_failure_modes = [d[0] for d in fleet_diagnostics]
fleet_failure_reasons = [d[1] for d in fleet_diagnostics]
fleet_actions = [d[2] for d in fleet_diagnostics]

fleet_statuses = []
fleet_status_ranks = []
for p in fleet_failure_probs:
    if p >= threshold:
        fleet_statuses.append("CRITICAL")
        fleet_status_ranks.append(1)
    elif p >= 0.35:
        fleet_statuses.append("WARNING")
        fleet_status_ranks.append(2)
    else:
        fleet_statuses.append("HEALTHY")
        fleet_status_ranks.append(3)

results_df = pd.DataFrame(
    {
        "Unit Identifier": unit_ids,
        "Status": fleet_statuses,
        "Failure Risk": fleet_failure_probs,
        "Machine Type": df_clean["Type"],
        "Suspected Root Cause": fleet_failure_modes,
        "Tool Wear (min)": df_clean["Tool wear [min]"],
        "Torque (Nm)": df_clean["Torque [Nm]"],
        "Spindle Speed (rpm)": df_clean["Rotational speed [rpm]"],
        "Process Temp (K)": df_clean["Process temperature [K]"],
        "Air Temp (K)": df_clean["Air temperature [K]"],
        "Mechanical Power (W)": (
            df_clean["Torque [Nm]"] * (df_clean["Rotational speed [rpm]"] * 2 * math.pi / 60)
        ).round(1),
        "Temp Delta (K)": (df_clean["Process temperature [K]"] - df_clean["Air temperature [K]"]).round(2),
        "Diagnostic Finding": fleet_failure_reasons,
        "Prescriptive Maintenance Action": fleet_actions,
        "_rank": fleet_status_ranks,
    }
).sort_values(by=["_rank", "Failure Risk"], ascending=[True, False]).drop(columns=["_rank"])

# Dynamic dataset statistics
stat_air_min = max(0.0, float(np.floor(df_clean["Air temperature [K]"].min() - 2.0)))
stat_air_max = float(np.ceil(df_clean["Air temperature [K]"].max() + 2.0))
stat_air_med = float(df_clean["Air temperature [K]"].median())

stat_proc_min = max(0.0, float(np.floor(df_clean["Process temperature [K]"].min() - 2.0)))
stat_proc_max = float(np.ceil(df_clean["Process temperature [K]"].max() + 2.0))
stat_proc_med = float(df_clean["Process temperature [K]"].median())

stat_speed_min = max(100, int(np.floor(df_clean["Rotational speed [rpm]"].min() - 100)))
stat_speed_max = int(np.ceil(df_clean["Rotational speed [rpm]"].max() + 100))
stat_speed_med = int(df_clean["Rotational speed [rpm]"].median())

stat_torque_min = max(0.0, float(np.floor(df_clean["Torque [Nm]"].min() - 2.0)))
stat_torque_max = float(np.ceil(df_clean["Torque [Nm]"].max() + 5.0))
stat_torque_med = float(df_clean["Torque [Nm]"].median())

stat_wear_min = 0
stat_wear_max = max(250, int(np.ceil(df_clean["Tool wear [min]"].max() + 20)))
stat_wear_med = int(df_clean["Tool wear [min]"].median())

available_types = sorted(list(df_clean["Type"].unique()))
if not available_types:
    available_types = ["L", "M", "H"]

# Sidebar dynamic specs display
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Dynamic Model Specs")
st.sidebar.markdown(f"**Operating Decision Threshold:** `{threshold:.4f}`")

if has_ground_truth:
    live_y_true = df_clean["Machine failure"]
    live_y_pred = (fleet_failure_probs >= threshold).astype(int)
    live_prec = precision_score(live_y_true, live_y_pred, zero_division=0)
    live_rec = recall_score(live_y_true, live_y_pred, zero_division=0)
    live_f1 = f1_score(live_y_true, live_y_pred, zero_division=0)
    live_pr_auc = (
        average_precision_score(live_y_true, fleet_failure_probs)
        if len(np.unique(live_y_true)) > 1
        else 1.0
    )
    st.sidebar.markdown(f"**Active Dataset PR-AUC:** `{live_pr_auc:.3f}`")
    st.sidebar.markdown(f"**Active Dataset Precision:** `{live_prec*100:.1f}%`")
    st.sidebar.markdown(f"**Active Dataset Recall:** `{live_rec*100:.1f}%`")
    st.sidebar.markdown(f"**Active Dataset F1 Score:** `{live_f1:.3f}`")
else:
    flag_count = int(fleet_is_flagged.sum())
    flag_rate = (flag_count / len(results_df)) * 100 if len(results_df) > 0 else 0
    st.sidebar.markdown(f"**Active Fleet Flag Rate:** `{flag_rate:.2f}%`")
    st.sidebar.markdown(f"**Mean Fleet Risk:** `{fleet_failure_probs.mean()*100:.2f}%`")
    st.sidebar.markdown(f"**Benchmark PR-AUC:** `{meta.get('pr_auc', 0.888):.3f}`")


# ==========================================
# PAGE 1: FLEET TELEMETRY & BATCH SCORING
# ==========================================
if page == "📂 Fleet Telemetry & Batch Scoring":
    st.subheader("📂 Fleet Telemetry Ingestion & Intelligent Diagnostics")
    st.write(
        "Upload factory telemetry CSV or inspect the active dataset to automatically calculate KPIs, generate interactive visual reports, and dispatch preventative maintenance work orders."
    )

    if is_custom_dataset:
        st.markdown(
            f"""
            <div class="active-badge-uploaded">
                <span>🟢</span>
                <strong>ACTIVE DATASET:</strong> {active_dataset_label}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="active-badge-benchmark">
                <span>⚙️</span>
                <strong>ACTIVE DATASET:</strong> {active_dataset_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Page-level Upload / Reset Box
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        page_upload = st.file_uploader(
            "Drop Telemetry CSV here to load into project",
            type=["csv"],
            key="main_page_uploader",
            help="Upload custom telemetry CSV. Automatically sets this as the active project dataset.",
        )
        if page_upload is not None and (
            st.session_state.get("uploaded_filename") != page_upload.name
            or st.session_state.get("dataset_source") != "uploaded"
        ):
            try:
                new_df = pd.read_csv(page_upload)
                st.session_state["uploaded_df"] = new_df
                st.session_state["uploaded_filename"] = page_upload.name
                st.session_state["dataset_source"] = "uploaded"
                st.session_state["custom_col_map"] = {}
                new_df.to_csv(UPLOADED_SAVE_PATH, index=False)
                st.rerun()
            except Exception as e:
                st.error(f"Error loading uploaded file: {e}")

    with col_up2:
        if is_custom_dataset:
            if st.button("🔄 Reset to Benchmark Data", help="Switch back to standard 10,000-unit benchmark dataset"):
                st.session_state["dataset_source"] = "benchmark"
                st.rerun()
        else:
            st.info("Currently viewing benchmark dataset.")

    # Telemetry Auto-Adjustment Summary Banner
    st.markdown(
        f"""
        <div class="telemetry-calibration-badge">
            <strong>✨ Telemetry Dynamic Calibration:</strong> Sensor channels, operating bounds, and diagnostics are automatically synchronized to <strong>{active_dataset_label}</strong>.<br>
            • Air Temp Range: <code>{stat_air_min:.1f}K – {stat_air_max:.1f}K</code> &nbsp;|&nbsp;
            • Process Temp Range: <code>{stat_proc_min:.1f}K – {stat_proc_max:.1f}K</code> &nbsp;|&nbsp;
            • Spindle Speed Range: <code>{stat_speed_min:,} – {stat_speed_max:,} rpm</code> &nbsp;|&nbsp;
            • Torque Range: <code>{stat_torque_min:.1f} – {stat_torque_max:.1f} Nm</code> &nbsp;|&nbsp;
            • Max Wear: <code>{stat_wear_max} min</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SCHEMA MAPPING & VALIDATION (WITH CALLBACKS/RERUN ON CHANGE)
    required_keys = ["air_temp", "proc_temp", "rot_speed", "torque", "tool_wear"]
    missing_keys = [k for k in required_keys if active_col_map.get(k) is None]

    with st.expander("🛠️ Telemetry Schema Mapping & Data Quality Inspector", expanded=bool(missing_keys)):
        if missing_keys:
            st.warning(f"⚠️ Some standard channels could not be auto-mapped ({', '.join(missing_keys)}). Please select corresponding columns below:")
        else:
            st.markdown("✅ **All required telemetry channels automatically matched.**")

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            sel_air = st.selectbox("Air Temperature Channel", options=list(df_active.columns), index=list(df_active.columns).index(air_col) if air_col in df_active.columns else 0)
            sel_proc = st.selectbox("Process Temperature Channel", options=list(df_active.columns), index=list(df_active.columns).index(proc_col) if proc_col in df_active.columns else 0)
        with mcol2:
            sel_speed = st.selectbox("Rotational Speed Channel", options=list(df_active.columns), index=list(df_active.columns).index(speed_col) if speed_col in df_active.columns else 0)
            sel_torque = st.selectbox("Torque Channel", options=list(df_active.columns), index=list(df_active.columns).index(torque_col) if torque_col in df_active.columns else 0)
        with mcol3:
            sel_wear = st.selectbox("Tool Wear Channel", options=list(df_active.columns), index=list(df_active.columns).index(wear_col) if wear_col in df_active.columns else 0)
            type_opts = ["(Auto Default: 'M')"] + list(df_active.columns)
            type_idx = type_opts.index(type_col) if type_col in df_active.columns else 0
            sel_type = st.selectbox("Machine Type / Variant Channel", options=type_opts, index=type_idx)
        with mcol4:
            id_opts = ["(Auto Generated / Row Index)"] + list(df_active.columns)
            id_idx = id_opts.index(id_col) if id_col in df_active.columns else 0
            sel_id = st.selectbox("Unit / Machine ID Channel", options=id_opts, index=id_idx)

            target_opts = ["(None / Unlabeled)"] + list(df_active.columns)
            target_idx = target_opts.index(target_col) if target_col in df_active.columns else 0
            sel_target = st.selectbox("Ground Truth Failure Channel (Optional)", options=target_opts, index=target_idx)

        # Check if user updated schema mapping
        updated_map = {
            "air_temp": sel_air,
            "proc_temp": sel_proc,
            "rot_speed": sel_speed,
            "torque": sel_torque,
            "tool_wear": sel_wear,
            "type": sel_type,
            "unit_id": sel_id,
            "target": sel_target,
        }
        if updated_map != st.session_state.get("custom_col_map"):
            st.session_state["custom_col_map"] = updated_map
            st.rerun()

    # ==========================================
    # EXECUTIVE KPI DASHBOARD
    # ==========================================
    total_units = len(results_df)
    critical_count = int(fleet_is_flagged.sum())
    warning_count = int(fleet_is_warning.sum())
    healthy_count = total_units - critical_count - warning_count
    flag_rate = (critical_count / total_units) * 100 if total_units > 0 else 0
    fleet_health_score = max(0.0, 100.0 - (critical_count * 2.5 + warning_count * 0.8) / total_units * 100)
    est_savings = critical_count * 15000

    st.markdown("### 📊 Executive Telemetry & Reliability KPIs")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <h4>Total Machines</h4>
                <div class="kpi-value">{total_units:,}</div>
                <p class="kpi-sub">Units Scored</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-color: rgba(248, 113, 113, 0.4);">
                <h4 style="color: #f87171;">Critical Flagged</h4>
                <div class="kpi-value" style="color: #f87171;">{critical_count:,}</div>
                <p class="kpi-sub">{flag_rate:.2f}% of fleet</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-color: rgba(251, 191, 36, 0.4);">
                <h4 style="color: #fbbf24;">Elevated Warning</h4>
                <div class="kpi-value" style="color: #fbbf24;">{warning_count:,}</div>
                <p class="kpi-sub">{(warning_count/total_units)*100:.2f}% of fleet</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-color: rgba(52, 211, 153, 0.4);">
                <h4 style="color: #34d399;">Nominal / Healthy</h4>
                <div class="kpi-value" style="color: #34d399;">{healthy_count:,}</div>
                <p class="kpi-sub">{(healthy_count/total_units)*100:.2f}% of fleet</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <h4>Fleet Health Index</h4>
                <div class="kpi-value" style="color: {'#34d399' if fleet_health_score >= 85 else '#fbbf24' if fleet_health_score >= 70 else '#f87171'};">
                    {fleet_health_score:.1f}%
                </div>
                <p class="kpi-sub">Composite Reliability</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <h4>Est. Loss Avoided</h4>
                <div class="kpi-value" style="color: #38bdf8;">${est_savings:,.0f}</div>
                <p class="kpi-sub">@ $15k / prevented failure</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ==========================================
    # INTERACTIVE VISUAL REPORTS & ANALYTICS
    # ==========================================
    st.markdown("### 📈 Interactive Telemetry Reports & Visual Analytics")

    tab_vis1, tab_vis2, tab_vis3, tab_vis4 = st.tabs(
        [
            "📊 Fleet Risk Distribution",
            "🔬 Root Cause & Failure Modes",
            "🛠️ Operational Physics & Boundaries",
            "🎯 Model Validation on Uploaded Batch",
        ]
    )

    with tab_vis1:
        r_col1, r_col2 = st.columns([1.5, 1])

        with r_col1:
            fig_hist = px.histogram(
                results_df,
                x="Failure Risk",
                nbins=50,
                color="Status",
                color_discrete_map={"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "HEALTHY": "#10b981"},
                title=f"Predicted Failure Risk Distribution ({active_dataset_label})",
                labels={"Failure Risk": "Predicted Probability of Failure", "count": "Machine Count"},
                template="plotly_dark",
            )
            fig_hist.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="#38bdf8",
                annotation_text=f"Decision Threshold ({threshold:.3f})",
                annotation_position="top right",
            )
            fig_hist.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=360,
            )
            st.plotly_chart(fig_hist, width="stretch")

        with r_col2:
            type_summary = results_df.groupby(["Machine Type", "Status"]).size().reset_index(name="Count")
            fig_type = px.bar(
                type_summary,
                x="Machine Type",
                y="Count",
                color="Status",
                color_discrete_map={"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "HEALTHY": "#10b981"},
                title=f"Risk Breakdown by Machine Variant ({len(results_df):,} Units)",
                template="plotly_dark",
                barmode="stack",
            )
            fig_type.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=360,
            )
            st.plotly_chart(fig_type, width="stretch")

    with tab_vis2:
        cause_col1, cause_col2 = st.columns([1, 1])

        with cause_col1:
            flagged_df = results_df[results_df["Status"].isin(["CRITICAL", "WARNING"])]
            if len(flagged_df) > 0:
                cause_counts = flagged_df["Suspected Root Cause"].value_counts().reset_index()
                cause_counts.columns = ["Root Cause", "Count"]

                fig_pie = px.pie(
                    cause_counts,
                    names="Root Cause",
                    values="Count",
                    title=f"Primary Root-Cause Diagnostics ({len(flagged_df)} At-Risk Units in {active_dataset_label.split('(')[0].strip()})",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.45,
                    template="plotly_dark",
                )
                fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=360)
                st.plotly_chart(fig_pie, width="stretch")
            else:
                st.info("No units flagged at risk in this dataset.")

        with cause_col2:
            metric_cols = ["Tool Wear (min)", "Torque (Nm)", "Temp Delta (K)", "Mechanical Power (W)"]
            grouped_metrics = results_df.groupby("Status")[metric_cols].mean().reset_index()
            melted = grouped_metrics.melt(id_vars="Status", var_name="Telemetry Channel", value_name="Mean Value")
            fig_stress = px.bar(
                melted,
                x="Telemetry Channel",
                y="Mean Value",
                color="Status",
                color_discrete_map={"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "HEALTHY": "#10b981"},
                barmode="group",
                title=f"Comparative Sensor Stress Profiles by Health Status ({active_dataset_label.split('(')[0].strip()})",
                template="plotly_dark",
            )
            fig_stress.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=360)
            st.plotly_chart(fig_stress, width="stretch")

    with tab_vis3:
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            plot_sample = results_df.sample(min(3000, len(results_df)), random_state=42)
            fig_scatter = px.scatter(
                plot_sample,
                x="Tool Wear (min)",
                y="Torque (Nm)",
                color="Failure Risk",
                color_continuous_scale="Turbo",
                hover_data=["Unit Identifier", "Status", "Suspected Root Cause"],
                title=f"Operational Stress Envelope ({len(plot_sample):,} sampled units from active data)",
                template="plotly_dark",
            )
            wear_curve = np.linspace(max(10, stat_wear_min), max(stat_wear_max, 50), 100)
            torque_curve = 12000 / (wear_curve + 1e-3)
            fig_scatter.add_trace(
                go.Scatter(
                    x=wear_curve,
                    y=torque_curve,
                    mode="lines",
                    name="OSF Yield Limit (Type M)",
                    line=dict(color="#f87171", dash="dot", width=2),
                )
            )
            fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=380)
            st.plotly_chart(fig_scatter, width="stretch")

        with p_col2:
            fig_therm = px.scatter(
                plot_sample,
                x="Spindle Speed (rpm)",
                y="Temp Delta (K)",
                color="Failure Risk",
                color_continuous_scale="Turbo",
                hover_data=["Unit Identifier", "Status", "Suspected Root Cause"],
                title=f"Thermal Dissipation vs Speed ({len(plot_sample):,} sampled units)",
                template="plotly_dark",
            )
            fig_therm.add_shape(
                type="rect",
                x0=stat_speed_min,
                x1=1380,
                y0=0,
                y1=8.6,
                line=dict(color="#f87171", width=1, dash="dash"),
                fillcolor="rgba(239, 68, 68, 0.15)",
            )
            fig_therm.add_annotation(
                x=min(1250, stat_speed_min + 150),
                y=6.0,
                text="HDF Danger Zone",
                showarrow=False,
                font=dict(color="#f87171", size=11),
            )
            fig_therm.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=380)
            st.plotly_chart(fig_therm, width="stretch")

    with tab_vis4:
        if has_ground_truth:
            st.markdown(f"#### 🎯 Ground Truth Validation on: **{active_dataset_label}**")
            y_true = df_clean["Machine failure"]
            y_pred = (fleet_failure_probs >= threshold).astype(int)

            val_acc = accuracy_score(y_true, y_pred)
            val_prec = precision_score(y_true, y_pred, zero_division=0)
            val_rec = recall_score(y_true, y_pred, zero_division=0)
            val_f1 = f1_score(y_true, y_pred, zero_division=0)
            val_auc = roc_auc_score(y_true, fleet_failure_probs) if len(np.unique(y_true)) > 1 else 1.0

            vc1, vc2, vc3, vc4, vc5 = st.columns(5)
            with vc1:
                st.metric("Batch Accuracy", f"{val_acc*100:.2f}%")
            with vc2:
                st.metric("Batch Precision", f"{val_prec*100:.2f}%", help="True Failures / Total Flagged")
            with vc3:
                st.metric("Batch Recall", f"{val_rec*100:.2f}%", help="Catches % of real breakdowns")
            with vc4:
                st.metric("Batch F1-Score", f"{val_f1:.3f}")
            with vc5:
                st.metric("Batch ROC-AUC", f"{val_auc:.3f}")

            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Healthy (0)", "Actual Failure (1)"],
                columns=["Predicted Healthy (0)", "Predicted Flagged (1)"],
            )

            cm_col1, cm_col2 = st.columns([1, 1.2])
            with cm_col1:
                fig_cm = px.imshow(
                    cm_df,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title=f"Confusion Matrix ({active_dataset_label.split('(')[0].strip()})",
                    template="plotly_dark",
                )
                fig_cm.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig_cm, width="stretch")

            with cm_col2:
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.6); padding: 18px; border-radius: 12px; margin-top: 20px;">
                        <h4 style="margin-top: 0;">Operational Impact Summary</h4>
                        <ul>
                            <li><strong style="color: #34d399;">True Positives (Breakdowns Prevented):</strong> {tp} machines</li>
                            <li><strong style="color: #94a3b8;">True Negatives (Healthy Uninterrupted):</strong> {tn} machines</li>
                            <li><strong style="color: #fbbf24;">False Positives (Nuisance Inspections):</strong> {fp} machines</li>
                            <li><strong style="color: #f87171;">False Negatives (Missed Breakdowns):</strong> {fn} machines</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "💡 No Ground-Truth failure label channel detected in this dataset. To evaluate Precision/Recall metrics on custom data, map a ground-truth column in the 'Schema Mapping' expander above."
            )

    st.markdown("---")

    # ==========================================
    # ACTIONABLE WORK-ORDER DISPATCH CENTER
    # ==========================================
    st.markdown("### 🛠️ Actionable Maintenance Work-Order Dispatch Center")

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 1.2, 1.5, 2])
    with f_col1:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All Statuses", "CRITICAL Only", "WARNING & CRITICAL", "HEALTHY Only"],
            index=0,
        )
    with f_col2:
        type_filter = st.selectbox("Machine Type", options=["All Types"] + available_types, index=0)
    with f_col3:
        cause_opts = ["All Diagnosed Causes"] + sorted(list(results_df["Suspected Root Cause"].unique()))
        cause_filter = st.selectbox("Suspected Failure Cause", options=cause_opts, index=0)
    with f_col4:
        search_query = st.text_input("🔍 Search Unit Identifier", "")

    display_df = results_df.copy()
    if status_filter == "CRITICAL Only":
        display_df = display_df[display_df["Status"] == "CRITICAL"]
    elif status_filter == "WARNING & CRITICAL":
        display_df = display_df[display_df["Status"].isin(["CRITICAL", "WARNING"])]
    elif status_filter == "HEALTHY Only":
        display_df = display_df[display_df["Status"] == "HEALTHY"]

    if type_filter != "All Types":
        display_df = display_df[display_df["Machine Type"] == type_filter]

    if cause_filter != "All Diagnosed Causes":
        display_df = display_df[display_df["Suspected Root Cause"] == cause_filter]

    if search_query.strip():
        display_df = display_df[
            display_df["Unit Identifier"].astype(str).str.contains(search_query.strip(), case=False)
        ]

    st.markdown(f"**Showing {len(display_df):,} machine record(s)**")

    column_config = {
        "Failure Risk": st.column_config.ProgressColumn(
            "Failure Risk",
            help="Predicted failure probability by Imbalance-Aware XGBoost",
            format="%.2f%%",
            min_value=0.0,
            max_value=1.0,
        ),
        "Status": st.column_config.TextColumn(
            "Status",
            help="Health status: CRITICAL / WARNING / HEALTHY",
        ),
        "Tool Wear (min)": st.column_config.NumberColumn(
            "Tool Wear (min)",
            format="%d min",
        ),
        "Torque (Nm)": st.column_config.NumberColumn(
            "Torque (Nm)",
            format="%.1f Nm",
        ),
        "Spindle Speed (rpm)": st.column_config.NumberColumn(
            "Spindle Speed (rpm)",
            format="%d rpm",
        ),
        "Process Temp (K)": st.column_config.NumberColumn(
            "Process Temp (K)",
            format="%.1f K",
        ),
        "Air Temp (K)": st.column_config.NumberColumn(
            "Air Temp (K)",
            format="%.1f K",
        ),
        "Mechanical Power (W)": st.column_config.NumberColumn(
            "Mechanical Power (W)",
            format="%.1f W",
        ),
        "Temp Delta (K)": st.column_config.NumberColumn(
            "Temp Delta (K)",
            format="%.2f K",
        ),
    }

    st.dataframe(display_df, column_config=column_config, width="stretch", height=440)

    # ==========================================
    # REPORT EXPORT CENTER
    # ==========================================
    st.markdown("### 📥 Report Export & Dispatch Downloads")

    d_col1, d_col2, d_col3 = st.columns(3)

    with d_col1:
        csv_full = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete Scored Telemetry (CSV)",
            data=csv_full,
            file_name=f"fleet_scored_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download all rows with risk scores and diagnosed actions",
        )

    with d_col2:
        critical_csv = results_df[results_df["Status"] == "CRITICAL"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="🚨 Download Priority Work Orders (CSV)",
            data=critical_csv,
            file_name=f"priority_work_orders_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download only flagged critical units requiring immediate maintenance",
        )

    with d_col3:
        briefing_text = f"""# Executive Predictive Maintenance Fleet Telemetry Briefing
**Generated At:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Source Dataset:** {active_dataset_label}

## 1. Executive Summary & KPIs
- **Total Machines Monitored:** {total_units:,}
- **Critical Flagged Units:** {critical_count:,} ({flag_rate:.2f}%)
- **Elevated Warning Units:** {warning_count:,} ({(warning_count/total_units)*100:.2f}%)
- **Healthy Units:** {healthy_count:,} ({(healthy_count/total_units)*100:.2f}%)
- **Fleet Composite Health Score:** {fleet_health_score:.1f}%
- **Estimated Unplanned Downtime Cost Avoided:** ${est_savings:,.0f} (at $15,000 / prevented catastrophic breakdown)

## 2. High-Priority Dispatch Units (Top 10 Critical Risk)
"""
        top_critical = results_df[results_df["Status"] == "CRITICAL"].head(10)
        for _, r in top_critical.iterrows():
            briefing_text += f"\n- **{r['Unit Identifier']}** | Risk: {r['Failure Risk']:.2%} | Type: {r['Machine Type']} | Root Cause: {r['Suspected Root Cause']}\n  - *Action:* {r['Prescriptive Maintenance Action']}\n"

        st.download_button(
            label="📄 Download Executive Briefing (Markdown)",
            data=briefing_text.encode("utf-8"),
            file_name=f"executive_maintenance_briefing_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            help="Download structured executive briefing report",
        )


# ==========================================
# PAGE 2: LIVE MACHINE TELEMETRY (DYNAMICALLY ADJUSTED)
# ==========================================
elif page == "⚡ Live Machine Telemetry":
    st.subheader("⚡ Real-Time Single Machine Telemetry Assessment")
    st.write("Simulate real-time sensor parameters to predict breakdown probability and inspect physics-based failure modes.")

    st.markdown(
        f"""
        <div class="telemetry-calibration-badge">
            <strong>✨ Telemetry Sliders Auto-Calibrated to:</strong> {active_dataset_label}<br>
            All slider limits, step intervals, and defaults are dynamically fitted to the real empirical sensor distributions of your active dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📥 Preload Real Telemetry from Active Dataset", expanded=True):
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)

        with p_col1:
            if st.button("🟢 Typical Healthy (Median)", help="Load dataset median values across all sensors"):
                st.session_state["preset_type"] = available_types[0] if available_types else "M"
                st.session_state["preset_air"] = stat_air_med
                st.session_state["preset_proc"] = stat_proc_med
                st.session_state["preset_speed"] = stat_speed_med
                st.session_state["preset_torque"] = stat_torque_med
                st.session_state["preset_wear"] = stat_wear_med
                st.rerun()

        with p_col2:
            if st.button("⚠️ High Stress (P90 Load)", help="Load high torque, high tool wear, and speed stress"):
                st.session_state["preset_type"] = "L" if "L" in available_types else available_types[0]
                st.session_state["preset_air"] = float(df_clean["Air temperature [K]"].quantile(0.90))
                st.session_state["preset_proc"] = float(df_clean["Process temperature [K]"].quantile(0.90))
                st.session_state["preset_speed"] = int(df_clean["Rotational speed [rpm]"].quantile(0.15))
                st.session_state["preset_torque"] = float(df_clean["Torque [Nm]"].quantile(0.90))
                st.session_state["preset_wear"] = int(df_clean["Tool wear [min]"].quantile(0.90))
                st.rerun()

        with p_col3:
            if st.button("🚨 Top Critical Machine", help="Load the highest-risk machine identified in this dataset"):
                crit_idx = results_df.index[0]
                orig_crit = df_clean.loc[crit_idx]
                st.session_state["preset_type"] = str(orig_crit["Type"]).upper()
                st.session_state["preset_air"] = float(orig_crit["Air temperature [K]"])
                st.session_state["preset_proc"] = float(orig_crit["Process temperature [K]"])
                st.session_state["preset_speed"] = int(orig_crit["Rotational speed [rpm]"])
                st.session_state["preset_torque"] = float(orig_crit["Torque [Nm]"])
                st.session_state["preset_wear"] = int(orig_crit["Tool wear [min]"])
                st.rerun()

        with p_col4:
            if st.button("🎲 Random Dataset Unit", help="Sample a random machine from the active dataset"):
                rand_idx = random.choice(list(df_clean.index))
                rand_row = df_clean.loc[rand_idx]
                st.session_state["preset_type"] = str(rand_row["Type"]).upper()
                st.session_state["preset_air"] = float(rand_row["Air temperature [K]"])
                st.session_state["preset_proc"] = float(rand_row["Process temperature [K]"])
                st.session_state["preset_speed"] = int(rand_row["Rotational speed [rpm]"])
                st.session_state["preset_torque"] = float(rand_row["Torque [Nm]"])
                st.session_state["preset_wear"] = int(rand_row["Tool wear [min]"])
                st.rerun()

        st.markdown("---")
        u_col1, u_col2 = st.columns([3, 1])
        with u_col1:
            unit_choices = results_df.head(200).apply(
                lambda r: f"{r['Unit Identifier']} | Risk: {r['Failure Risk']:.1%} | {r['Status']} ({r['Suspected Root Cause']})",
                axis=1,
            ).tolist()
            selected_choice = st.selectbox("Or Select Specific Machine Record to Inspect:", options=unit_choices)
        with u_col2:
            st.write("")
            st.write("")
            if st.button("Load Selected Unit Telemetry"):
                sel_unit_id = selected_choice.split(" | ")[0]
                matched_idx = results_df[results_df["Unit Identifier"] == sel_unit_id].index[0]
                matched_row = df_clean.loc[matched_idx]
                st.session_state["preset_type"] = str(matched_row["Type"]).upper()
                st.session_state["preset_air"] = float(matched_row["Air temperature [K]"])
                st.session_state["preset_proc"] = float(matched_row["Process temperature [K]"])
                st.session_state["preset_speed"] = int(matched_row["Rotational speed [rpm]"])
                st.session_state["preset_torque"] = float(matched_row["Torque [Nm]"])
                st.session_state["preset_wear"] = int(matched_row["Tool wear [min]"])
                st.rerun()

    cur_air = float(np.clip(st.session_state.get("preset_air", stat_air_med), stat_air_min, stat_air_max))
    cur_proc = float(np.clip(st.session_state.get("preset_proc", stat_proc_med), stat_proc_min, stat_proc_max))
    cur_speed = int(np.clip(st.session_state.get("preset_speed", stat_speed_med), stat_speed_min, stat_speed_max))
    cur_torque = float(np.clip(st.session_state.get("preset_torque", stat_torque_med), stat_torque_min, stat_torque_max))
    cur_wear = int(np.clip(st.session_state.get("preset_wear", stat_wear_med), stat_wear_min, stat_wear_max))
    preset_t = st.session_state.get("preset_type", available_types[0])
    cur_type_idx = available_types.index(preset_t) if preset_t in available_types else 0

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 🎛️ Live Sensor Inputs (Calibrated to Dataset)")

        machine_type = st.selectbox(
            "Product Quality Variant (Type)",
            options=available_types,
            index=cur_type_idx,
            help="Quality variant category present in the dataset",
        )

        air_temp = st.slider(
            "Air Temperature [K]",
            min_value=stat_air_min,
            max_value=stat_air_max,
            value=cur_air,
            step=0.1,
            help="Ambient room / factory floor temperature",
        )
        st.markdown(f'<div class="sensor-stat-hint">Dataset Range: {stat_air_min:.1f}K – {stat_air_max:.1f}K | Median: {stat_air_med:.1f}K</div>', unsafe_allow_html=True)

        process_temp = st.slider(
            "Process Temperature [K]",
            min_value=stat_proc_min,
            max_value=stat_proc_max,
            value=cur_proc,
            step=0.1,
            help="Operational temperature measured at the workpiece / spindle",
        )
        st.markdown(f'<div class="sensor-stat-hint">Dataset Range: {stat_proc_min:.1f}K – {stat_proc_max:.1f}K | Median: {stat_proc_med:.1f}K</div>', unsafe_allow_html=True)

        rot_speed = st.slider(
            "Rotational Speed [rpm]",
            min_value=stat_speed_min,
            max_value=stat_speed_max,
            value=cur_speed,
            step=10,
            help="Spindle rotational velocity",
        )
        st.markdown(f'<div class="sensor-stat-hint">Dataset Range: {stat_speed_min:,} – {stat_speed_max:,} rpm | Median: {stat_speed_med:,} rpm</div>', unsafe_allow_html=True)

        torque = st.slider(
            "Torque [Nm]",
            min_value=stat_torque_min,
            max_value=stat_torque_max,
            value=cur_torque,
            step=0.1,
            help="Spindle drive torque",
        )
        st.markdown(f'<div class="sensor-stat-hint">Dataset Range: {stat_torque_min:.1f} – {stat_torque_max:.1f} Nm | Median: {stat_torque_med:.1f} Nm</div>', unsafe_allow_html=True)

        tool_wear = st.slider(
            "Tool Wear [min]",
            min_value=stat_wear_min,
            max_value=stat_wear_max,
            value=cur_wear,
            step=1,
            help="Accumulated cutting tool contact time",
        )
        st.markdown(f'<div class="sensor-stat-hint">Dataset Range: {stat_wear_min} – {stat_wear_max} min | Median: {stat_wear_med} min</div>', unsafe_allow_html=True)

    power_w = torque * (rot_speed * 2 * math.pi / 60)
    temp_diff = process_temp - air_temp
    wear_per_torque = tool_wear / (torque + 1e-3)

    single_df = pd.DataFrame(
        [
            {
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rot_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ]
    )

    X_single = prepare_features(single_df, feature_cols)
    failure_prob = float(model.predict_proba(X_single)[0, 1])
    is_flagged = failure_prob >= threshold

    cause_name, cause_desc, action_desc = diagnose_failure_mode(single_df.iloc[0])

    with col2:
        st.markdown("#### 🩺 Diagnostic Assessment Verdict")

        if failure_prob < 0.35:
            status_html = '<span class="status-badge-healthy">✅ NORMAL / HEALTHY</span>'
            status_color = "#34d399"
        elif failure_prob < threshold:
            status_html = '<span class="status-badge-warning">⚠️ ELEVATED WEAR / MONITOR</span>'
            status_color = "#fbbf24"
        else:
            status_html = '<span class="status-badge-critical">🚨 CRITICAL RISK / DISPATCH MAINTENANCE</span>'
            status_color = "#f87171"

        st.markdown(f"### Status: {status_html}", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="kpi-card" style="margin-top: 10px; margin-bottom: 15px;">
                <h4>Predicted Failure Probability</h4>
                <div class="kpi-value" style="color: {status_color}; font-size: 2.3rem;">{failure_prob:.2%}</div>
                <p class="kpi-sub">Operating Decision Threshold: {threshold:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(1.0, failure_prob))

        st.markdown("#### 🔬 Physics-Based Root Cause Diagnostics")
        st.info(f"**Vulnerability Mode:** {cause_name}\n\n**Physics Mechanism:** {cause_desc}")

        st.markdown("#### 🛠️ Prescriptive Maintenance Work-Order")
        st.warning(f"**Recommended Action:** {action_desc}")

        st.markdown("#### 📐 Key Derived Telemetry Indicators")
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.metric("Mechanical Power", f"{power_w:.1f} W", help="Torque × Spindle Angular Velocity")
        with dcol2:
            st.metric("Temp Differential", f"{temp_diff:.1f} K", help="Process Temp - Air Temp")
        with dcol3:
            st.metric("Wear / Load Ratio", f"{wear_per_torque:.2f}", help="Tool Wear ÷ Torque")


# ==========================================
# PAGE 3: DYNAMIC MODEL ANALYTICS & SPECS (UPDATED TO ACTIVE DATASET)
# ==========================================
elif page == "📊 Model Analytics & Specs":
    st.subheader("📊 Dynamic Model Analytics & Performance Specifications")
    st.write(
        "Real-time model evaluation, live Precision-Recall curves, confusion matrices, and feature importance calibrated to the active dataset."
    )

    if is_custom_dataset:
        st.markdown(
            f"""
            <div class="active-badge-uploaded">
                <span>🟢</span>
                <strong>EVALUATING ON ACTIVE DATASET:</strong> {active_dataset_label}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="active-badge-benchmark">
                <span>⚙️</span>
                <strong>EVALUATING ON BENCHMARK DATASET:</strong> {active_dataset_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if has_ground_truth:
        y_true = df_clean["Machine failure"]
        y_pred = (fleet_failure_probs >= threshold).astype(int)

        dyn_prec = precision_score(y_true, y_pred, zero_division=0)
        dyn_rec = recall_score(y_true, y_pred, zero_division=0)
        dyn_f1 = f1_score(y_true, y_pred, zero_division=0)
        dyn_pr_auc = (
            average_precision_score(y_true, fleet_failure_probs)
            if len(np.unique(y_true)) > 1
            else 1.0
        )
        dyn_roc_auc = (
            roc_auc_score(y_true, fleet_failure_probs)
            if len(np.unique(y_true)) > 1
            else 1.0
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Dataset PR-AUC", f"{dyn_pr_auc:.3f}", help="Area under the Precision-Recall curve on active dataset")
        with c2:
            st.metric("Precision @ Threshold", f"{dyn_prec*100:.1f}%", help="True Failures / Total Flagged Alarms")
        with c3:
            st.metric("Recall @ Threshold", f"{dyn_rec*100:.1f}%", help="% of actual machine breakdowns caught")
        with c4:
            st.metric("F1 Score", f"{dyn_f1:.3f}", help="Harmonic mean of precision and recall")
        with c5:
            st.metric("ROC-AUC", f"{dyn_roc_auc:.3f}", help="Area under the ROC curve on active dataset")
    else:
        flagged_total = int(fleet_is_flagged.sum())
        flag_rate_pct = (flagged_total / len(results_df)) * 100 if len(results_df) > 0 else 0
        avg_risk_val = float(fleet_failure_probs.mean()) * 100
        p95_risk_val = float(np.percentile(fleet_failure_probs, 95)) * 100

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Scored", f"{len(results_df):,}")
        with c2:
            st.metric("Flagged Critical", f"{flagged_total:,}", help=f"{flag_rate_pct:.2f}% of active fleet")
        with c3:
            st.metric("Avg Fleet Risk", f"{avg_risk_val:.2f}%")
        with c4:
            st.metric("95th %ile Risk", f"{p95_risk_val:.2f}%")
        with c5:
            st.metric("Decision Threshold", f"{threshold:.4f}")

    st.markdown("---")

    tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs(
        [
            "📈 Live PR & ROC Curves",
            "🎯 Confusion Matrix & Classification",
            "🌟 Feature Importance & Weights",
            "🏛️ Training Benchmark Comparison",
        ]
    )

    with tab_m1:
        if has_ground_truth:
            curve_col1, curve_col2 = st.columns(2)

            with curve_col1:
                prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, fleet_failure_probs)
                fig_dyn_pr = go.Figure()
                fig_dyn_pr.add_trace(
                    go.Scatter(
                        x=rec_arr,
                        y=prec_arr,
                        mode="lines",
                        name=f"PR Curve (AUC = {dyn_pr_auc:.3f})",
                        line=dict(color="#38bdf8", width=3),
                    )
                )
                fig_dyn_pr.add_trace(
                    go.Scatter(
                        x=[dyn_rec],
                        y=[dyn_prec],
                        mode="markers+text",
                        name="Operating Threshold",
                        text=[f"Threshold = {threshold:.3f}"],
                        textposition="bottom left",
                        marker=dict(color="#f87171", size=12, symbol="circle"),
                    )
                )
                fig_dyn_pr.update_layout(
                    title=f"Precision-Recall Curve on Active Dataset (AUC: {dyn_pr_auc:.3f})",
                    xaxis_title="Recall (Failure Detection Rate)",
                    yaxis_title="Precision (Alarm Reliability)",
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=380,
                )
                st.plotly_chart(fig_dyn_pr, width="stretch")

            with curve_col2:
                fpr_arr, tpr_arr, _ = roc_curve(y_true, fleet_failure_probs)
                fig_dyn_roc = go.Figure()
                fig_dyn_roc.add_trace(
                    go.Scatter(
                        x=fpr_arr,
                        y=tpr_arr,
                        mode="lines",
                        name=f"ROC Curve (AUC = {dyn_roc_auc:.3f})",
                        line=dict(color="#a855f7", width=3),
                    )
                )
                fig_dyn_roc.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random Classifier",
                        line=dict(color="#64748b", dash="dash"),
                    )
                )
                fig_dyn_roc.update_layout(
                    title=f"Receiver Operating Characteristic (ROC-AUC: {dyn_roc_auc:.3f})",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=380,
                )
                st.plotly_chart(fig_dyn_roc, width="stretch")
        else:
            st.info(
                "💡 **Ground Truth Labeling Required for PR/ROC Curves:** The active uploaded dataset does not have a labeled `Machine failure` channel. Map a ground truth target column in the **Schema Mapping** tab on Page 1 to generate live PR & ROC curves."
            )

            fig_cum = px.ecdf(
                results_df,
                x="Failure Risk",
                color="Machine Type",
                title="Empirical Cumulative Failure Risk Distribution (ECDF) by Machine Type",
                template="plotly_dark",
            )
            fig_cum.add_vline(x=threshold, line_dash="dash", line_color="#38bdf8", annotation_text=f"Decision Threshold ({threshold:.3f})")
            fig_cum.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=380)
            st.plotly_chart(fig_cum, width="stretch")

    with tab_m2:
        if has_ground_truth:
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Healthy (0)", "Actual Failure (1)"],
                columns=["Predicted Healthy (0)", "Predicted Flagged (1)"],
            )

            cm_left, cm_right = st.columns([1, 1.2])

            with cm_left:
                fig_dyn_cm = px.imshow(
                    cm_df,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title="Active Dataset Confusion Matrix",
                    template="plotly_dark",
                )
                fig_dyn_cm.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
                st.plotly_chart(fig_dyn_cm, width="stretch")

            with cm_right:
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.6); padding: 18px; border-radius: 12px; margin-top: 10px;">
                        <h4 style="margin-top: 0;">Live Classification Breakdown on Active Data</h4>
                        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <td style="padding: 6px 0;"><strong>True Positives (Breakdowns Prevented):</strong></td>
                                <td style="text-align: right; color: #34d399; font-weight: bold;">{tp:,}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <td style="padding: 6px 0;"><strong>True Negatives (Healthy Uninterrupted):</strong></td>
                                <td style="text-align: right; color: #94a3b8; font-weight: bold;">{tn:,}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <td style="padding: 6px 0;"><strong>False Positives (Nuisance Alarms):</strong></td>
                                <td style="text-align: right; color: #fbbf24; font-weight: bold;">{fp:,}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <td style="padding: 6px 0;"><strong>False Negatives (Missed Breakdowns):</strong></td>
                                <td style="text-align: right; color: #f87171; font-weight: bold;">{fn:,}</td>
                            </tr>
                            <tr>
                                <td style="padding: 6px 0;"><strong>Specificity (Healthy Retention Rate):</strong></td>
                                <td style="text-align: right; color: #38bdf8; font-weight: bold;">{specificity*100:.2f}%</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Upload or map a dataset with ground truth failures to view the live confusion matrix.")

    with tab_m3:
        importances = model.feature_importances_
        fi_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance Gain": importances}
        ).sort_values("Importance Gain", ascending=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance Gain",
            y="Feature",
            orientation="h",
            title="XGBoost Feature Gain / Telemetry Channel Importance",
            template="plotly_dark",
            color="Importance Gain",
            color_continuous_scale="Viridis",
        )
        fig_fi.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
            xaxis_title="Relative Feature Gain",
            yaxis_title="Telemetry Feature",
        )
        st.plotly_chart(fig_fi, width="stretch")

        st.markdown(
            """
            **Key Physics Takeaways from Feature Importance:**
            - **`power_w` & `Torque (Nm)`:** Torque and delivered mechanical power dominate early fault detection, representing drive motor loading anomalies.
            - **`wear_per_torque` & `Tool wear (min)`:** Key indicators for catastrophic cutter overstrain (OSF).
            - **`temp_diff_k`:** Temperature differential isolates thermal dissipation degradation (HDF) before overheating causes line shutdowns.
            """
        )

    with tab_m4:
        st.markdown("#### 🏛️ Baseline Training vs. Active Dataset Comparison")

        comp_data = {
            "Metric / Spec": [
                "Dataset Name",
                "Total Record Count",
                "Decision Threshold",
                "PR-AUC",
                "Precision @ Threshold",
                "Recall @ Threshold",
                "ROC-AUC",
            ],
            "Baseline Model (Training Benchmark)": [
                "AI4I 2020 Holdout Split",
                "2,000 holdout units (10,000 total)",
                f"{meta.get('threshold', 0.846):.4f}",
                f"{meta.get('pr_auc', 0.888):.3f}",
                f"{meta.get('test_precision', 0.859)*100:.1f}%",
                f"{meta.get('test_recall', 0.809)*100:.1f}%",
                f"{meta.get('roc_auc', 0.985):.3f}",
            ],
            "Active Evaluated Dataset": [
                active_dataset_label,
                f"{len(results_df):,} units",
                f"{threshold:.4f}",
                f"{dyn_pr_auc:.3f}" if has_ground_truth else "N/A (Unlabeled)",
                f"{dyn_prec*100:.1f}%" if has_ground_truth else "N/A (Unlabeled)",
                f"{dyn_rec*100:.1f}%" if has_ground_truth else "N/A (Unlabeled)",
                f"{dyn_roc_auc:.3f}" if has_ground_truth else "N/A (Unlabeled)",
            ],
        }
        comp_table = pd.DataFrame(comp_data)
        st.dataframe(comp_table, width="stretch", hide_index=True)


# ==========================================
# PAGE 4: SYSTEM ARCHITECTURE
# ==========================================
elif page == "💡 System Architecture":
    st.subheader("💡 End-to-End Enterprise Predictive Maintenance Architecture")
    st.markdown(
        """
        ### Scalable Industrial Edge-to-Cloud Integration

        | Architecture Layer | Local Prototype (This App) | AWS / Enterprise Production Architecture |
        | :--- | :--- | :--- |
        | **Edge Telemetry Ingestion** | Local CSV Upload / Stream | AWS Kinesis Data Streams / IoT Greengrass / MQTT |
        | **Stream Processing & Feature Store** | Derived physics in `features.py` | Apache Flink / Amazon Managed Service for Apache Flink + SageMaker Feature Store |
        | **Model Training & Optimization** | XGBoost with `scale_pos_weight` | AWS SageMaker Pipelines (Hyperparameter Optimization & Imbalance Tuning) |
        | **Model Registry & Governance** | Joblib & metadata JSON | AWS SageMaker Model Registry with Lineage & Approval Workflows |
        | **Inference & Batch Scoring** | Interactive Streamlit Engine | AWS SageMaker Real-Time Multi-Model Endpoints + Nightly Batch Transform |
        | **Action & ERP Dispatch** | UI Status Badges & Priority CSV | AWS EventBridge / Lambda -> SAP PM / IBM Maximo Work Orders |
        | **Observability & Drift** | PR-AUC & Live Confusion Matrix | Amazon SageMaker Model Monitor & CloudWatch Real-Time Drift Alarms |
        """
    )
