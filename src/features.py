"""
Feature engineering for the AI4I 2020 predictive maintenance dataset.

The AI4I dataset gives one snapshot reading per manufacturing process (no
repeated timestamps per machine), so there's no true rolling time-window to
compute. Instead this mirrors the *spirit* of the full design's feature
categories using what's available in a single snapshot:

  - Operational context features  -> raw sensor readings
  - Degradation-style features    -> tool wear, derived stress/power features
  - Categorical features          -> product quality variant (L/M/H)

If you swap in real time-series telemetry (multiple readings per machine
over time), add true rolling mean/std/max features here using
df.groupby("machine_id")[col].rolling(window).mean() etc.
"""

import pandas as pd


RAW_NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered features added and IDs/leak
    columns dropped. Keeps 'Machine failure' as the label if present."""
    df = df.copy()

    # --- Derived physical/degradation features -----------------------
    # Mechanical power delivered (torque * angular speed) - a classic
    # predictive-maintenance signal: abnormal power draw often precedes
    # a motor/bearing failure.
    df["power_w"] = df["Torque [Nm]"] * (
        df["Rotational speed [rpm]"] * 2 * 3.14159265 / 60
    )

    # Temperature differential between process and ambient air - a
    # widening gap suggests degraded heat dissipation.
    df["temp_diff_k"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Tool wear per unit torque - a proxy for how much stress the tool
    # has accumulated relative to the load it's under.
    df["wear_per_torque"] = df["Tool wear [min]"] / (df["Torque [Nm]"] + 1e-3)

    # --- Categorical encoding ------------------------------------------
    # Product quality variant (Low/Medium/High) is low-cardinality -> one-hot.
    df = pd.get_dummies(df, columns=["Type"], prefix="type")

    # --- Drop identifier / leakage columns ------------------------------
    # UDI and Product ID are just row identifiers.
    # TWF/HDF/PWF/OSF/RNF are the *individual* failure-mode flags that sum
    # to "Machine failure" - keeping them would leak the label directly,
    # so they're dropped from the feature set (they're useful later only
    # for the optional multi-class "which subsystem" model).
    drop_cols = [
        c
        for c in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    # XGBoost rejects feature names containing [, ], or < - sanitize.
    df.columns = (
        df.columns.str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", "lt", regex=False)
    )

    return df


def get_feature_label_cols(df: pd.DataFrame):
    label_col = "Machine failure"
    feature_cols = [c for c in df.columns if c != label_col]
    return feature_cols, label_col
