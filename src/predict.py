"""
Batch-score machines and produce a "flagged for maintenance" list.

This is the simplified stand-in for the full design's real-time endpoint +
nightly Batch Transform job + AWS Lambda action layer: one script that
scores a batch of machines and writes out who needs a maintenance visit.

Run:
    python src/predict.py --input data/ai4i2020.csv --output outputs/flagged_units.csv

You can point --input at any CSV with the same raw columns as the training
data (a new day's readings, for example).
"""

import argparse
import json
import os

import joblib
import pandas as pd

from features import build_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.json")


def load_model():
    bundle = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        meta = json.load(f)
    return bundle["model"], bundle["feature_cols"], meta["threshold"]


def score(input_path: str, output_path: str):
    model, feature_cols, threshold = load_model()

    raw = pd.read_csv(input_path)
    # Keep an identifier column for the output report if present.
    id_col = "Product ID" if "Product ID" in raw.columns else None
    ids = raw[id_col] if id_col else pd.Series(raw.index, name="row_id")

    feat = build_features(raw)
    # If the label happens to be in this file (e.g. re-scoring training
    # data for a demo), drop it - we're only predicting here.
    feat = feat.drop(columns=[c for c in ["Machine failure"] if c in feat.columns])
    X = feat[feature_cols]

    proba = model.predict_proba(X)[:, 1]
    flagged = proba >= threshold

    report = pd.DataFrame(
        {
            "id": ids,
            "failure_probability": proba.round(4),
            "flagged_for_maintenance": flagged,
        }
    ).sort_values("failure_probability", ascending=False)

    report.to_csv(output_path, index=False)

    n_flagged = int(flagged.sum())
    print(f"Scored {len(report)} units.")
    print(f"Flagged {n_flagged} unit(s) for maintenance (threshold={threshold:.3f}).")
    print(f"Full report written to {output_path}")
    if n_flagged:
        print("\nTop flagged units:")
        print(report[report.flagged_for_maintenance].head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(BASE_DIR, "data", "ai4i2020.csv"))
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "outputs", "flagged_units.csv"))
    args = parser.parse_args()
    score(args.input, args.output)
