"""
Train an XGBoost failure-prediction model on the AI4I 2020 dataset.

Run:
    python src/train.py

Outputs:
    models/xgb_model.joblib       - trained model
    models/threshold.json         - chosen operating threshold + metrics
    outputs/pr_curve.png          - precision-recall curve
    outputs/confusion_matrix.png  - confusion matrix at chosen threshold
    outputs/feature_importance.png
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import build_features, get_feature_label_cols

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PRECISION = 0.85  # operating point: pick the threshold that hits
# roughly this precision, then report the recall we get at that precision.
# Tune this to match how much "alert fatigue" the maintenance team can take.


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def main():
    print("Loading data...")
    raw = load_data()
    print(f"  {len(raw)} rows, failure rate = {raw['Machine failure'].mean():.3%}")

    print("Building features...")
    feat = build_features(raw)
    feature_cols, label_col = get_feature_label_cols(feat)
    X = feat[feature_cols]
    y = feat[label_col]

    # Stratified split so the (rare) failure class is proportionally
    # represented in both train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  train: {len(X_train)} rows ({y_train.sum()} failures)")
    print(f"  test:  {len(X_test)} rows ({y_test.sum()} failures)")

    # --- Class imbalance handling: scale_pos_weight -----------------
    # Ratio of negative to positive examples, so the model penalizes
    # missing a failure roughly as many times more than a false alarm.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"  scale_pos_weight = {scale_pos_weight:.1f}")

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Evaluate --------------------------------------------------
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nPR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}  (reference only - see note on imbalance)")

    # Threshold tuning: pick the lowest threshold that still hits our
    # target precision, to maximize recall within an acceptable false-alarm
    # budget (mirrors Section 8.2 of the design doc).
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    valid = np.where(precisions[:-1] >= TARGET_PRECISION)[0]
    if len(valid) > 0:
        best_idx = valid[np.argmax(recalls[:-1][valid])]
        chosen_threshold = float(thresholds[best_idx])
    else:
        chosen_threshold = 0.5
        print(f"  (no threshold hit {TARGET_PRECISION:.0%} precision - using 0.5)")

    y_pred = (y_proba >= chosen_threshold).astype(int)
    final_precision = precision_score(y_test, y_pred)
    final_recall = recall_score(y_test, y_pred)
    final_f1 = f1_score(y_test, y_pred)

    print(f"\nOperating threshold: {chosen_threshold:.3f}")
    print(f"  Precision: {final_precision:.3f}")
    print(f"  Recall:    {final_recall:.3f}")
    print(f"  F1:        {final_f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion matrix @ threshold {chosen_threshold:.3f}:")
    print(f"  True negatives:  {tn}")
    print(f"  False positives: {fp}  (unnecessary maintenance flags)")
    print(f"  False negatives: {fn}  (missed failures - the costly ones)")
    print(f"  True positives:  {tp}")

    # --- Plots -------------------------------------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label=f"PR-AUC = {pr_auc:.3f}")
    plt.scatter(
        [final_recall], [final_precision], color="red", zorder=5,
        label=f"chosen threshold = {chosen_threshold:.2f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"), dpi=120)
    plt.close()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Healthy", "Failure"]
    )
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix @ threshold {chosen_threshold:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=True)
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=120)
    plt.close()

    # --- Save model + threshold ---------------------------------------
    joblib.dump({"model": model, "feature_cols": feature_cols}, os.path.join(MODEL_DIR, "xgb_model.joblib"))
    with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
        json.dump(
            {
                "threshold": chosen_threshold,
                "target_precision": TARGET_PRECISION,
                "test_precision": final_precision,
                "test_recall": final_recall,
                "test_f1": final_f1,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            },
            f,
            indent=2,
        )

    print(f"\nSaved model to {MODEL_DIR}/xgb_model.joblib")
    print(f"Saved plots to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
