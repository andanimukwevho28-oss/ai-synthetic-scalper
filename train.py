"""
train.py

Trains a LightGBM binary classifier on the processed feature/label dataset
and exports it to ONNX for use by the MT5 EA via OnnxCreate()/OnnxRun().

Usage:
    python train.py --symbol "Volatility 75 Index"

Output:
    models/<symbol>.onnx           - the model MT5 loads
    models/<symbol>_report.txt     - accuracy / feature importance report
"""

import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from config import PROCESSED_DIR, MODEL_OUT_DIR, TRAIN_TEST_SPLIT, RANDOM_SEED
from features import FEATURE_COLUMNS


def train_symbol(symbol: str):
    data_path = os.path.join(PROCESSED_DIR, f"{symbol}.parquet")
    if not os.path.exists(data_path):
        raise SystemExit(f"No processed dataset for {symbol}. "
                          f"Run build_dataset.py first.")

    df = pd.read_parquet(data_path)
    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)

    # Chronological split - never shuffle time series data, it leaks the future
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[{symbol}] train={len(X_train)} test={len(X_test)} "
          f"positive_rate_train={y_train.mean():.3f} "
          f"positive_rate_test={y_test.mean():.3f}")

    model = lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        min_child_samples=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds)

    print(f"[{symbol}] test accuracy={acc:.4f} auc={auc:.4f}")
    print(report)

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)

    # --- Export to ONNX ---
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, len(FEATURE_COLUMNS)]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type, zipmap=False)

    onnx_path = os.path.join(MODEL_OUT_DIR, f"{symbol}.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[{symbol}] wrote {onnx_path}")

    report_path = os.path.join(MODEL_OUT_DIR, f"{symbol}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"Test AUC: {auc:.4f}\n\n")
        f.write("Feature importances:\n")
        for name, imp in sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda t: -t[1]
        ):
            f.write(f"  {name}: {imp}\n")
        f.write("\n" + report)
    print(f"[{symbol}] wrote {report_path}")

    print(f"\n[{symbol}] IMPORTANT: copy {symbol}.onnx into your MT5 "
          f"terminal's MQL5/Files/ folder and set the EA's ModelFileName "
          f"input to match, e.g. \"{symbol}.onnx\".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    args = parser.parse_args()
    train_symbol(args.symbol)


if __name__ == "__main__":
    main()
