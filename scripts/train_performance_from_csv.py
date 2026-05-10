"""
Train StandardScaler + Ridge on a CSV (numeric columns + target) and write
artifacts/best_model.joblib + best_model_meta.json — shortcut if notebook 3 is not run end-to-end.

Usage:
  python scripts/train_performance_from_csv.py path/to/dataset.csv
  python scripts/train_performance_from_csv.py data.csv --target MyScore
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TARGET = "Final_Performance_Score"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--target", default=DEFAULT_TARGET)
    args = p.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"File not found: {args.csv}")

    df = pd.read_csv(args.csv)
    id_cols = [
        c
        for c in df.columns
        if str(c).strip().lower() in ("id", "student_id", "student id")
    ]
    df = df.drop(columns=id_cols, errors="ignore")
    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' missing. Columns: {list(df.columns)}")

    X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
    y = df[args.target]
    if X.shape[1] == 0:
        raise SystemExit("No numeric columns for X (excluding target).")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=2.0)),
        ]
    )
    pipe.fit(X, y)

    art = ROOT / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, art / "best_model.joblib")
    meta = {
        "modele": "Ridge pipeline (train_performance_from_csv.py)",
        "target": args.target,
        "colonnes_features": list(X.columns),
        "fichier_modele": "best_model.joblib",
        "source_csv": str(args.csv.resolve()),
    }
    with open(art / "best_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("OK →", art / "best_model.joblib")


if __name__ == "__main__":
    main()
