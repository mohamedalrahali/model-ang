"""
Train LightGBM + LabelEncoder like notebook916 (online course engagement) and save
artifacts/engagement/*.pkl

Usage:
  python scripts/train_engagement_from_csv.py path/to/online_course_engagement_data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit("Run: pip install lightgbm") from e

from app.models_loader import ENGAGEMENT_COLUMNS  # noqa: E402

TARGET = "CourseCompletion"
REQUIRED = ENGAGEMENT_COLUMNS[:-1] + ["CourseCategory", TARGET]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path, help="CSV Kaggle online_course_engagement_data")
    args = p.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"File not found: {args.csv}")

    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    le = LabelEncoder()
    df = df.copy()
    df["CourseCategory_encoded"] = le.fit_transform(df["CourseCategory"])

    X = df[ENGAGEMENT_COLUMNS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=1.5,
        random_state=42,
        verbose=-1,
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    out = ROOT / "artifacts" / "engagement"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "best_model_lightgbm.pkl")
    joblib.dump(le, out / "label_encoder.pkl")
    print("Models saved to", out)


if __name__ == "__main__":
    main()
