"""
Run after Optuna in the GMM notebook, or standalone on a CSV that contains
GMM_FEATURE_NAMES columns.

Usage:
  python scripts/export_gmm_from_notebook.py path/to/dataset.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models_loader import GMM_FEATURE_NAMES  # noqa: E402

# Strong default hyperparameters from a representative Optuna run (max silhouette)
DEFAULT_BEST_PARAMS = {
    "n_components": 8,
    "covariance_type": "spherical",
    "init_params": "kmeans",
    "max_iter": 100,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        help="CSV with GMM columns (e.g. Adaptive English Learning dataset).",
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        help="JSON with n_components, covariance_type, init_params, max_iter",
    )
    args = parser.parse_args()

    out_dir = ROOT / "artifacts" / "gmm"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = dict(DEFAULT_BEST_PARAMS)
    if args.params_json and args.params_json.is_file():
        with open(args.params_json, encoding="utf-8") as f:
            params.update(json.load(f))

    if not args.csv or not args.csv.is_file():
        raise SystemExit("Usage: python scripts/export_gmm_from_notebook.py path/to/dataset.csv")

    df = pd.read_csv(args.csv)
    missing = [c for c in GMM_FEATURE_NAMES if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")
    X = df[GMM_FEATURE_NAMES].astype(float).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    gmm = GaussianMixture(
        n_components=int(params["n_components"]),
        covariance_type=params["covariance_type"],
        init_params=params["init_params"],
        max_iter=int(params["max_iter"]),
        random_state=42,
    )
    gmm.fit(Xs)
    joblib.dump(gmm, out_dir / "gmm_model.joblib")
    joblib.dump(scaler, out_dir / "gmm_scaler.joblib")
    meta = {
        "feature_names": GMM_FEATURE_NAMES,
        "gmm_params": params,
        "source_csv": str(args.csv.resolve()),
    }
    with open(out_dir / "gmm_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Exported to", out_dir)


if __name__ == "__main__":
    main()
