"""
Create demo artifacts under artifacts/ so the web app runs before you re-run every notebook.

Replace these files with your real exports:
- notebook3_finale_deux_gagnants.ipynb → artifacts/best_model.joblib + best_model_meta.json
- notebook916b6caa72-4.ipynb → artifacts/engagement/*.pkl
- GMM notebook → scripts/export_gmm_from_notebook.py with your CSV
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models_loader import (  # noqa: E402
    COURSE_CATEGORIES,
    ENGAGEMENT_COLUMNS,
    GMM_FEATURE_NAMES,
)

RNG = np.random.RandomState(42)


def _demo_performance():
    ART = ROOT / "artifacts"
    ART.mkdir(parents=True, exist_ok=True)
    n = 600
    X = RNG.randn(n, len(GMM_FEATURE_NAMES)) * 12 + 55
    X = np.clip(X, 0, 100)
    df = pd.DataFrame(X, columns=GMM_FEATURE_NAMES)
    df["Final_Performance_Score"] = df.mean(axis=1) + RNG.randn(n) * 6

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=2.0)),
        ]
    )
    pipe.fit(df[GMM_FEATURE_NAMES], df["Final_Performance_Score"])
    joblib.dump(pipe, ART / "best_model.joblib")
    meta = {
        "modele": "Demo Ridge pipeline (replace with notebook 3 export)",
        "target": "Final_Performance_Score",
        "colonnes_features": list(GMM_FEATURE_NAMES),
        "fichier_modele": "best_model.joblib",
        "random_state": 42,
    }
    with open(ART / "best_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("OK performance ->", ART / "best_model.joblib")


def _demo_engagement():
    ENG = ROOT / "artifacts" / "engagement"
    ENG.mkdir(parents=True, exist_ok=True)
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise SystemExit("Install lightgbm: pip install lightgbm") from e

    n = 1200
    rows = []
    le = LabelEncoder()
    le.fit(COURSE_CATEGORIES)
    for _ in range(n):
        cat = COURSE_CATEGORIES[RNG.randint(0, len(COURSE_CATEGORIES))]
        rows.append(
            {
                "TimeSpentOnCourse": float(RNG.uniform(5, 120)),
                "NumberOfVideosWatched": float(RNG.randint(0, 21)),
                "NumberOfQuizzesTaken": float(RNG.randint(0, 11)),
                "QuizScores": float(RNG.uniform(40, 100)),
                "CompletionRate": float(RNG.uniform(5, 100)),
                "DeviceType": int(RNG.randint(0, 2)),
                "CourseCategory_encoded": int(le.transform([cat])[0]),
                "CourseCompletion": int(
                    RNG.rand()
                    < 0.35
                    + 0.003 * RNG.uniform(5, 120)
                    + 0.008 * RNG.uniform(40, 100)
                ),
            }
        )
    df = pd.DataFrame(rows)
    X = df[ENGAGEMENT_COLUMNS]
    y = df["CourseCompletion"]
    if y.nunique() < 2:
        y = pd.Series(RNG.randint(0, 2, size=len(df)), index=df.index)
        df["CourseCompletion"] = y
    clf = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X, y)
    joblib.dump(clf, ENG / "best_model_lightgbm.pkl")
    joblib.dump(le, ENG / "label_encoder.pkl")
    print("OK engagement ->", ENG / "best_model_lightgbm.pkl")


def _demo_gmm():
    GMM_DIR = ROOT / "artifacts" / "gmm"
    GMM_DIR.mkdir(parents=True, exist_ok=True)
    n = 800
    X = RNG.randn(n, len(GMM_FEATURE_NAMES)) * 15 + 50
    X = np.clip(X, 0, 100)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Paramètres proches du meilleur essai du notebook Optuna (silhouette)
    gmm = GaussianMixture(
        n_components=8,
        covariance_type="spherical",
        init_params="kmeans",
        max_iter=100,
        random_state=42,
    )
    gmm.fit(Xs)
    joblib.dump(gmm, GMM_DIR / "gmm_model.joblib")
    joblib.dump(scaler, GMM_DIR / "gmm_scaler.joblib")
    meta = {
        "feature_names": GMM_FEATURE_NAMES,
        "note": "Demo fit — replace with GMM trained on your Adaptive English Learning CSV",
    }
    with open(GMM_DIR / "gmm_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("OK gmm ->", GMM_DIR / "gmm_model.joblib")


def main():
    _demo_performance()
    _demo_engagement()
    _demo_gmm()
    print("\nArtifacts ready. Run: python serve.py")


if __name__ == "__main__":
    main()
