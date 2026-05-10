"""Load optional ML artifacts produced by the notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ENG = ART / "engagement"
GMM_DIR = ART / "gmm"

GMM_FEATURE_NAMES = [
    "Reading_Comprehension_Score",
    "Listening_Accuracy",
    "Writing_Score",
    "Speaking_Score",
    "Engagement_Level",
    "Confidence_Rating",
    "Task_Difficulty",
    "Reward_Signal",
]

COURSE_CATEGORIES = ["Arts", "Business", "Health", "Programming", "Science"]

# Same column order as notebook916 (FEATURES_LE)
ENGAGEMENT_COLUMNS = [
    "TimeSpentOnCourse",
    "NumberOfVideosWatched",
    "NumberOfQuizzesTaken",
    "QuizScores",
    "CompletionRate",
    "DeviceType",
    "CourseCategory_encoded",
]

ENGAGEMENT_FIELD_LABEL_FR: dict[str, str] = {
    "TimeSpentOnCourse": "Temps sur le cours",
    "NumberOfVideosWatched": "Vidéos regardées",
    "NumberOfQuizzesTaken": "Quiz réalisés",
    "QuizScores": "Score aux quiz",
    "CompletionRate": "Taux de complétion",
    "DeviceType": "Type d'appareil",
    "CourseCategory": "Catégorie du cours",
}

GMM_FEATURE_LABEL_FR: dict[str, str] = {
    "Reading_Comprehension_Score": "Compréhension écrite",
    "Listening_Accuracy": "Précision à l'écoute",
    "Writing_Score": "Expression écrite",
    "Speaking_Score": "Expression orale",
    "Engagement_Level": "Engagement",
    "Confidence_Rating": "Confiance perçue",
    "Task_Difficulty": "Difficulté des tâches",
    "Reward_Signal": "Signal de récompense (système adaptatif)",
}


def _score_band(v: float) -> dict[str, str]:
    """Tertile labels on a 0–100 style scale."""
    if v < 34:
        return {"code": "low", "label_fr": "faible", "label_en": "low"}
    if v < 67:
        return {"code": "medium", "label_fr": "moyen", "label_en": "medium"}
    return {"code": "high", "label_fr": "élevé", "label_en": "high"}


def _tier_low_med_high(val: float, low: float, high: float) -> dict[str, str]:
    if val < low:
        return {"code": "low", "label_fr": "faible", "label_en": "low"}
    if val < high:
        return {"code": "medium", "label_fr": "moyen", "label_en": "medium"}
    return {"code": "high", "label_fr": "élevé", "label_en": "high"}


def _performance_feature_property(name: str, value: float) -> dict[str, Any]:
    label_fr = GMM_FEATURE_LABEL_FR.get(name, name.replace("_", " "))
    if 0 <= value <= 100:
        b = _score_band(value)
        return {
            "feature": name,
            "feature_label_fr": label_fr,
            "value": value,
            "level": b["code"],
            "level_label_fr": b["label_fr"],
            "level_label_en": b["label_en"],
            "scale_note_fr": "Échelle 0–100 (tiers faible / moyen / élevé).",
            "scale_note_en": "0–100 scale (low / medium / high tertiles).",
        }
    return {
        "feature": name,
        "feature_label_fr": label_fr,
        "value": value,
        "level": "na",
        "level_label_fr": "hors plage 0–100",
        "level_label_en": "outside 0–100",
        "scale_note_fr": "Niveau relatif non affiché : utilisez la valeur brute telle qu'en base d'entraînement.",
        "scale_note_en": "Relative band not shown; interpret the raw value as in your training data.",
    }


def _predicted_score_properties(pred: float, target: str) -> dict[str, Any]:
    if pred < 45:
        band = {"code": "low", "label_fr": "faible", "label_en": "low"}
    elif pred < 70:
        band = {"code": "medium", "label_fr": "moyen", "label_en": "medium"}
    else:
        band = {"code": "high", "label_fr": "élevé", "label_en": "high"}
    summary_fr = (
        f"Score prédit pour « {target} » : {pred:.2f} — niveau interprété comme « {band['label_fr']} » "
        f"(seuils indicatifs : moins de 45 / 45 à 70 / au-delà de 70)."
    )
    summary_en = (
        f"Predicted « {target} »: {pred:.2f} — interpreted as « {band['label_en']} » "
        f"(indicative thresholds: below 45 / 45–70 / above 70)."
    )
    return {
        "predicted_value": pred,
        "target": target,
        "level": band["code"],
        "level_label_fr": band["label_fr"],
        "level_label_en": band["label_en"],
        "summary_fr": summary_fr,
        "summary_en": summary_en,
    }


def _engagement_learner_properties(
    time_spent: float,
    videos: float,
    quizzes: float,
    quiz_scores: float,
    completion_rate: float,
    device_type: int,
    course_category: str,
    *,
    include_category_row: bool = True,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    t_band = _tier_low_med_high(float(time_spent), 30.0, 90.0)
    out.append(
        {
            "field": "TimeSpentOnCourse",
            "field_label_fr": ENGAGEMENT_FIELD_LABEL_FR["TimeSpentOnCourse"],
            "value": time_spent,
            "level": t_band["code"],
            "level_label_fr": t_band["label_fr"],
            "level_label_en": t_band["label_en"],
            "hint_fr": "Temps total (unités du CSV, repères 30 / 90).",
            "hint_en": "Total time (CSV units; reference 30 / 90).",
        }
    )
    v_band = _tier_low_med_high(float(videos), 5.0, 15.0)
    out.append(
        {
            "field": "NumberOfVideosWatched",
            "field_label_fr": ENGAGEMENT_FIELD_LABEL_FR["NumberOfVideosWatched"],
            "value": videos,
            "level": v_band["code"],
            "level_label_fr": v_band["label_fr"],
            "level_label_en": v_band["label_en"],
            "hint_fr": "Nombre de vidéos (repères 5 / 15).",
            "hint_en": "Video count (reference 5 / 15).",
        }
    )
    q_band = _tier_low_med_high(float(quizzes), 3.0, 8.0)
    out.append(
        {
            "field": "NumberOfQuizzesTaken",
            "field_label_fr": ENGAGEMENT_FIELD_LABEL_FR["NumberOfQuizzesTaken"],
            "value": quizzes,
            "level": q_band["code"],
            "level_label_fr": q_band["label_fr"],
            "level_label_en": q_band["label_en"],
            "hint_fr": "Nombre de quiz (repères 3 / 8).",
            "hint_en": "Quiz count (reference 3 / 8).",
        }
    )
    qb = _score_band(float(quiz_scores))
    out.append(
        {
            "field": "QuizScores",
            "field_label_fr": ENGAGEMENT_FIELD_LABEL_FR["QuizScores"],
            "value": quiz_scores,
            "level": qb["code"],
            "level_label_fr": qb["label_fr"],
            "level_label_en": qb["label_en"],
            "hint_fr": "Score 0–100.",
            "hint_en": "Score 0–100.",
        }
    )
    cb = _score_band(float(completion_rate))
    out.append(
        {
            "field": "CompletionRate",
            "field_label_fr": ENGAGEMENT_FIELD_LABEL_FR["CompletionRate"],
            "value": completion_rate,
            "level": cb["code"],
            "level_label_fr": cb["label_fr"],
            "level_label_en": cb["label_en"],
            "hint_fr": "Pourcentage de complétion.",
            "hint_en": "Completion percentage.",
        }
    )
    dev_fr = "Bureau / web (0)" if int(device_type) == 0 else "Mobile ou autre (1)"
    dev_en = "Desktop / web (0)" if int(device_type) == 0 else "Mobile or other (1)"
    out.append(
        {
            "field": "DeviceType",
            "field_label_fr": ENGAGEMENT_FIELD_LABEL_FR["DeviceType"],
            "value": device_type,
            "level": "na",
            "level_label_fr": dev_fr,
            "level_label_en": dev_en,
            "hint_fr": "Codage identique au notebook Kaggle.",
            "hint_en": "Same encoding as the Kaggle notebook.",
        }
    )
    if include_category_row:
        out.append(
            {
                "field": "CourseCategory",
                "field_label_fr": ENGAGEMENT_FIELD_LABEL_FR["CourseCategory"],
                "value": course_category,
                "level": "na",
                "level_label_fr": course_category,
                "level_label_en": course_category,
                "hint_fr": "Parcours évalué pour cette catégorie.",
                "hint_en": "Path evaluated for this category.",
            }
        )
    return out


def _engagement_prediction_properties(
    pred_cls: int, p_completed: float, p_not: float
) -> dict[str, Any]:
    pc, pn = float(p_completed), float(p_not)
    if pred_cls == 1:
        verdict_fr = "Le modèle estime une complétion probable du cours."
        verdict_en = "The model predicts likely course completion."
    else:
        verdict_fr = "Le modèle estime un risque de non-complétion."
        verdict_en = "The model predicts risk of non-completion."
    if max(pc, pn) >= 0.65:
        cert_fr, cert_en = "confiance forte", "strong confidence"
    elif max(pc, pn) >= 0.45:
        cert_fr, cert_en = "confiance modérée", "moderate confidence"
    else:
        cert_fr, cert_en = "confiance faible (scores proches)", "low confidence (close scores)"
    summary_fr = (
        f"{verdict_fr} Probabilité de complétion : {100 * pc:.1f} %, "
        f"de non-complétion : {100 * pn:.1f} %. Certitude perçue : {cert_fr}."
    )
    summary_en = (
        f"{verdict_en} P(completed)={100 * pc:.1f}%, P(not)={100 * pn:.1f}%. "
        f"Perceived certainty: {cert_en}."
    )
    return {
        "verdict_fr": verdict_fr,
        "verdict_en": verdict_en,
        "probability_completed_percent": round(100 * pc, 2),
        "probability_not_completed_percent": round(100 * pn, 2),
        "certainty_fr": cert_fr,
        "certainty_en": cert_en,
        "summary_fr": summary_fr,
        "summary_en": summary_en,
    }


def _recommendation_properties(
    ranked: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(ranked):
        p = float(r["completion_probability"])
        rows.append(
            {
                "rank": i + 1,
                "category": r["category"],
                "probability_percent": round(100 * p, 2),
                "role_fr": "meilleur parcours" if i == 0 else f"alternative {i + 1}",
                "role_en": "best match" if i == 0 else f"alternative {i + 1}",
            }
        )
    parts_fr = [f"{x['rank']}. {x['category']} ({x['probability_percent']}%)" for x in rows]
    parts_en = [f"{x['rank']}. {x['category']} ({x['probability_percent']}%)" for x in rows]
    summary_fr = (
        "Ordre des catégories les plus favorables à la complétion (même profil, catégorie variable) : "
        + " ; ".join(parts_fr)
        + "."
    )
    summary_en = (
        "Categories most favorable to completion (same learner profile): "
        + "; ".join(parts_en)
        + "."
    )
    return {"ranks": rows, "summary_fr": summary_fr, "summary_en": summary_en}


@dataclass
class ModelState:
    performance_pipe: Any | None = None
    performance_meta: dict | None = None
    performance_error: str | None = None

    lgbm: Any | None = None
    label_encoder: Any | None = None
    engagement_error: str | None = None

    gmm: Any | None = None
    gmm_scaler: Any | None = None
    gmm_meta: dict | None = field(default_factory=dict)
    gmm_error: str | None = None


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_models() -> ModelState:
    state = ModelState()

    perf_job = ART / "best_model.joblib"
    perf_meta_path = ART / "best_model_meta.json"
    if perf_job.is_file():
        try:
            state.performance_pipe = joblib.load(perf_job)
            state.performance_meta = _load_json(perf_meta_path) or {}
        except Exception as e:  # noqa: BLE001
            state.performance_error = str(e)
    else:
        state.performance_error = f"Missing file: {perf_job}"

    lgbm_p = ENG / "best_model_lightgbm.pkl"
    le_p = ENG / "label_encoder.pkl"
    if lgbm_p.is_file() and le_p.is_file():
        try:
            state.lgbm = joblib.load(lgbm_p)
            state.label_encoder = joblib.load(le_p)
        except Exception as e:  # noqa: BLE001
            state.engagement_error = str(e)
    else:
        state.engagement_error = (
            f"Place engagement notebook exports in {ENG} "
            "(best_model_lightgbm.pkl, label_encoder.pkl)."
        )

    gmm_m = GMM_DIR / "gmm_model.joblib"
    gmm_s = GMM_DIR / "gmm_scaler.joblib"
    meta_g = _load_json(GMM_DIR / "gmm_meta.json") or {}
    if gmm_m.is_file() and gmm_s.is_file():
        try:
            state.gmm = joblib.load(gmm_m)
            state.gmm_scaler = joblib.load(gmm_s)
            state.gmm_meta = meta_g
        except Exception as e:  # noqa: BLE001
            state.gmm_error = str(e)
    else:
        state.gmm_error = (
            f"Place gmm_model.joblib and gmm_scaler.joblib in {GMM_DIR} "
            "(see scripts/export_gmm_from_notebook.py)."
        )

    return state


def predict_performance(state: ModelState, row: dict[str, float]) -> dict[str, Any]:
    if state.performance_pipe is None:
        raise RuntimeError(state.performance_error or "Performance model is not loaded.")
    meta = state.performance_meta or {}
    cols = meta.get("colonnes_features")
    if not cols:
        raise RuntimeError("best_model_meta.json has no colonnes_features list.")
    missing = [c for c in cols if c not in row]
    if missing:
        raise ValueError(f"Missing keys in `features`: {missing}")
    X = pd.DataFrame([{c: float(row[c]) for c in cols}])
    pred = float(state.performance_pipe.predict(X)[0])
    target = str(meta.get("target") or "Final_Performance_Score")
    input_profile = [_performance_feature_property(c, float(row[c])) for c in cols]
    pred_props = _predicted_score_properties(pred, target)
    return {
        "predicted_Final_Performance_Score": pred,
        "feature_columns": cols,
        "properties": {
            "input_profile": input_profile,
            "prediction": pred_props,
            "summary_fr": pred_props["summary_fr"],
            "summary_en": pred_props["summary_en"],
        },
    }


def predict_engagement(
    state: ModelState,
    time_spent: float,
    videos: float,
    quizzes: float,
    quiz_scores: float,
    completion_rate: float,
    device_type: int,
    course_category: str,
) -> dict[str, Any]:
    if state.lgbm is None or state.label_encoder is None:
        raise RuntimeError(state.engagement_error or "Engagement model is not loaded.")
    cat = course_category.strip()
    if cat not in COURSE_CATEGORIES:
        raise ValueError(
            f"Invalid CourseCategory. Allowed values: {', '.join(COURSE_CATEGORIES)}"
        )
    enc = int(state.label_encoder.transform([cat])[0])
    row = {
        "TimeSpentOnCourse": float(time_spent),
        "NumberOfVideosWatched": float(videos),
        "NumberOfQuizzesTaken": float(quizzes),
        "QuizScores": float(quiz_scores),
        "CompletionRate": float(completion_rate),
        "DeviceType": int(device_type),
        "CourseCategory_encoded": enc,
    }
    X = pd.DataFrame([row], columns=ENGAGEMENT_COLUMNS)
    proba = state.lgbm.predict_proba(X)[0]
    pred_cls = int(state.lgbm.predict(X)[0])
    pc, pn = float(proba[1]), float(proba[0])
    learner = _engagement_learner_properties(
        time_spent, videos, quizzes, quiz_scores, completion_rate, device_type, cat, include_category_row=True
    )
    pred_props = _engagement_prediction_properties(pred_cls, pc, pn)
    return {
        "CourseCompletion_predicted": pred_cls,
        "probability_completed": pc,
        "probability_not_completed": pn,
        "properties": {
            "learner_profile": learner,
            "prediction": pred_props,
            "summary_fr": pred_props["summary_fr"],
            "summary_en": pred_props["summary_en"],
        },
    }


def recommend_categories(
    state: ModelState,
    time_spent: float,
    videos: float,
    quizzes: float,
    quiz_scores: float,
    completion_rate: float,
    device_type: int,
    top_k: int = 3,
) -> dict[str, Any]:
    if state.lgbm is None or state.label_encoder is None:
        raise RuntimeError(state.engagement_error or "Engagement model is not loaded.")
    recs: list[dict[str, Any]] = []
    for category in COURSE_CATEGORIES:
        enc = int(state.label_encoder.transform([category])[0])
        row = {
            "TimeSpentOnCourse": float(time_spent),
            "NumberOfVideosWatched": float(videos),
            "NumberOfQuizzesTaken": float(quizzes),
            "QuizScores": float(quiz_scores),
            "CompletionRate": float(completion_rate),
            "DeviceType": int(device_type),
            "CourseCategory_encoded": enc,
        }
        X = pd.DataFrame([row], columns=ENGAGEMENT_COLUMNS)
        p = float(state.lgbm.predict_proba(X)[0][1])
        recs.append({"category": category, "completion_probability": round(p, 4)})
    recs.sort(key=lambda x: x["completion_probability"], reverse=True)
    top = recs[: max(1, min(top_k, 5))]
    learner = _engagement_learner_properties(
        time_spent,
        videos,
        quizzes,
        quiz_scores,
        completion_rate,
        device_type,
        "",
        include_category_row=False,
    )
    props = _recommendation_properties(top)
    props["learner_profile"] = learner
    props["note_fr"] = (
        "Même profil numérique pour chaque catégorie : seule la matière change pour estimer "
        "la probabilité de complétion."
    )
    props["note_en"] = (
        "Same numeric profile for each category: only the subject changes when estimating completion odds."
    )
    return {"recommendations": top, "properties": props}


def _gmm_properties_fr(
    feat_names: list[str],
    row: dict[str, float],
    Xs: np.ndarray,
    gmm: Any,
    label: int,
    proba: list[float],
) -> dict[str, Any]:
    """Human-readable cluster assignment + learner skill bands (French + English)."""
    learner_profile: list[dict[str, Any]] = []
    for name in feat_names:
        val = float(row[name])
        band = _score_band(val)
        learner_profile.append(
            {
                "feature": name,
                "feature_label_fr": GMM_FEATURE_LABEL_FR.get(name, name.replace("_", " ")),
                "value": val,
                "level": band["code"],
                "level_label_fr": band["label_fr"],
                "level_label_en": band["label_en"],
            }
        )

    proba_arr = np.asarray(proba, dtype=np.float64)
    max_p = float(proba_arr.max())
    if max_p >= 0.5:
        certainty_fr = "affectation claire"
        certainty_en = "clear assignment"
    elif max_p >= 0.25:
        certainty_fr = "affectation modérée"
        certainty_en = "moderate assignment"
    else:
        certainty_fr = "profil ambigu (plusieurs segments possibles)"
        certainty_en = "ambiguous profile (several segments plausible)"

    ranked: list[dict[str, Any]] = []
    order = np.argsort(-proba_arr)
    for rank, j in enumerate(order, start=1):
        pj = float(proba_arr[int(j)])
        ranked.append(
            {
                "rank": rank,
                "cluster": int(j),
                "probability": pj,
                "probability_percent": round(100 * pj, 2),
                "role_fr": "segment dominant" if rank == 1 else f"alternative n°{rank}",
                "role_en": "dominant segment" if rank == 1 else f"alternative #{rank}",
            }
        )

    x = Xs.ravel()
    best_j, best_d = 0, float("inf")
    for j in range(gmm.n_components):
        d = float(np.linalg.norm(x - gmm.means_[j]))
        if d < best_d:
            best_d = d
            best_j = j

    highs_fr = [p["feature"] for p in learner_profile if p["level"] == "high"]
    lows_fr = [p["feature"] for p in learner_profile if p["level"] == "low"]

    pct_label = round(100 * float(proba_arr[label]), 1)
    summary_fr = (
        f"Segment dominant : n°{label} (probabilité a posteriori ≈ {pct_label} %). "
        f"{certainty_fr.capitalize()}."
    )
    if highs_fr:
        summary_fr += f" Scores plutôt élevés : {', '.join(highs_fr)}."
    if lows_fr:
        summary_fr += f" Scores plutôt faibles : {', '.join(lows_fr)}."
    if int(best_j) != int(label):
        summary_fr += (
            f" Remarque : dans l'espace normalisé, le centroïde le plus proche est le segment {best_j} "
            f"(≠ segment prédit {label})."
        )

    summary_en = (
        f"Dominant segment: #{label} (posterior ≈ {pct_label} %). "
        f"{certainty_en.capitalize()}."
    )
    if highs_fr:
        summary_en += f" Relatively high scores: {', '.join(highs_fr)}."
    if lows_fr:
        summary_en += f" Relatively low scores: {', '.join(lows_fr)}."

    return {
        "learner_profile": learner_profile,
        "assignment": {
            "predicted_cluster": label,
            "posterior_probability_percent": pct_label,
            "max_posterior_percent": round(100 * max_p, 2),
            "certainty_fr": certainty_fr,
            "certainty_en": certainty_en,
        },
        "segments_ranked": ranked,
        "geometry_note_fr": (
            f"Plus proche centroïde (données normalisées) : segment {best_j} (distance L2 ≈ {best_d:.3f})."
        ),
        "geometry_note_en": (
            f"Closest centroid (scaled data): component {best_j} (L2 distance ≈ {best_d:.3f})."
        ),
        "summary_fr": summary_fr.strip(),
        "summary_en": summary_en.strip(),
    }


def predict_gmm(state: ModelState, row: dict[str, float]) -> dict[str, Any]:
    if state.gmm is None or state.gmm_scaler is None:
        raise RuntimeError(state.gmm_error or "GMM model is not loaded.")
    feats = state.gmm_meta.get("feature_names") or GMM_FEATURE_NAMES
    missing = [c for c in feats if c not in row]
    if missing:
        raise ValueError(f"Missing keys in `features`: {missing}")
    X = np.array([[float(row[c]) for c in feats]], dtype=np.float64)
    Xs = state.gmm_scaler.transform(X)
    label = int(state.gmm.predict(Xs)[0])
    proba = state.gmm.predict_proba(Xs)[0].tolist()
    properties = _gmm_properties_fr(feats, row, Xs, state.gmm, label, proba)
    return {
        "cluster": label,
        "cluster_probabilities": [float(p) for p in proba],
        "n_components": int(state.gmm.n_components),
        "properties": properties,
    }


def health_payload(state: ModelState) -> dict[str, Any]:
    return {
        "performance": {
            "loaded": state.performance_pipe is not None,
            "error": state.performance_error,
            "target": (state.performance_meta or {}).get("target"),
            "model_label": (state.performance_meta or {}).get("modele"),
        },
        "engagement": {"loaded": state.lgbm is not None, "error": state.engagement_error},
        "gmm": {"loaded": state.gmm is not None, "error": state.gmm_error},
    }
