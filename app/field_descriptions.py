"""Human-readable field descriptions for API schemas and the web UI (English)."""

from __future__ import annotations

# --- GMM / Optuna notebook (8 numeric skills) ---
GMM_FIELD_DETAILS: list[dict[str, str]] = [
    {
        "name": "Reading_Comprehension_Score",
        "description": "Score from reading comprehension tasks.",
        "typical_range": "0–100",
    },
    {
        "name": "Listening_Accuracy",
        "description": "Accuracy on listening exercises (proportion or score, same scale as training CSV).",
        "typical_range": "0–100",
    },
    {
        "name": "Writing_Score",
        "description": "Evaluated writing performance.",
        "typical_range": "0–100",
    },
    {
        "name": "Speaking_Score",
        "description": "Oral / speaking assessment score.",
        "typical_range": "0–100",
    },
    {
        "name": "Engagement_Level",
        "description": "Observed engagement with learning activities.",
        "typical_range": "0–100",
    },
    {
        "name": "Confidence_Rating",
        "description": "Learner self-reported confidence.",
        "typical_range": "0–100",
    },
    {
        "name": "Task_Difficulty",
        "description": "Perceived or assigned difficulty of tasks.",
        "typical_range": "0–100",
    },
    {
        "name": "Reward_Signal",
        "description": "Reward / feedback signal from the adaptive system.",
        "typical_range": "0–100",
    },
]

# --- Online course engagement / LightGBM notebook ---
ENGAGEMENT_INPUT_DETAILS: list[dict[str, str]] = [
    {
        "name": "TimeSpentOnCourse",
        "description": "Time spent on the course (same units as in the Kaggle CSV, e.g. hours).",
        "constraints": "number ≥ 0",
    },
    {
        "name": "NumberOfVideosWatched",
        "description": "Count of videos watched.",
        "constraints": "number ≥ 0",
    },
    {
        "name": "NumberOfQuizzesTaken",
        "description": "Count of quizzes taken.",
        "constraints": "number ≥ 0",
    },
    {
        "name": "QuizScores",
        "description": "Quiz performance score.",
        "constraints": "0–100",
    },
    {
        "name": "CompletionRate",
        "description": "Completion rate for the course segment.",
        "constraints": "0–100 (percent)",
    },
    {
        "name": "DeviceType",
        "description": "Device category encoded as in the dataset.",
        "constraints": "0 or 1",
    },
    {
        "name": "CourseCategory",
        "description": "Course subject category (model uses a label encoding).",
        "constraints": "one of allowed_course_categories",
    },
]

# --- Optional: richer hints for common performance-regression column names ---
_PERF_KNOWN: dict[str, tuple[str, str]] = {
    "Reading_Comprehension_Score": (
        "Reading comprehension score.",
        "0–100",
    ),
    "Listening_Accuracy": ("Listening accuracy.", "0–100"),
    "Writing_Score": ("Writing score.", "0–100"),
    "Speaking_Score": ("Speaking score.", "0–100"),
    "Engagement_Level": ("Engagement level.", "0–100"),
    "Confidence_Rating": ("Confidence rating.", "0–100"),
    "Task_Difficulty": ("Task difficulty.", "0–100"),
    "Reward_Signal": ("Reward signal.", "0–100"),
}


def performance_fields_detail(column_names: list[str]) -> list[dict[str, str]]:
    """Build {name, description, typical_range} for each training feature."""
    out: list[dict[str, str]] = []
    for name in column_names:
        if name in _PERF_KNOWN:
            desc, rng = _PERF_KNOWN[name]
            out.append(
                {
                    "name": name,
                    "description": desc,
                    "typical_range": rng,
                }
            )
        else:
            out.append(
                {
                    "name": name,
                    "description": "Numeric feature used when training the exported pipeline (see your CSV).",
                    "typical_range": "Use values comparable to your training data",
                }
            )
    return out
