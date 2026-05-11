from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from app.field_descriptions import (
    ENGAGEMENT_INPUT_DETAILS,
    GMM_FIELD_DETAILS,
    performance_fields_detail,
)
from app.models_loader import (
    GMM_FEATURE_NAMES,
    ModelState,
    COURSE_CATEGORIES,
    health_payload,
    load_models,
    predict_engagement,
    predict_gmm,
    predict_performance,
    recommend_categories,
)

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"

state: ModelState = ModelState()


class PerformanceHealthBlock(BaseModel):
    """Status of the final-score regression pipeline."""

    loaded: bool = Field(description="True if artifacts/best_model.joblib is loaded.")
    error: str | None = Field(None, description="Error message when the model is missing or invalid.")
    target: str | None = Field(None, description="Target column name from best_model_meta.json.")
    model_label: str | None = Field(None, description="Human-readable model label from metadata.")


class SimpleLoadedBlock(BaseModel):
    """Generic loaded / error pair for engagement and GMM."""

    loaded: bool
    error: str | None = None


class HealthResponse(BaseModel):
    """Payload returned by GET /api/health and POST /api/admin/reload-models."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "performance": {
                        "loaded": True,
                        "error": None,
                        "target": "Final_Performance_Score",
                        "model_label": "Ridge pipeline",
                    },
                    "engagement": {"loaded": True, "error": None},
                    "gmm": {"loaded": True, "error": None},
                    "schema_endpoints": {
                        "engagement": "/api/schema/engagement",
                        "gmm": "/api/schema/gmm",
                        "performance": "/api/schema/performance",
                    },
                    "api_build": "schema-routes-v2",
                }
            ]
        }
    )

    performance: PerformanceHealthBlock
    engagement: SimpleLoadedBlock
    gmm: SimpleLoadedBlock
    schema_endpoints: dict[str, str] = Field(
        description="Relative paths to JSON schemas for each model.",
    )
    api_build: str = Field(description="Build tag for debugging deployments.")


def _health_api_body() -> dict:
    body = health_payload(state)
    body["schema_endpoints"] = {
        "engagement": "/api/schema/engagement",
        "gmm": "/api/schema/gmm",
        "performance": "/api/schema/performance",
    }
    body["api_build"] = "schema-routes-v2"
    return body


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global state
    state = load_models()
    yield


app = FastAPI(
    title="Jungle in English — ML Lab",
    description="Web API for performance regression, course-completion (LightGBM), and GMM segmentation.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    index_path = STATIC / "index.html"
    if not index_path.is_file():
        raise HTTPException(404, "static/index.html is missing")
    return FileResponse(index_path)


@app.get("/api/health", response_model=HealthResponse)
async def api_health():
    return HealthResponse.model_validate(_health_api_body())


@app.post("/api/admin/reload-models", response_model=HealthResponse)
async def api_reload_models():
    """Reload joblib/pkl artifacts from disk (after copy or retraining)."""
    global state
    state = load_models()
    return HealthResponse.model_validate(_health_api_body())


class PerformanceIn(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Numeric features required by artifacts/best_model_meta.json (excluding the target column).",
    )


@app.post("/api/predict/performance")
async def api_predict_performance(body: PerformanceIn):
    try:
        return predict_performance(state, body.features)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e


class EngagementIn(BaseModel):
    TimeSpentOnCourse: float = Field(..., ge=0, description="Time spent on the course (dataset units).")
    NumberOfVideosWatched: float = Field(..., ge=0, description="Number of videos watched.")
    NumberOfQuizzesTaken: float = Field(..., ge=0, description="Number of quizzes taken.")
    QuizScores: float = Field(..., ge=0, le=100, description="Quiz score 0–100.")
    CompletionRate: float = Field(..., ge=0, le=100, description="Completion rate 0–100 (%).")
    DeviceType: int = Field(0, ge=0, le=1, description="Device type flag: 0 or 1 as in the CSV.")
    CourseCategory: str = Field(
        "Programming",
        description="One of: Arts, Business, Health, Programming, Science.",
    )


@app.post("/api/predict/engagement")
async def api_predict_engagement(body: EngagementIn):
    try:
        return predict_engagement(
            state,
            body.TimeSpentOnCourse,
            body.NumberOfVideosWatched,
            body.NumberOfQuizzesTaken,
            body.QuizScores,
            body.CompletionRate,
            body.DeviceType,
            body.CourseCategory,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e


class RecommendIn(BaseModel):
    TimeSpentOnCourse: float = Field(..., ge=0)
    NumberOfVideosWatched: float = Field(..., ge=0)
    NumberOfQuizzesTaken: float = Field(..., ge=0)
    QuizScores: float = Field(..., ge=0, le=100)
    CompletionRate: float = Field(..., ge=0, le=100)
    DeviceType: int = Field(0, ge=0, le=1)
    top_k: int = Field(3, ge=1, le=5, description="Number of top categories to return (1–5).")


@app.post("/api/recommend/categories")
async def api_recommend(body: RecommendIn):
    try:
        return recommend_categories(
            state,
            body.TimeSpentOnCourse,
            body.NumberOfVideosWatched,
            body.NumberOfQuizzesTaken,
            body.QuizScores,
            body.CompletionRate,
            body.DeviceType,
            body.top_k,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e


class GmmIn(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Eight learner-skill numeric features (same names as the GMM / Optuna notebook).",
    )


@app.post("/api/predict/gmm")
async def api_predict_gmm(body: GmmIn):
    try:
        return predict_gmm(state, body.features)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/api/schema/gmm")
async def schema_gmm():
    return {
        "feature_names": GMM_FEATURE_NAMES,
        "fields": GMM_FIELD_DETAILS,
        "required_parameters": "Send all eight feature names in POST /api/predict/gmm under `features`.",
    }


@app.get("/api/schema/engagement")
async def schema_engagement():
    return {
        "inputs": ENGAGEMENT_INPUT_DETAILS,
        "allowed_course_categories": COURSE_CATEGORIES,
        "required_parameters": "POST /api/predict/engagement with JSON keys matching the engagement notebook CSV.",
    }


@app.get("/api/schema/performance")
async def schema_performance():
    meta = state.performance_meta or {}
    cols = meta.get("colonnes_features")
    if not cols:
        return {
            "loaded": False,
            "target": None,
            "feature_names": [],
            "fields": [],
            "hint": "Run notebook 3 and copy artifacts/best_model_meta.json, or run: python scripts/bootstrap_demo.py",
        }
    return {
        "loaded": True,
        "target": meta.get("target"),
        "feature_names": cols,
        "fields": performance_fields_detail(cols),
        "required_parameters": "POST /api/predict/performance with JSON `{ \"features\": { \"ColName\": value, ... } }` for every name in feature_names.",
    }


# Mount static files last so API routes always take precedence (Starlette/FastAPI best practice).
if STATIC.is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC)), name="assets")
