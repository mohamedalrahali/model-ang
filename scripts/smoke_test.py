"""Smoke-test API routes with FastAPI TestClient (no external server)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def main() -> None:
    with TestClient(app) as client:
        h = client.get("/api/health")
        assert h.status_code == 200, h.text
        body = h.json()
        assert body.get("api_build") == "schema-routes-v2"
        assert "/api/schema/engagement" in (body.get("schema_endpoints") or {}).values()
        assert body["performance"]["loaded"] is True
        assert body["engagement"]["loaded"] is True
        assert body["gmm"]["loaded"] is True

        r = client.post("/api/admin/reload-models")
        assert r.status_code == 200

        eng_schema = client.get("/api/schema/engagement")
        assert eng_schema.status_code == 200
        assert "inputs" in eng_schema.json()

        gmm_schema = client.get("/api/schema/gmm")
        assert gmm_schema.status_code == 200
        assert len(gmm_schema.json().get("fields", [])) == 8

        perf = client.get("/api/schema/performance").json()
        names = perf["feature_names"]
        feats = {k: 55.0 for k in names}
        pr = client.post("/api/predict/performance", json={"features": feats})
        assert pr.status_code == 200, pr.text
        assert "properties" in pr.json()
        assert "input_profile" in pr.json()["properties"]

        eng = client.post(
            "/api/predict/engagement",
            json={
                "TimeSpentOnCourse": 60,
                "NumberOfVideosWatched": 10,
                "NumberOfQuizzesTaken": 5,
                "QuizScores": 75,
                "CompletionRate": 50,
                "DeviceType": 0,
                "CourseCategory": "Science",
            },
        )
        assert eng.status_code == 200, eng.text
        assert "properties" in eng.json()
        assert "learner_profile" in eng.json()["properties"]

        reco = client.post(
            "/api/recommend/categories",
            json={
                "TimeSpentOnCourse": 50,
                "NumberOfVideosWatched": 10,
                "NumberOfQuizzesTaken": 5,
                "QuizScores": 70,
                "CompletionRate": 40,
                "DeviceType": 0,
                "top_k": 3,
            },
        )
        assert reco.status_code == 200, reco.text
        rj = reco.json()
        assert "recommendations" in rj and "properties" in rj
        assert len(rj["recommendations"]) <= 3

        gmm_names = client.get("/api/schema/gmm").json()["feature_names"]
        gmm_body = {"features": {k: 60.0 for k in gmm_names}}
        gm = client.post("/api/predict/gmm", json=gmm_body)
        assert gm.status_code == 200, gm.text
        gj = gm.json()
        assert "cluster" in gj
        assert "properties" in gj
        assert "summary_fr" in gj["properties"]
        assert len(gj["properties"]["learner_profile"]) == 8

        idx = client.get("/")
        assert idx.status_code == 200
        assert b"Jungle in English" in idx.content

    print("smoke_test OK (English UI + schemas)")


if __name__ == "__main__":
    main()
