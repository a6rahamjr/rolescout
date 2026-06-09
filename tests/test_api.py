from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from rolescout.data.contracts import JobPosting, SearchProfile
from rolescout.data.providers.base import JobProvider
from rolescout.inference.alerts import SQLiteAlertRepository
from rolescout.utils.config import load_config


class StubProvider(JobProvider):
    async def search(self, profile: SearchProfile, limit: int) -> list[JobPosting]:
        return [
            JobPosting(
                job_id="stub-1",
                title="Machine Learning Engineer",
                company="Northstar Labs",
                description="Build production machine learning systems with Python.",
                url="https://example.com/stub-1",
                location="Remote",
                workplace="remote",
                experience_level="senior",
                job_type="full time",
                skills=("python", "pytorch", "mlops"),
                source="stub",
            )
        ]


def test_rank_and_alert_endpoints(trained_result, tmp_path) -> None:
    config = load_config()
    config = replace(
        config,
        storage=replace(config.storage, database_path=tmp_path / "alerts.db"),
    )
    repository = SQLiteAlertRepository(config.storage.database_path)
    app = create_app(
        config=config,
        ranker=trained_result.model,
        provider=StubProvider(),
        alerts=repository,
    )

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        assert health.json()["providers"] == ["stub"]
        model = client.get("/v1/model")
        assert model.status_code == 200
        assert model.json()["artifact_version"] == 2

        request = {
            "profile": {
                "query": "machine learning engineer",
                "skills": ["python", "pytorch"],
                "workplace": "remote",
            },
            "jobs": [
                {
                    "job_id": "one",
                    "title": "Machine Learning Engineer",
                    "company": "Northstar Labs",
                    "description": "Build ML services with Python.",
                    "url": "https://example.com/one",
                    "workplace": "remote",
                    "skills": ["python", "pytorch"],
                }
            ],
        }
        response = client.post("/v1/rank", json=request)
        assert response.status_code == 200
        assert response.json()["results"][0]["job"]["job_id"] == "one"

        search = client.post(
            "/v1/search",
            json={"profile": request["profile"], "limit": 10, "min_score": 0.1},
        )
        assert search.status_code == 200
        assert search.json()["sources"] == ["stub"]
        assert search.json()["results"][0]["job"]["job_id"] == "stub-1"
        assert "/v1/stream" in client.get("/openapi.json").json()["paths"]

        created = client.post(
            "/v1/alerts",
            json={
                "name": "ML roles",
                "profile": request["profile"],
                "min_score": 0.1,
            },
        )
        assert created.status_code == 201
        alert_id = created.json()["alert_id"]

        updated = client.patch(
            f"/v1/alerts/{alert_id}",
            json={"name": "Senior ML roles", "active": False},
        )
        assert updated.status_code == 200
        assert updated.json()["name"] == "Senior ML roles"
        assert updated.json()["active"] is False

        inactive = client.post(f"/v1/alerts/{alert_id}/check?limit=10")
        assert inactive.status_code == 200
        assert inactive.json()["count"] == 0

        client.patch(f"/v1/alerts/{alert_id}", json={"active": True})
        first = client.post(f"/v1/alerts/{alert_id}/check?limit=10")
        second = client.post(f"/v1/alerts/{alert_id}/check?limit=10")
        assert first.status_code == 200
        assert first.json()["count"] == 1
        assert second.json()["count"] == 0

        cleared = client.delete(f"/v1/alerts/{alert_id}/deliveries")
        assert cleared.status_code == 204
        repeated = client.post(f"/v1/alerts/{alert_id}/check?limit=10")
        assert repeated.json()["count"] == 1

        all_alerts = client.post("/v1/alerts/check-all?limit=10")
        assert all_alerts.status_code == 200
        assert all_alerts.json()["alerts_checked"] == 1
