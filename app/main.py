"""FastAPI application factory for RoleScout."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.schemas import (
    AlertCreateRequest,
    AlertUpdateRequest,
    RankRequest,
    SearchRequest,
)
from rolescout.data.contracts import JobPosting, SearchProfile
from rolescout.data.providers.base import JobProvider
from rolescout.data.providers.factory import build_provider
from rolescout.inference.alerts import AlertService, SQLiteAlertRepository
from rolescout.inference.service import RankingService
from rolescout.inference.streaming import LiveJobStreamer
from rolescout.models.ranker import JobRanker
from rolescout.utils.config import AppConfig, load_config
from rolescout.utils.logging import configure_logging


class ApplicationState:
    def __init__(
        self,
        config: AppConfig,
        ranker: JobRanker | None,
        provider: JobProvider | None,
        alerts: SQLiteAlertRepository | None,
    ) -> None:
        self.config = config
        self.ranker = ranker
        self.provider = provider or build_provider(config.provider)
        self._alerts = alerts
        self.model_error: str | None = None

    def load_model(self) -> None:
        if self.ranker is not None:
            return
        try:
            self.ranker = JobRanker.load(self.config.model.artifact_path)
            self.model_error = None
        except (FileNotFoundError, TypeError, ValueError) as exc:
            self.model_error = str(exc)

    def ranking_service(self) -> RankingService:
        self.load_model()
        if self.ranker is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Model is unavailable. Run `python scripts/train.py --regenerate`. "
                    f"Details: {self.model_error}"
                ),
            )
        return RankingService(self.ranker, self.provider)

    def alert_repository(self) -> SQLiteAlertRepository:
        if self._alerts is None:
            self._alerts = SQLiteAlertRepository(self.config.storage.database_path)
        return self._alerts


def _profile(payload: Any) -> SearchProfile:
    return SearchProfile.from_mapping(payload.model_dump())


def _jobs(payloads: list[Any]) -> list[JobPosting]:
    return [
        JobPosting.from_mapping(
            {
                **payload.model_dump(),
                "url": str(payload.url),
            }
        )
        for payload in payloads
    ]


def create_app(
    *,
    config: AppConfig | None = None,
    ranker: JobRanker | None = None,
    provider: JobProvider | None = None,
    alerts: SQLiteAlertRepository | None = None,
) -> FastAPI:
    configure_logging()
    resolved_config = config or load_config()
    state_container = ApplicationState(resolved_config, ranker, provider, alerts)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state_container.load_model()
        state_container.alert_repository()
        yield

    application = FastAPI(
        title="RoleScout",
        version="1.2.0",
        description="Job search ranking, live streams, explanations, and saved alerts.",
        lifespan=lifespan,
    )

    @application.get("/health")
    async def health() -> dict[str, Any]:
        state_container.load_model()
        return {
            "status": "ok" if state_container.ranker else "degraded",
            "model_loaded": state_container.ranker is not None,
            "model_error": state_container.model_error,
            "artifact_path": str(resolved_config.model.artifact_path),
            "providers": list(state_container.provider.names),
        }

    @application.get("/v1/model")
    async def model_info() -> dict[str, Any]:
        service = state_container.ranking_service()
        details = service.ranker.describe()
        return {
            "artifact_version": details["artifact_version"],
            "trained_at": details["trained_at"],
            "training_rows": details["training_rows"],
            "hyperparameters": details["hyperparameters"],
            "metrics": details.get("metrics", {}),
            "feature_weights": details["coefficients"],
        }

    @application.post("/v1/rank")
    async def rank(request: RankRequest) -> dict[str, Any]:
        service = state_container.ranking_service()
        ranked = service.rank(
            _profile(request.profile),
            _jobs(request.jobs),
            limit=request.limit,
            min_score=request.min_score,
        )
        return {"count": len(ranked), "results": [item.to_dict() for item in ranked]}

    @application.post("/v1/search")
    async def search(request: SearchRequest) -> dict[str, Any]:
        service = state_container.ranking_service()
        try:
            ranked = await service.search(
                _profile(request.profile),
                limit=request.limit,
                min_score=request.min_score,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Job provider request failed: {exc}",
            ) from exc
        sources = sorted({item.job.source for item in ranked})
        return {
            "count": len(ranked),
            "sources": sources,
            "source_notice": (
                f"Jobs sourced from {', '.join(sources)}; links point to original listings."
                if sources
                else "No matching jobs were returned by the configured providers."
            ),
            "results": [item.to_dict() for item in ranked],
        }

    @application.get("/v1/stream")
    async def stream_jobs(
        query: str = Query(min_length=1, max_length=300),
        location: str = Query(default="", max_length=200),
        skills: str = Query(default="", max_length=1000),
        experience_level: str = Query(default="", max_length=50),
        workplace: str = Query(default="any", max_length=30),
        job_types: str = Query(default="", max_length=300),
        excluded_keywords: str = Query(default="", max_length=1000),
        excluded_companies: str = Query(default="", max_length=1000),
        limit: int = Query(default=20, ge=1, le=100),
        min_score: float = Query(default=0.0, ge=0, le=1),
        poll_seconds: int | None = Query(default=None, ge=10, le=300),
        include_existing: bool = Query(default=True),
    ) -> StreamingResponse:
        profile = SearchProfile.from_mapping(
            {
                "query": query,
                "location": location,
                "skills": skills,
                "experience_level": experience_level,
                "workplace": workplace,
                "job_types": job_types,
                "excluded_keywords": excluded_keywords,
                "excluded_companies": excluded_companies,
            }
        )
        service = state_container.ranking_service()
        streamer = LiveJobStreamer(service)
        interval = poll_seconds or resolved_config.api.stream_poll_seconds
        return StreamingResponse(
            streamer.events(
                profile,
                limit=limit,
                min_score=min_score,
                poll_seconds=interval,
                include_existing=include_existing,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @application.post("/v1/alerts", status_code=status.HTTP_201_CREATED)
    async def create_alert(request: AlertCreateRequest) -> dict[str, Any]:
        record = state_container.alert_repository().create(
            request.name,
            _profile(request.profile),
            request.min_score,
        )
        return record.to_dict()

    @application.get("/v1/alerts")
    async def list_alerts() -> dict[str, Any]:
        records = state_container.alert_repository().list()
        return {"count": len(records), "alerts": [record.to_dict() for record in records]}

    @application.get("/v1/alerts/{alert_id}")
    async def get_alert(alert_id: int) -> dict[str, Any]:
        record = state_container.alert_repository().get(alert_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found",
            )
        return record.to_dict()

    @application.patch("/v1/alerts/{alert_id}")
    async def update_alert(
        alert_id: int,
        request: AlertUpdateRequest,
    ) -> dict[str, Any]:
        record = state_container.alert_repository().update(
            alert_id,
            name=request.name,
            profile=_profile(request.profile) if request.profile else None,
            min_score=request.min_score,
            active=request.active,
        )
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found",
            )
        return record.to_dict()

    @application.delete("/v1/alerts/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_alert(alert_id: int) -> None:
        if not state_container.alert_repository().delete(alert_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    @application.delete(
        "/v1/alerts/{alert_id}/deliveries",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def clear_alert_history(alert_id: int) -> None:
        if not state_container.alert_repository().clear_deliveries(alert_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found",
            )

    @application.post("/v1/alerts/check-all")
    async def check_all_alerts(
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, Any]:
        alert_service = AlertService(
            state_container.alert_repository(),
            state_container.ranking_service(),
        )
        try:
            matches = await alert_service.check_all(limit=limit)
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Job provider request failed: {exc}",
            ) from exc
        return {
            "alerts_checked": len(matches),
            "new_jobs": sum(len(items) for items in matches.values()),
            "results": {
                str(alert_id): [item.to_dict() for item in items]
                for alert_id, items in matches.items()
            },
        }

    @application.post("/v1/alerts/{alert_id}/check")
    async def check_alert(
        alert_id: int,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, Any]:
        alert_service = AlertService(
            state_container.alert_repository(),
            state_container.ranking_service(),
        )
        try:
            matches = await alert_service.check(alert_id, limit=limit)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found"
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Job provider request failed: {exc}",
            ) from exc
        return {"count": len(matches), "results": [item.to_dict() for item in matches]}

    return application


app = create_app()
