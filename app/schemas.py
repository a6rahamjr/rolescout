"""Pydantic request and response contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class SearchProfilePayload(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    location: str = Field(default="", max_length=200)
    skills: list[str] = Field(default_factory=list, max_length=100)
    experience_level: str = Field(default="", max_length=50)
    workplace: str = Field(default="any", max_length=30)
    job_types: list[str] = Field(default_factory=list, max_length=20)
    excluded_keywords: list[str] = Field(default_factory=list, max_length=50)
    excluded_companies: list[str] = Field(default_factory=list, max_length=50)


class JobPayload(BaseModel):
    job_id: str = Field(min_length=1, max_length=300)
    title: str = Field(min_length=1, max_length=300)
    company: str = Field(default="", max_length=300)
    description: str = Field(default="", max_length=50000)
    url: HttpUrl | str = ""
    location: str = Field(default="", max_length=300)
    workplace: str = Field(default="unknown", max_length=30)
    experience_level: str = Field(default="", max_length=50)
    job_type: str = Field(default="", max_length=50)
    posted_at: datetime | None = None
    skills: list[str] = Field(default_factory=list, max_length=100)
    salary: str = Field(default="", max_length=300)
    source: str = Field(default="provided", max_length=50)


class RankRequest(BaseModel):
    profile: SearchProfilePayload
    jobs: list[JobPayload] = Field(min_length=1, max_length=1000)
    limit: int | None = Field(default=None, ge=1, le=1000)
    min_score: float = Field(default=0.0, ge=0, le=1)


class SearchRequest(BaseModel):
    profile: SearchProfilePayload
    limit: int = Field(default=20, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0, le=1)


class AlertCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    profile: SearchProfilePayload
    min_score: float = Field(default=0.65, ge=0, le=1)


class AlertUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    profile: SearchProfilePayload | None = None
    min_score: float | None = Field(default=None, ge=0, le=1)
    active: bool | None = None
