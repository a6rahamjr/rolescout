"""Rank a CSV export without running the API."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rolescout.data.contracts import JobPosting, SearchProfile
from rolescout.models.ranker import JobRanker
from rolescout.utils.config import load_config
from rolescout.utils.text import split_values

REQUIRED_COLUMNS = {"title"}


def rank_csv(
    input_path: str | Path,
    output_path: str | Path,
    profile: SearchProfile,
    model: JobRanker,
    *,
    limit: int | None = None,
    min_score: float = 0.0,
) -> Path:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input CSV not found: {source}")

    frame = pd.read_csv(source).fillna("")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    jobs = [
        JobPosting.from_mapping(
            {
                "job_id": row.get("job_id") or f"csv-{index}",
                "title": row["title"],
                "company": row.get("company", ""),
                "description": row.get("description", ""),
                "url": row.get("url", ""),
                "location": row.get("location", ""),
                "workplace": row.get("workplace", "unknown"),
                "experience_level": row.get("experience_level", ""),
                "job_type": row.get("job_type", ""),
                "posted_at": row.get("posted_at") or None,
                "skills": row.get("skills", ""),
                "salary": row.get("salary", ""),
                "source": row.get("source", "csv"),
            }
        )
        for index, row in frame.iterrows()
    ]
    ranked = model.rank(profile, jobs, limit=limit, min_score=min_score)
    rows = [
        {
            "rank": item.rank,
            "score": item.score,
            "match_level": item.match_level,
            "title": item.job.title,
            "company": item.job.company,
            "location": item.job.location,
            "workplace": item.job.workplace,
            "job_type": item.job.job_type,
            "url": item.job.url,
            "reasons": "; ".join(item.reasons),
            "concerns": "; ".join(item.concerns),
        }
        for item in ranked
    ]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank jobs from a CSV file")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ranked_jobs.csv"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--location", default="")
    parser.add_argument("--skills", default="")
    parser.add_argument("--experience", default="")
    parser.add_argument("--workplace", default="any")
    parser.add_argument("--job-types", default="")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--exclude-companies", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    profile = SearchProfile(
        query=args.query,
        location=args.location,
        skills=split_values(args.skills),
        experience_level=args.experience,
        workplace=args.workplace,
        job_types=split_values(args.job_types),
        excluded_keywords=split_values(args.exclude),
        excluded_companies=split_values(args.exclude_companies),
    )
    model = JobRanker.load(config.model.artifact_path)
    destination = rank_csv(
        args.input,
        args.output,
        profile,
        model,
        limit=args.limit,
        min_score=args.min_score,
    )
    print(f"Ranked jobs written to {destination}")


if __name__ == "__main__":
    main()
