from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rolescout.data.contracts import JobPosting, SearchProfile


def test_inference_ranks_matching_job_first_and_explains_score(trained_result) -> None:
    profile = SearchProfile(
        query="senior machine learning engineer",
        location="Berlin, Germany",
        skills=("python", "pytorch", "mlops"),
        experience_level="senior",
        workplace="remote",
        job_types=("full time",),
    )
    now = datetime.now(UTC)
    matching = JobPosting(
        job_id="match",
        title="Senior Machine Learning Engineer",
        company="Northstar Labs",
        description="Build production machine learning models with Python and PyTorch.",
        url="https://example.com/match",
        location="Remote",
        workplace="remote",
        experience_level="senior",
        job_type="full time",
        posted_at=now,
        skills=("python", "pytorch", "mlops"),
    )
    irrelevant = JobPosting(
        job_id="irrelevant",
        title="Product Designer",
        company="Vertex Studio",
        description="Create design systems and conduct usability research in Figma.",
        url="https://example.com/irrelevant",
        location="New York, USA",
        workplace="onsite",
        experience_level="entry",
        job_type="contract",
        posted_at=now - timedelta(days=30),
        skills=("figma", "research"),
    )

    ranked = trained_result.model.rank(profile, [irrelevant, matching])

    assert [item.job.job_id for item in ranked] == ["match", "irrelevant"]
    assert ranked[0].score > ranked[1].score
    assert ranked[0].match_level in {"strong", "possible", "weak"}
    assert ranked[0].reasons
    assert isinstance(ranked[0].concerns, tuple)
    assert set(ranked[0].contributions) == set(trained_result.model.extractor.feature_names)


def test_inference_deduplicates_jobs(trained_result) -> None:
    profile = SearchProfile(query="backend engineer")
    job = JobPosting(
        job_id="same",
        title="Backend Engineer",
        company="Orbit Works",
        description="Build Python APIs.",
        url="https://example.com/1",
    )

    ranked = trained_result.model.rank(profile, [job, job])

    assert len(ranked) == 1


def test_inference_applies_exclusions_and_score_threshold(trained_result) -> None:
    profile = SearchProfile(
        query="backend engineer",
        excluded_keywords=("wordpress",),
        excluded_companies=("blocked corp",),
    )
    jobs = [
        JobPosting(
            job_id="keep",
            title="Backend Engineer",
            company="Orbit Works",
            description="Build Python APIs.",
            url="https://example.com/keep",
        ),
        JobPosting(
            job_id="keyword-block",
            title="Backend Engineer",
            company="Northstar Labs",
            description="Maintain WordPress websites.",
            url="https://example.com/keyword-block",
        ),
        JobPosting(
            job_id="company-block",
            title="Backend Engineer",
            company="Blocked Corp",
            description="Build Python APIs.",
            url="https://example.com/company-block",
        ),
    ]

    ranked = trained_result.model.rank(profile, jobs, min_score=0.0)
    assert [item.job.job_id for item in ranked] == ["keep"]

    filtered = trained_result.model.rank(profile, jobs, min_score=1.0)
    assert filtered == []
