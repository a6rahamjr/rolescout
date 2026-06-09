from __future__ import annotations

import pandas as pd

from rolescout.data.contracts import SearchProfile
from rolescout.inference.csv_runner import rank_csv


def test_csv_runner_writes_ranked_output(trained_result, tmp_path) -> None:
    source = tmp_path / "jobs.csv"
    destination = tmp_path / "ranked.csv"
    pd.DataFrame(
        [
            {
                "job_id": "design",
                "title": "Product Designer",
                "company": "Vertex Studio",
                "description": "Design interfaces in Figma.",
            },
            {
                "job_id": "backend",
                "title": "Backend Engineer",
                "company": "Orbit Works",
                "description": "Build Python APIs with PostgreSQL.",
                "skills": "python|postgresql|fastapi",
            },
        ]
    ).to_csv(source, index=False)

    output = rank_csv(
        source,
        destination,
        SearchProfile(query="backend engineer", skills=("python", "fastapi")),
        trained_result.model,
    )
    ranked = pd.read_csv(output)

    assert output == destination
    assert ranked.iloc[0]["title"] == "Backend Engineer"
    assert {"rank", "score", "match_level", "reasons", "concerns"}.issubset(ranked.columns)
