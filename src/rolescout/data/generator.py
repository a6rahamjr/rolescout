"""Seeded synthetic job relevance dataset generator."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from rolescout.data.loaders import save_dataset
from rolescout.utils.config import load_config
from rolescout.utils.seeding import seed_everything

ROLE_FAMILIES = {
    "machine_learning": {
        "titles": ["Machine Learning Engineer", "ML Engineer", "Applied AI Engineer"],
        "generic_title": "Technical Specialist",
        "skills": ["python", "pytorch", "scikit learn", "sql", "mlops", "nlp"],
        "description": "build production machine learning systems and deploy predictive models",
    },
    "data_engineering": {
        "titles": ["Data Engineer", "Analytics Engineer", "Data Platform Engineer"],
        "generic_title": "Platform Specialist",
        "skills": ["python", "sql", "spark", "airflow", "dbt", "kafka"],
        "description": "design reliable data pipelines warehouses and streaming platforms",
    },
    "backend": {
        "titles": ["Backend Engineer", "Python Developer", "API Engineer"],
        "generic_title": "Software Specialist",
        "skills": ["python", "fastapi", "postgresql", "docker", "redis", "rest api"],
        "description": "develop scalable backend services APIs and distributed systems",
    },
    "frontend": {
        "titles": ["Frontend Engineer", "React Developer", "UI Engineer"],
        "generic_title": "Product Engineer",
        "skills": ["typescript", "react", "css", "html", "testing", "webpack"],
        "description": "build accessible responsive web applications and design systems",
    },
    "devops": {
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer"],
        "generic_title": "Infrastructure Specialist",
        "skills": ["kubernetes", "terraform", "aws", "linux", "python", "observability"],
        "description": "operate cloud infrastructure delivery pipelines and reliable services",
    },
    "product": {
        "titles": ["Product Manager", "Technical Product Manager", "Product Owner"],
        "generic_title": "Program Lead",
        "skills": ["roadmaps", "analytics", "experimentation", "agile", "sql", "discovery"],
        "description": "lead product discovery strategy experiments and cross functional delivery",
    },
    "design": {
        "titles": ["Product Designer", "UX Designer", "UX Researcher"],
        "generic_title": "Creative Specialist",
        "skills": ["figma", "prototyping", "research", "usability", "design systems"],
        "description": "research user needs and design intuitive digital product experiences",
    },
    "security": {
        "titles": ["Security Engineer", "Application Security Engineer", "SOC Analyst"],
        "generic_title": "Risk Specialist",
        "skills": ["siem", "threat modeling", "python", "cloud security", "incident response"],
        "description": "protect applications investigate threats and improve security controls",
    },
}

LOCATIONS = ["Berlin, Germany", "London, UK", "New York, USA", "Toronto, Canada", "Remote"]
WORKPLACES = ["remote", "hybrid", "onsite"]
EXPERIENCE_LEVELS = ["entry", "mid", "senior", "lead"]
JOB_TYPES = ["full time", "contract", "part time"]
COMPANIES = ["Northstar Labs", "Helio Systems", "Orbit Works", "Maple Cloud", "Vertex Studio"]


def _compatible_location(desired: str, workplace: str, actual: str) -> bool:
    return workplace == "remote" or desired == "Remote" or desired == actual


def _make_row(
    *,
    query_id: str,
    query: str,
    desired_location: str,
    profile_skills: list[str],
    desired_experience: str,
    workplace_preference: str,
    desired_job_type: str,
    job_id: str,
    family: dict[str, object],
    title: str,
    location: str,
    workplace: str,
    experience_level: str,
    job_type: str,
    skills: list[str],
    label: int,
    rng: np.random.Generator,
    now: datetime,
    description_override: str | None = None,
) -> dict[str, object]:
    posted_at = now - timedelta(days=int(rng.integers(0, 45)), hours=int(rng.integers(0, 24)))
    description = description_override or (
        f"We are hiring someone to {family['description']}. "
        f"You will work with {' '.join(skills)} in a collaborative team."
    )
    return {
        "query_id": query_id,
        "query": query,
        "desired_location": desired_location,
        "profile_skills": "|".join(profile_skills),
        "desired_experience": desired_experience,
        "workplace_preference": workplace_preference,
        "desired_job_types": desired_job_type,
        "job_id": job_id,
        "title": title,
        "company": str(rng.choice(COMPANIES)),
        "description": description,
        "location": location,
        "workplace": workplace,
        "experience_level": experience_level,
        "job_type": job_type,
        "posted_at": posted_at.isoformat(),
        "skills": "|".join(skills),
        "salary": "$90,000 - $140,000" if rng.random() > 0.35 else "",
        "url": f"https://jobs.example/{job_id}",
        "source": "synthetic",
        "label": label,
    }


def generate_synthetic_dataset(
    n_queries: int = 300,
    candidates_per_query: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    if n_queries < 10:
        raise ValueError("n_queries must be at least 10")
    if candidates_per_query < 4:
        raise ValueError("candidates_per_query must be at least 4")

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    family_names = tuple(ROLE_FAMILIES)
    now = datetime(2026, 6, 1, 12, tzinfo=UTC)
    rows: list[dict[str, object]] = []

    for query_index in range(n_queries):
        family_name = str(rng.choice(family_names))
        family = ROLE_FAMILIES[family_name]
        desired_experience = str(rng.choice(EXPERIENCE_LEVELS))
        desired_location = str(rng.choice(LOCATIONS))
        workplace = str(rng.choice(WORKPLACES, p=[0.45, 0.35, 0.20]))
        desired_job_type = str(rng.choice(JOB_TYPES, p=[0.75, 0.18, 0.07]))
        profile_skills = list(rng.choice(family["skills"], size=3, replace=False))
        base_title = str(rng.choice(family["titles"]))
        query = f"{desired_experience} {base_title}"
        query_id = f"query-{query_index:04d}"

        positive_specs = [
            (base_title, desired_location, workplace, desired_experience, desired_job_type),
            (
                str(family["generic_title"]),
                desired_location,
                workplace,
                desired_experience,
                desired_job_type,
            ),
        ]
        for candidate_index, spec in enumerate(positive_specs):
            title, location, job_workplace, level, job_type = spec
            skills = list(dict.fromkeys(profile_skills + list(rng.choice(family["skills"], 2))))
            rows.append(
                _make_row(
                    query_id=query_id,
                    query=query,
                    desired_location=desired_location,
                    profile_skills=profile_skills,
                    desired_experience=desired_experience,
                    workplace_preference=workplace,
                    desired_job_type=desired_job_type,
                    job_id=f"{query_id}-job-{candidate_index}",
                    family=family,
                    title=title,
                    location=location,
                    workplace=job_workplace,
                    experience_level=level,
                    job_type=job_type,
                    skills=skills,
                    label=1,
                    rng=rng,
                    now=now,
                )
            )

        remaining = candidates_per_query - len(positive_specs)
        for offset in range(remaining):
            candidate_index = offset + len(positive_specs)
            if offset == 0:
                negative_family_name = family_name
                negative_family = family
                wrong_level = str(
                    rng.choice(
                        [
                            level
                            for level in EXPERIENCE_LEVELS
                            if level != desired_experience
                        ]
                    )
                )
                title = base_title
                location = str(rng.choice([item for item in LOCATIONS if item != desired_location]))
                job_workplace = "onsite" if workplace == "remote" else workplace
                skills = list(rng.choice(family["skills"], size=2, replace=False))
                description_override = (
                    f"Role focused on {family['description']}, but requires {wrong_level} "
                    f"experience and attendance in {location}."
                )
            else:
                negative_family_name = str(
                    rng.choice([name for name in family_names if name != family_name])
                )
                negative_family = ROLE_FAMILIES[negative_family_name]
                wrong_level = str(rng.choice(EXPERIENCE_LEVELS))
                if family_name in {"machine_learning", "data_engineering"} and offset == 1:
                    title = "Data Entry Specialist"
                else:
                    title = str(rng.choice(negative_family["titles"]))
                location = str(rng.choice(LOCATIONS))
                job_workplace = str(rng.choice(WORKPLACES))
                skills = list(rng.choice(negative_family["skills"], size=4, replace=False))
                description_override = None

            rows.append(
                _make_row(
                    query_id=query_id,
                    query=query,
                    desired_location=desired_location,
                    profile_skills=profile_skills,
                    desired_experience=desired_experience,
                    workplace_preference=workplace,
                    desired_job_type=desired_job_type,
                    job_id=f"{query_id}-job-{candidate_index}",
                    family=negative_family,
                    title=title,
                    location=location,
                    workplace=job_workplace,
                    experience_level=wrong_level,
                    job_type=str(rng.choice(JOB_TYPES)),
                    skills=skills,
                    label=0,
                    rng=rng,
                    now=now,
                    description_override=description_override,
                )
            )

    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the RoleScout synthetic dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--queries", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = generate_synthetic_dataset(
        n_queries=args.queries or config.data.n_queries,
        candidates_per_query=args.candidates or config.data.candidates_per_query,
        seed=args.seed if args.seed is not None else config.random_seed,
    )
    output = save_dataset(frame, args.output or config.data.synthetic_output)
    print(f"Generated {len(frame)} query-job pairs at {output}")


if __name__ == "__main__":
    main()
