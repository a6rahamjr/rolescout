# RoleScout

RoleScout ranks job listings against a search profile and explains why each result landed
where it did. It can be used as a Python package, a small HTTP service, or a command-line
tool for ranking a CSV export.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## What It Does

- Ranks supplied job listings by title, description, skills, location, seniority,
  workplace preference, job type, recency, and listing quality.
- Fetches remote listings from Remotive and reranks them.
- Explains positive signals and likely tradeoffs for every result.
- Filters by minimum score, excluded keywords, and excluded companies.
- Stores alerts in SQLite with pause/resume, editing, delivery history, and check-all.
- Ranks ordinary CSV exports without requiring the API.
- Trains and evaluates the included model from a reproducible dataset.

## Setup

RoleScout requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The repository includes a trained artifact. To rebuild it:

```bash
python scripts/train.py --regenerate
```

## Run The API

```bash
uvicorn app.main:app --reload
```

Useful URLs:

- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Model details: `http://localhost:8000/v1/model`

### Rank Supplied Jobs

```bash
curl -X POST http://localhost:8000/v1/rank \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "query": "senior backend engineer",
      "location": "Berlin",
      "skills": ["python", "postgresql", "fastapi"],
      "workplace": "remote",
      "excluded_keywords": ["wordpress"],
      "excluded_companies": ["Example Staffing"]
    },
    "min_score": 0.45,
    "jobs": [{
      "job_id": "job-1",
      "title": "Senior Python Backend Engineer",
      "company": "Acme",
      "description": "Build APIs and data services with FastAPI and PostgreSQL.",
      "url": "https://example.com/jobs/1",
      "location": "Remote",
      "workplace": "remote",
      "experience_level": "senior",
      "job_type": "full time",
      "skills": ["python", "postgresql", "fastapi"]
    }]
  }'
```

Each result includes:

- `score`: model probability between 0 and 1.
- `match_level`: `strong`, `possible`, or `weak`.
- `reasons`: the strongest positive signals.
- `concerns`: the strongest mismatches or missing information.
- `contributions`: all feature-level model contributions.

### Search The Remote Feed

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "query": "python backend engineer",
      "skills": ["python", "fastapi"],
      "workplace": "remote"
    },
    "limit": 10,
    "min_score": 0.4
  }'
```

Live listings are sourced from Remotive and retain their original links and attribution.

## Alerts

Create an alert:

```bash
curl -X POST http://localhost:8000/v1/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Remote Python roles",
    "profile": {
      "query": "python backend engineer",
      "workplace": "remote"
    },
    "min_score": 0.6
  }'
```

Common operations:

```text
GET    /v1/alerts
GET    /v1/alerts/{id}
PATCH  /v1/alerts/{id}
DELETE /v1/alerts/{id}
POST   /v1/alerts/{id}/check
DELETE /v1/alerts/{id}/deliveries
POST   /v1/alerts/check-all
```

Pause an alert:

```bash
curl -X PATCH http://localhost:8000/v1/alerts/1 \
  -H "Content-Type: application/json" \
  -d '{"active": false}'
```

Checks only return jobs that have not already been delivered. Clearing delivery history
makes previously seen jobs eligible again. Scheduling is intentionally left outside the
web process; call `check-all` from cron, a systemd timer, or a task worker.

## Rank A CSV File

The input only requires a `title` column. Optional columns are `job_id`, `company`,
`description`, `url`, `location`, `workplace`, `experience_level`, `job_type`,
`posted_at`, `skills`, `salary`, and `source`.

```bash
python scripts/rank_csv.py jobs.csv \
  --query "backend engineer" \
  --skills "python,fastapi,postgresql" \
  --workplace remote \
  --exclude wordpress \
  --min-score 0.4 \
  --output ranked_jobs.csv
```

## Training And Evaluation

```bash
python scripts/generate_data.py
python scripts/train.py
python scripts/evaluate.py
```

Generated files:

```text
data/processed/job_matches.csv
artifacts/job_ranker.joblib
artifacts/metrics.json
```

The default dataset is synthetic because the original project had no labeled relevance
data. It is useful for checking the engineering pipeline, leakage controls, serialization,
and ranking behavior. Its near-perfect score is not evidence of real-world quality.

For a real deployment, replace it with consented search impressions and relevance
feedback using the schema in `rolescout.data.loaders`.

## Project Layout

```text
app/                         HTTP routes and request validation
configs/                     Runtime and training settings
scripts/                     Generate, train, evaluate, and rank CSV files
src/rolescout/data/          Domain contracts, dataset code, provider adapters
src/rolescout/models/        Feature extraction and ranking model
src/rolescout/training/      Grouped model selection and artifact export
src/rolescout/evaluation/    Classification and ranking metrics
src/rolescout/inference/     Ranking, alerts, and CSV workflows
tests/                       Data, model, API, alert, and CLI tests
```

Implementation details are in [docs/architecture.md](docs/architecture.md).

## Configuration

Defaults live in `configs/default.yaml`. These environment variables override common
deployment paths:

```env
ROLESCOUT_CONFIG=configs/default.yaml
ROLESCOUT_MODEL_PATH=artifacts/job_ranker.joblib
ROLESCOUT_DATABASE_PATH=rolescout.db
ROLESCOUT_LOG_LEVEL=INFO
```

## Checks

```bash
python -m pytest
ruff check src app scripts tests
```

## Docker

```bash
docker build -t rolescout .
docker run --rm -p 8000:8000 rolescout
```

## Responsible Use

RoleScout ranks opportunities for job seekers. It is not designed to rank candidates,
infer personal traits, or automate employment decisions.

## License

MIT
