# RoleScout

RoleScout ranks job listings against a search profile and explains why each result landed
where it did. It can be used as a Python package, a small HTTP service, or a command-line
tool for ranking a CSV export.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## What It Does

- Ranks supplied job listings by title, description, skills, location, seniority,
  workplace preference, job type, recency, and listing quality.
- Fetches and reranks listings from Remotive and an optional approved LinkedIn feed.
- Streams newly discovered ranked jobs over server-sent events.
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

Results retain their original links and provider attribution. Remotive works out of the
box; LinkedIn is included when an approved feed is configured.

## Stream New Jobs

`/v1/stream` keeps an HTTP connection open and emits each newly discovered ranked job once:

```bash
curl -N -G http://localhost:8000/v1/stream \
  --data-urlencode "query=python backend engineer" \
  --data-urlencode "skills=python,fastapi,postgresql" \
  --data-urlencode "workplace=remote" \
  --data-urlencode "min_score=0.4" \
  --data-urlencode "poll_seconds=30" \
  --data-urlencode "include_existing=false"
```

The stream uses standard server-sent events:

- `ready`: confirms the active sources and polling interval.
- `job`: contains a ranked job that has not appeared on this connection before.
- `provider_error`: reports a recoverable provider outage without closing the stream.
- comment heartbeats keep idle connections alive.

Browser clients can connect with `EventSource`:

```javascript
const params = new URLSearchParams({
  query: "python backend engineer",
  workplace: "remote",
  include_existing: "false",
});
const stream = new EventSource(`/v1/stream?${params}`);

stream.addEventListener("job", (event) => {
  const { result } = JSON.parse(event.data);
  console.log(result.job.title, result.score);
});
```

Delivery is near real time and depends on how quickly each upstream provider publishes
and exposes new listings. RoleScout polls every 30 seconds by default and supports
intervals from 10 to 300 seconds.

## LinkedIn Feed

LinkedIn's general developer APIs do not provide an unrestricted public job-search
firehose. Its Talent APIs require approval, and the Job Posting API sends jobs to LinkedIn
rather than searching LinkedIn jobs. RoleScout therefore connects through a configurable
approved partner feed or organization-owned gateway instead of scraping pages.

Enable it at deployment time:

```env
ROLESCOUT_LINKEDIN_ENABLED=true
ROLESCOUT_LINKEDIN_FEED_URL=https://jobs-gateway.example.com/linkedin
ROLESCOUT_LINKEDIN_BEARER_TOKEN=
```

The bearer value is optional and read only from the environment. The feed contract and
accepted field aliases are documented in
[docs/linkedin-feed.md](docs/linkedin-feed.md).

Official access references:

- [Getting access to LinkedIn APIs](https://learn.microsoft.com/en-us/linkedin/shared/authentication/getting-access)
- [LinkedIn Job Posting API](https://learn.microsoft.com/en-us/linkedin/talent/job-postings/api/overview)

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

The default dataset is synthetic and is intended for checking the engineering pipeline,
leakage controls, serialization, and ranking behavior. Its near-perfect score is not
evidence of real-world quality.

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
ROLESCOUT_LINKEDIN_ENABLED=false
ROLESCOUT_LINKEDIN_FEED_URL=
ROLESCOUT_LINKEDIN_BEARER_TOKEN=
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
