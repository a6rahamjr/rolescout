# Architecture

RoleScout keeps data acquisition, feature extraction, model training, ranking, and alert
state separate so each part can be replaced without rewriting the API.

## Request Flow

1. The API validates a search profile and either supplied jobs or a provider-backed search.
2. Configured providers are queried concurrently and normalized into `JobPosting` records.
3. Exact IDs and canonical title/company pairs are deduplicated.
4. Provider-backed searches require title and query-intent overlap.
5. Excluded keywords and companies are removed before scoring.
6. `PairFeatureExtractor` builds text similarity and compatibility features.
7. `JobRanker` scores the batch and returns ranked jobs with feature contributions.
8. Alert checks persist delivered job IDs in SQLite to avoid repeat notifications.
9. Live streams poll the same ranking service and emit unseen results over SSE.

## Package Boundaries

- `rolescout.data`: domain records, dataset validation, generation, and provider adapters.
- `rolescout.models`: feature extraction and the serialized ranking model.
- `rolescout.training`: grouped splitting, parameter selection, fitting, and artifact export.
- `rolescout.evaluation`: classification and query-grouped ranking metrics.
- `rolescout.inference`: service orchestration, alert state, and CSV workflows.
- `app`: HTTP request validation and route wiring.

## Providers

`CompositeProvider` queries Remotive and any enabled LinkedIn feed concurrently. A
provider failure is isolated when another source still returns results. Jobs are ordered
by posting time and deduplicated before entering the ranker.

The LinkedIn adapter targets an approved feed or organization gateway because LinkedIn
does not expose unrestricted public job search through its general APIs. Its endpoint and
authorization value are deployment settings; neither is returned by health checks.

## Live Delivery

`GET /v1/stream` returns `text/event-stream`. Each connection keeps a bounded in-memory
set of job IDs, emits newly discovered results once, and sends heartbeats while idle.
Provider errors are delivered as recoverable events so clients can stay connected.

The stream is stateless across replicas and reconnects. Clients that require durable,
exactly-once delivery should use saved alerts and persist their own last event ID.

## Model

The current model is logistic regression over fourteen features:

- word and character TF-IDF similarity for titles and descriptions;
- query coverage in titles and descriptions;
- skill overlap;
- location, workplace, seniority, and job-type compatibility;
- recency, salary availability, and description quality.

The artifact includes fitted vectorizers, the feature scaler, classifier, metadata, and
holdout metrics. Artifacts use `joblib`; they must only be loaded from trusted sources.

## Data

The bundled dataset is deterministic and synthetic. Query IDs are split as groups during
training, so candidate rows from the same search cannot appear in both development and
test sets. Real deployments should replace the synthetic records with consented relevance
feedback while preserving the validated schema.

## Storage

SQLite is the default alert store because it keeps local setup small. The repository class
contains the SQL boundary, making a PostgreSQL implementation possible without changing
the ranking service or HTTP contracts.

## Deployment

The web process loads one immutable model artifact and serves requests through FastAPI.
Live polling only runs while a client stream is connected. Alert scheduling remains
external: cron, a systemd timer, or a task worker can call the check-all endpoint.
