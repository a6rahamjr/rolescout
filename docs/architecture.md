# Architecture

RoleScout keeps data acquisition, feature extraction, model training, ranking, and alert
state separate so each part can be replaced without rewriting the API.

## Request Flow

1. The API validates a search profile and either supplied jobs or a provider-backed search.
2. Provider responses are normalized into `JobPosting` records.
3. Exact IDs and canonical title/company pairs are deduplicated.
4. Excluded keywords and companies are removed before scoring.
5. `PairFeatureExtractor` builds text similarity and compatibility features.
6. `JobRanker` scores the batch and returns ranked jobs with feature contributions.
7. Alert checks persist delivered job IDs in SQLite to avoid repeat notifications.

## Package Boundaries

- `rolescout.data`: domain records, dataset validation, generation, and provider adapters.
- `rolescout.models`: feature extraction and the serialized ranking model.
- `rolescout.training`: grouped splitting, parameter selection, fitting, and artifact export.
- `rolescout.evaluation`: classification and query-grouped ranking metrics.
- `rolescout.inference`: service orchestration, alert state, and CSV workflows.
- `app`: HTTP request validation and route wiring.

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
Alert scheduling is deliberately external: cron, a systemd timer, or a task worker can
call the check-all endpoint. This avoids hidden background work inside API replicas.
