# Changelog

All notable changes are documented here.

## 1.2.0 - 2026-06-09

- Added concurrent multi-provider search and cross-provider deduplication.
- Added an optional approved LinkedIn feed adapter with environment-only authorization.
- Added a server-sent events endpoint for near-real-time ranked job delivery.
- Added dynamic provider health metadata and search attribution.
- Added a query-intent gate so broad provider results do not pollute live searches.
- Added provider normalization, outage, deduplication, and streaming tests.

## 1.1.0 - 2026-06-09

- Added job and company exclusion filters.
- Added score thresholds, match levels, reasons, and concerns.
- Added alert editing, pause/resume, delivery reset, and batch checks.
- Added model metadata and CSV ranking commands.
- Reduced repeated feature extraction during inference.
- Added repository security, contribution, issue, and pull request templates.

## 1.0.0 - 2026-06-08

- Added the initial ranking model, training pipeline, Remotive provider, FastAPI service,
  SQLite alerts, evaluation metrics, and test suite.
