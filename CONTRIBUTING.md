# Contributing

Thanks for taking the time to improve RoleScout.

## Local Setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Activate the virtual environment using the command for your shell, then run:

```bash
ruff check src app scripts tests
ruff format --check src app scripts tests
python -m pytest
```

## Pull Requests

- Keep changes focused and explain the behavior they change.
- Add or update tests for user-visible behavior.
- Do not commit local databases, credentials, logs, or ad hoc job exports.
- Preserve provider attribution and original listing links.
- Add provider contract tests for normalization, partial outages, and deduplication.
- Call out changes to the dataset schema or model artifact format.

Synthetic benchmark results are useful for regression testing, but they must not be
presented as evidence of real-world hiring quality.
