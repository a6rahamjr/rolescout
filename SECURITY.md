# Security

## Reporting

Please report security issues privately through GitHub's security advisory feature rather
than opening a public issue.

Include the affected version, reproduction steps, and the expected impact when possible.

## Artifact Safety

RoleScout model artifacts use `joblib`, which is based on Python pickle. Only load model
files produced by this project or obtained from a trusted source. A malicious artifact can
execute code when deserialized.

## Data Handling

The default dataset is synthetic. If you replace it with user feedback or search history,
remove personal information, document retention rules, and avoid committing raw production
data to the repository.

## Provider Configuration

Keep provider authorization values in environment variables or a managed secret store.
Do not add them to YAML files, Docker images, logs, issue reports, or sample requests.
Use only LinkedIn data obtained through an approved integration and preserve the original
listing URL and source attribution.
