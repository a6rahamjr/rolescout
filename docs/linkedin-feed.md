# LinkedIn Feed Contract

RoleScout accepts LinkedIn jobs from an approved partner integration or an
organization-owned gateway. It does not scrape LinkedIn pages.

## Request

The provider sends an HTTP `GET` request to `ROLESCOUT_LINKEDIN_FEED_URL` with:

| Parameter | Description |
| --- | --- |
| `query` | Search text from the RoleScout profile |
| `location` | Desired location, or an empty string |
| `limit` | Maximum requested records |

When `ROLESCOUT_LINKEDIN_BEARER_TOKEN` is set, the request includes an
`Authorization: Bearer ...` header.

## Response

The response may be a JSON array or an object containing an array under `jobs`,
`results`, `elements`, or `data`.

```json
{
  "jobs": [
    {
      "jobPostingId": "123",
      "jobTitle": "Senior Backend Engineer",
      "companyName": "Example Company",
      "jobDescription": "Build and operate Python services.",
      "jobUrl": "https://www.linkedin.com/jobs/view/123",
      "formattedLocation": "Berlin, Germany",
      "workplaceType": "REMOTE",
      "employmentStatus": "FULL_TIME",
      "listedAt": 1710000000000,
      "skillNames": ["Python", "PostgreSQL"]
    }
  ]
}
```

Common snake_case alternatives are accepted for IDs, titles, company names,
descriptions, URLs, locations, workplace type, employment type, posting time, salary,
and skills. Posting times may be ISO 8601 strings or Unix timestamps in seconds or
milliseconds.

## Operations

- Keep the feed URL in deployment configuration and secret values in the environment.
- Return original listing URLs so attribution and application links remain intact.
- Respect the upstream provider's request limits when selecting the stream interval.
- Use HTTPS in production.
- Rotate authorization values without rebuilding the application image.

LinkedIn API access is product-specific and most Talent integrations require explicit
approval. See LinkedIn's official
[API access guide](https://learn.microsoft.com/en-us/linkedin/shared/authentication/getting-access)
before connecting production data.
