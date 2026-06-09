"""Text normalization helpers used across data and inference."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable

_HTML_TAG = re.compile(r"<[^>]+>")
_NON_WORD = re.compile(r"[^a-z0-9+#.]+")
_WHITESPACE = re.compile(r"\s+")


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    return _WHITESPACE.sub(" ", html.unescape(_HTML_TAG.sub(" ", value))).strip()


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", strip_html(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", normalized)).strip()


def tokens(value: str | None) -> set[str]:
    return {token for token in normalize_text(value).split() if len(token) > 1}


def split_values(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = re.split(r"[|,;]", value)
    else:
        parts = list(value)
    return tuple(dict.fromkeys(normalize_text(str(item)) for item in parts if str(item).strip()))
