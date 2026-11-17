"""Shared plumbing for the platform adapters.

Every adapter is a function `is_live(channel, http_get) -> LiveStatus`. The
HTTP call is injected so the adapters can be tested with mocked responses —
this project never calls a streamer site in tests.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

#: The User-Agent every real HTTP call carries.
USER_AGENT = "StreamClipper/0.1"
#: The timeout for a real HTTP call, seconds.
HTTP_TIMEOUT = 10.0


class PlatformError(Exception):
    """A platform call failed in a way the operator needs to know about."""


class MissingCredentialError(PlatformError):
    """A platform credential is not configured. No HTTP call is made."""


@dataclass(frozen=True)
class LiveStatus:
    """What we know about one channel's live state right now."""

    live: bool
    title: str = ""
    started_at: str = ""
    stream_url: str = ""
    #: The platform's own numeric id of the channel (Kick has one, and the
    #: chat replay needs it).
    channel_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "live": self.live,
            "title": self.title,
            "started_at": self.started_at,
            "stream_url": self.stream_url,
            "channel_id": self.channel_id,
        }


#: The signature every adapter's `http_get` must have.
HttpGet = Callable[..., bytes]


def urllib_get(url: str, headers: dict[str, str] | None = None, timeout: float = HTTP_TIMEOUT) -> bytes:
    """The real HTTP call: urllib, our User-Agent, a 10 s timeout."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise PlatformError(f"{url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise PlatformError(f"{url}: {exc.reason}") from exc


def json_get(http_get: HttpGet, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Call `http_get` and parse the body as a JSON object."""
    raw = http_get(url, headers)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlatformError(f"{url}: response is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise PlatformError(f"{url}: expected a JSON object")
    return parsed
