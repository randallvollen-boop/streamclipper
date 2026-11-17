"""The platform adapters — one `is_live()` per platform.

Each adapter takes the HTTP call as an argument, so the whole platform layer
is testable with mocked responses and this project never has to touch a
streamer site to be tested.
"""
from __future__ import annotations

from . import kick, twitch, youtube
from .base import (
    HTTP_TIMEOUT,
    USER_AGENT,
    HttpGet,
    LiveStatus,
    MissingCredentialError,
    PlatformError,
    json_get,
    urllib_get,
)

#: platform name -> the adapter that answers "is this channel live?"
ADAPTERS = {
    "kick": kick.is_live,
    "twitch": twitch.is_live,
    "youtube": youtube.is_live,
}

__all__ = [
    "ADAPTERS",
    "HTTP_TIMEOUT",
    "USER_AGENT",
    "HttpGet",
    "LiveStatus",
    "MissingCredentialError",
    "PlatformError",
    "json_get",
    "kick",
    "twitch",
    "urllib_get",
    "youtube",
]
