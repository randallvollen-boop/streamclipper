"""YouTube — the Data API `search` endpoint with `eventType=live`.

Needs `YOUTUBE_API_KEY` in the environment. When it is missing the call is
refused before any HTTP request goes out.

The Data API needs a channel id, so a handle (`@ExampleTube`) is resolved to
one with `channels?forHandle=` first; a value that already looks like a
channel id is used as it is.
"""
from __future__ import annotations

import os
import urllib.parse

from .base import HttpGet, LiveStatus, MissingCredentialError, PlatformError, json_get

SEARCH_URL = (
    "https://www.googleapis.com/youtube/v3/search"
    "?channelId={channel_id}&eventType=live&type=video&key={key}"
)
CHANNELS_URL = (
    "https://www.googleapis.com/youtube/v3/channels?part=id&forHandle={handle}&key={key}"
)


def is_live(
    channel: str,
    http_get: HttpGet,
    *,
    api_key: str | None = None,
) -> LiveStatus:
    """Is this YouTube channel live right now?"""
    api_key = api_key if api_key is not None else os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise MissingCredentialError(
            "missing YouTube credential: YOUTUBE_API_KEY (set it in the environment)"
        )
    channel_id = resolve_channel_id(channel, http_get, api_key)
    url = SEARCH_URL.format(channel_id=channel_id, key=api_key)
    data = json_get(http_get, url)
    items = data.get("items") or []
    if not items:
        return LiveStatus(live=False)
    item = items[0]
    snippet = item.get("snippet") or {}
    video_id = str((item.get("id") or {}).get("videoId") or "")
    return LiveStatus(
        live=True,
        title=str(snippet.get("title") or ""),
        started_at=str(snippet.get("publishedAt") or ""),
        stream_url=f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
    )


def resolve_channel_id(channel: str, http_get: HttpGet, api_key: str) -> str:
    """Turn a handle into a channel id (or pass an id through)."""
    raw = channel.strip().lstrip("@")
    if raw.startswith("UC") and len(raw) >= 20:
        return raw
    url = CHANNELS_URL.format(handle=urllib.parse.quote(raw, safe=""), key=api_key)
    data = json_get(http_get, url)
    items = data.get("items") or []
    if not items:
        raise PlatformError(f"YouTube channel {channel!r} not found")
    channel_id = str((items[0] or {}).get("id") or "")
    if not channel_id:
        raise PlatformError(f"YouTube channel {channel!r} has no channel id")
    return channel_id
