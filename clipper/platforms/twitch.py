"""Twitch — the Helix `streams` endpoint.

Needs `TWITCH_CLIENT_ID` and `TWITCH_TOKEN` in the environment. When either
is missing the call is refused before any HTTP request goes out.
"""
from __future__ import annotations

import os

from .base import HttpGet, LiveStatus, MissingCredentialError, json_get

STREAMS_URL = "https://api.twitch.tv/helix/streams?user_login={login}"


def is_live(
    channel: str,
    http_get: HttpGet,
    *,
    client_id: str | None = None,
    token: str | None = None,
) -> LiveStatus:
    """Is this Twitch channel live right now?"""
    client_id = client_id if client_id is not None else os.environ.get("TWITCH_CLIENT_ID")
    token = token if token is not None else os.environ.get("TWITCH_TOKEN")
    missing = [
        name
        for name, value in (("TWITCH_CLIENT_ID", client_id), ("TWITCH_TOKEN", token))
        if not value
    ]
    if missing:
        raise MissingCredentialError(
            f"missing Twitch credential(s): {', '.join(missing)} (set them in the environment)"
        )
    login = channel.strip().lstrip("@").lower()
    url = STREAMS_URL.format(login=login)
    headers = {"Client-Id": str(client_id), "Authorization": f"Bearer {token}"}
    data = json_get(http_get, url, headers)
    streams = data.get("data") or []
    if not streams:
        return LiveStatus(live=False)
    stream = streams[0]
    return LiveStatus(
        live=True,
        title=str(stream.get("title") or ""),
        started_at=str(stream.get("started_at") or ""),
        stream_url=str(stream.get("stream_url") or ""),
    )
