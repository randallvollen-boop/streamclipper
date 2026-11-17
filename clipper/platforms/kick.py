"""Kick — `GET https://kick.com/api/v2/channels/<slug>`.

A null `livestream` means the channel is offline.
"""
from __future__ import annotations

from .base import HttpGet, LiveStatus, json_get

CHANNEL_URL = "https://kick.com/api/v2/channels/{slug}"


def is_live(channel: str, http_get: HttpGet) -> LiveStatus:
    """Is this Kick channel live right now?"""
    slug = channel.strip().lstrip("@")
    url = CHANNEL_URL.format(slug=slug)
    data = json_get(http_get, url)
    # The channel's own numeric id. The chat replay needs it, so it travels
    # with the live status and ends up on the recording plan.
    channel_id = str(data.get("id") or "")
    livestream = data.get("livestream")
    if not livestream:
        return LiveStatus(live=False, channel_id=channel_id)
    if livestream.get("is_live") is False:
        return LiveStatus(live=False, channel_id=channel_id)
    return LiveStatus(
        live=True,
        title=str(livestream.get("session_title") or ""),
        started_at=str(livestream.get("start_time") or ""),
        # streamlink resolves the channel page itself.
        stream_url=f"https://kick.com/{slug}",
        channel_id=channel_id,
    )
