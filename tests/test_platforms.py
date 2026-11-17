"""The platform adapters, tested with mocked responses only.

No test here touches Kick, Twitch or YouTube. The HTTP call is injected.
"""
from __future__ import annotations

import json

import pytest

from clipper.platforms import kick, twitch, youtube
from clipper.platforms.base import MissingCredentialError, PlatformError


def fake_get(routes: dict[str, object], calls: list | None = None):
    """A fake `http_get`: routes are matched by URL substring."""

    def _get(url: str, headers: dict | None = None) -> bytes:
        if calls is not None:
            calls.append((url, headers))
        for fragment, payload in routes.items():
            if fragment in url:
                if isinstance(payload, str) and payload.startswith("!"):
                    raise PlatformError(payload)
                return json.dumps(payload).encode("utf-8")
        raise AssertionError(f"unexpected URL: {url}")

    return _get


# --- kick --------------------------------------------------------------------


def test_kick_offline_when_livestream_is_null():
    status = kick.is_live("examplestreamer", fake_get({"api/v2/channels": {"livestream": None}}))
    assert status.live is False
    assert status.title == ""


def test_kick_live_reports_the_session():
    payload = {
        "livestream": {
            "id": "123",
            "session_title": "Ranked grind",
            "start_time": "2026-11-16T20:00:00Z",
            "is_live": True,
        }
    }
    status = kick.is_live("examplestreamer", fake_get({"api/v2/channels": payload}))
    assert status.live is True
    assert status.title == "Ranked grind"
    assert status.started_at == "2026-11-16T20:00:00Z"
    assert status.stream_url == "https://kick.com/examplestreamer"


def test_kick_calls_the_channel_endpoint():
    calls: list = []
    kick.is_live("examplestreamer", fake_get({"api/v2/channels": {"livestream": None}}, calls))
    assert calls[0][0] == "https://kick.com/api/v2/channels/examplestreamer"


# --- twitch ------------------------------------------------------------------


def test_twitch_offline_when_data_is_empty():
    status = twitch.is_live(
        "someone",
        fake_get({"helix/streams": {"data": []}}),
        client_id="id",
        token="tok",
    )
    assert status.live is False


def test_twitch_live_reports_the_stream():
    payload = {
        "data": [
            {
                "id": "1",
                "title": "Just chatting",
                "started_at": "2026-11-16T20:00:00Z",
                "stream_url": "https://example.test/live.m3u8",
            }
        ]
    }
    status = twitch.is_live(
        "someone", fake_get({"helix/streams": payload}), client_id="id", token="tok"
    )
    assert status.live is True
    assert status.title == "Just chatting"
    assert status.stream_url == "https://example.test/live.m3u8"


def test_twitch_sends_the_credentials():
    calls: list = []
    twitch.is_live(
        "someone", fake_get({"helix/streams": {"data": []}}, calls), client_id="id", token="tok"
    )
    _, headers = calls[0]
    assert headers["Client-Id"] == "id"
    assert headers["Authorization"] == "Bearer tok"


def test_twitch_missing_credentials_is_a_clear_error_and_makes_no_call(monkeypatch):
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_TOKEN", raising=False)
    calls: list = []

    def explode(url, headers=None):  # pragma: no cover - must not run
        calls.append(url)
        raise AssertionError("the HTTP call must not happen without credentials")

    with pytest.raises(MissingCredentialError, match="TWITCH_CLIENT_ID"):
        twitch.is_live("someone", explode)
    assert calls == []


# --- youtube -----------------------------------------------------------------


def test_youtube_resolves_a_handle_then_searches():
    routes = {
        "channels?part=id": {"items": [{"id": "UC1234567890123456789012"}]},
        "search?": {
            "items": [
                {
                    "id": {"videoId": "abc123"},
                    "snippet": {"title": "Live now!", "publishedAt": "2026-11-16T20:00:00Z"},
                }
            ]
        },
    }
    status = youtube.is_live("@ExampleTube", fake_get(routes), api_key="key")
    assert status.live is True
    assert status.title == "Live now!"
    assert status.stream_url == "https://www.youtube.com/watch?v=abc123"


def test_youtube_offline_when_no_live_items():
    routes = {
        "channels?part=id": {"items": [{"id": "UC1234567890123456789012"}]},
        "search?": {"items": []},
    }
    status = youtube.is_live("@ExampleTube", fake_get(routes), api_key="key")
    assert status.live is False


def test_youtube_skips_the_lookup_for_a_channel_id():
    calls: list = []
    routes = {"search?": {"items": []}}
    status = youtube.is_live(
        "UC1234567890123456789012", fake_get(routes, calls), api_key="key"
    )
    assert status.live is False
    assert len(calls) == 1
    assert "channelId=UC1234567890123456789012" in calls[0][0]


def test_youtube_missing_api_key_is_a_clear_error_and_makes_no_call(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    calls: list = []

    def explode(url, headers=None):  # pragma: no cover - must not run
        calls.append(url)
        raise AssertionError("the HTTP call must not happen without an API key")

    with pytest.raises(MissingCredentialError, match="YOUTUBE_API_KEY"):
        youtube.is_live("@ExampleTube", explode)
    assert calls == []


def test_youtube_unknown_channel_is_an_error():
    routes = {"channels?part=id": {"items": []}}
    with pytest.raises(PlatformError, match="not found"):
        youtube.is_live("@nobody", fake_get(routes), api_key="key")
