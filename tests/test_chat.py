"""The chat replay fetcher.

Nothing here touches Kick: the HTTP call is injected, exactly as the
platform adapters do it. The fake answers like the real endpoint: a call
with `start_time` returns the messages of the next five seconds.
"""
from __future__ import annotations

import datetime as dt
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from clipper import chat
from clipper.allowlist import Conditions, Proof, Streamer
from clipper.chat import ChatRefused, fetch_chat, fetch_kick_chat, write_chat
from clipper.platforms.base import PlatformError

BASE = dt.datetime(2026, 11, 14, 1, 0, 0, tzinfo=dt.timezone.utc)


def make_streamer(
    streamer_id="examplestreamer",
    platform="kick",
    status="verified",
    proof=True,
) -> Streamer:
    return Streamer(
        id=streamer_id,
        name=streamer_id.capitalize(),
        platform=platform,
        channel=streamer_id,
        status=status,
        proof=(
            (Proof(url="https://example.test/brief", kind="program-brief", checked="2026-11-16"),)
            if proof
            else ()
        ),
        conditions=Conditions(),
    )


def message(index: int, at: dt.datetime, text="W", user="viewer") -> dict:
    """One message, as the API sends it."""
    return {
        "id": f"m{index}",
        "content": text,
        "created_at": at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "sender": {"username": user},
    }


def fake_api(all_messages, calls=None, fail_from=None, always=None):
    """A fake `http_get` that answers like the Kick messages endpoint.

    `fail_from` is the number of calls after which the endpoint starts
    failing; `always` is a list of messages every call returns (to make a
    boundary repeat itself).
    """
    sent = calls if calls is not None else []

    def _get(url: str, headers: dict | None = None) -> bytes:
        sent.append(url)
        if fail_from is not None and len(sent) > fail_from:
            raise PlatformError(f"{url}: HTTP 500")
        query = parse_qs(urlsplit(url).query)
        start = dt.datetime.fromisoformat(
            query["start_time"][0].replace("Z", "+00:00")
        )
        page = [
            entry
            for entry in all_messages
            if start <= dt.datetime.fromisoformat(
                entry["created_at"].replace("Z", "+00:00")
            )
            < start + dt.timedelta(seconds=5)
        ]
        if always is not None:
            page = list(always)
        return json.dumps({"data": page}).encode("utf-8")

    return _get, sent


# --- paging -------------------------------------------------------------------


def test_the_fetcher_pages_through_the_window():
    """The API gives five seconds per call, so the fetcher walks the window."""
    messages = [message(index, BASE + dt.timedelta(seconds=index * 2)) for index in range(6)]
    http_get, calls = fake_api(messages)
    out = fetch_kick_chat("123", BASE, BASE + dt.timedelta(seconds=12), http_get, pause=0)
    assert len(out) == 6
    assert len(calls) > 1
    # Every call asks for a later moment than the one before.
    starts = [
        parse_qs(urlsplit(url).query)["start_time"][0] for url in calls
    ]
    assert starts == sorted(starts)
    assert starts[0].startswith("2026-11-14T01:00:00")


def test_the_fetcher_calls_the_messages_endpoint():
    http_get, calls = fake_api([])
    fetch_kick_chat("123", BASE, BASE + dt.timedelta(seconds=1), http_get, pause=0)
    assert calls[0].startswith(
        "https://kick.com/api/v2/channels/123/messages?start_time="
    )


def test_the_fetch_stops_at_the_end_of_the_window():
    messages = [message(0, BASE + dt.timedelta(seconds=30))]
    http_get, calls = fake_api(messages)
    out = fetch_kick_chat("123", BASE, BASE + dt.timedelta(seconds=10), http_get, pause=0)
    assert out == []
    assert len(calls) < 10


# --- deduplication --------------------------------------------------------------


def test_a_message_seen_twice_is_kept_once():
    """The boundary between two calls can repeat the last message."""
    repeated = message(0, BASE + dt.timedelta(seconds=4))
    http_get, _calls = fake_api([], always=[repeated])
    out = fetch_kick_chat("123", BASE, BASE + dt.timedelta(seconds=20), http_get, pause=0)
    assert len(out) == 1
    assert out[0]["text"] == "W"


def test_a_message_inside_the_window_is_kept_once():
    messages = [message(index, BASE + dt.timedelta(seconds=index)) for index in range(3)]
    http_get, _calls = fake_api(messages)
    out = fetch_kick_chat("123", BASE, BASE + dt.timedelta(seconds=3), http_get, pause=0)
    texts = [entry["text"] for entry in out]
    assert texts == ["W", "W", "W"]
    assert len(out) == 3


# --- the time offset --------------------------------------------------------------


def test_the_time_is_the_seconds_from_the_start():
    messages = [
        message(0, BASE, text="first"),
        message(1, BASE + dt.timedelta(seconds=7, milliseconds=500), text="second"),
    ]
    http_get, _calls = fake_api(messages)
    out = fetch_kick_chat("123", BASE, BASE + dt.timedelta(seconds=10), http_get, pause=0)
    assert out[0]["t"] == pytest.approx(0.0)
    assert out[1]["t"] == pytest.approx(7.5)


def test_the_user_and_the_text_are_kept():
    messages = [message(0, BASE, text="no way", user="someone")]
    http_get, _calls = fake_api(messages)
    out = fetch_kick_chat("123", BASE, BASE + dt.timedelta(seconds=5), http_get, pause=0)
    assert out[0]["user"] == "someone"
    assert out[0]["text"] == "no way"


# --- the HTTP error -----------------------------------------------------------------


def test_an_http_error_stops_the_fetch_and_keeps_what_was_fetched():
    messages = [message(index, BASE + dt.timedelta(seconds=index)) for index in range(3)]
    http_get, _calls = fake_api(messages, fail_from=1)
    out = fetch_kick_chat("123", BASE, BASE + dt.timedelta(seconds=30), http_get, pause=0)
    # The first page is kept.
    assert len(out) >= 1
    assert out.complete is False
    assert "chat fetch stopped" in out.notes


def test_an_http_error_is_retried_once():
    """One failure is not the end of the fetch."""
    calls: list = []

    def flaky(url: str, headers: dict | None = None) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            raise PlatformError(f"{url}: HTTP 500")
        return json.dumps({"data": []}).encode("utf-8")

    out = fetch_kick_chat("123", BASE, BASE + dt.timedelta(seconds=6), flaky, pause=0)
    assert out.complete is True
    assert len(calls) >= 2


# --- the allow-list -------------------------------------------------------------------


def test_a_streamer_who_is_not_allowed_is_refused_before_any_call():
    calls: list = []

    def explode(url, headers=None):  # pragma: no cover - must not run
        calls.append(url)
        raise AssertionError("the HTTP call must not happen for a refused streamer")

    with pytest.raises(ChatRefused, match="not allowed|status is"):
        fetch_chat(
            make_streamer(status="pending"),
            "123",
            BASE,
            BASE + dt.timedelta(seconds=10),
            explode,
        )
    assert calls == []


def test_a_streamer_without_proof_is_refused():
    with pytest.raises(ChatRefused):
        fetch_chat(
            make_streamer(proof=False),
            "123",
            BASE,
            BASE + dt.timedelta(seconds=10),
            lambda url, headers=None: b'{"data": []}',
        )


def test_an_allowed_streamer_gets_its_chat():
    http_get, _calls = fake_api([message(0, BASE)])
    out = fetch_chat(
        make_streamer(),
        "123",
        BASE,
        BASE + dt.timedelta(seconds=5),
        http_get,
        pause=0,
    )
    assert len(out) == 1
    assert out.complete is True


# --- the platforms without a chat replay -----------------------------------------------


def test_twitch_has_no_chat_replay_yet():
    out = fetch_chat(
        make_streamer("someone", platform="twitch"),
        "123",
        BASE,
        BASE + dt.timedelta(seconds=10),
        lambda url, headers=None: b'{"data": []}',
    )
    assert out == []
    assert "not supported yet" in out.notes
    assert "twitch" in out.notes


def test_youtube_has_no_chat_replay_yet():
    out = fetch_chat(
        make_streamer("someone", platform="youtube"),
        "123",
        BASE,
        BASE + dt.timedelta(seconds=10),
        lambda url, headers=None: b'{"data": []}',
    )
    assert out == []
    assert "not supported yet" in out.notes


# --- the run folder ----------------------------------------------------------------------


def test_the_messages_are_written_in_the_format_listen_reads(tmp_path):
    from clipper.listen import load_chat

    path = write_chat(
        [{"t": 1.5, "user": "a", "text": "no way"}, {"t": 2.0, "user": "b", "text": "LUL"}],
        tmp_path / "chat.jsonl",
    )
    messages = load_chat(path)
    assert messages == [
        {"t": 1.5, "user": "a", "text": "no way"},
        {"t": 2.0, "user": "b", "text": "LUL"},
    ]


def test_the_written_file_is_one_json_object_per_line(tmp_path):
    path = write_chat([{"t": 0.0, "user": "a", "text": "hi"}], tmp_path / "chat.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"t": 0.0, "user": "a", "text": "hi"}


# --- the live check carries the channel id -------------------------------------------------


def test_the_kick_live_check_gives_the_numeric_channel_id():
    from clipper.platforms import kick

    payload = {"id": 4242, "livestream": {"is_live": True, "session_title": "x"}}
    status = kick.is_live(
        "examplestreamer", lambda url, headers=None: json.dumps(payload).encode("utf-8")
    )
    assert status.channel_id == "4242"


def test_the_recording_plan_keeps_the_channel_id(tmp_path):
    from clipper import record

    plan = record.build_plan(make_streamer(), tmp_path / "out", 600, channel_id="4242")
    assert plan.channel_id == "4242"
