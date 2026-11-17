"""The live watcher.

The rules under test:
  * only ALLOWED streamers are ever polled;
  * a transition emits exactly one event (not one per poll);
  * a missing platform credential is a clear error and stops no one else.
"""
from __future__ import annotations

import datetime as dt
import json

from clipper.allowlist import Conditions, Proof, Streamer
from clipper.watch import Watcher, build_notifier


def make_streamer(streamer_id="x", platform="kick", status="verified", proof=True) -> Streamer:
    return Streamer(
        id=streamer_id,
        name=streamer_id.upper(),
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


class Recorder:
    """A notifier that keeps the events."""

    def __init__(self) -> None:
        self.events = []

    def notify(self, event):
        self.events.append(event)


def kick_get(live: bool, calls: list):
    payload = {"livestream": None}
    if live:
        payload = {
            "livestream": {
                "session_title": "Ranked grind",
                "start_time": "2026-11-16T20:00:00Z",
                "is_live": True,
            }
        }

    def _get(url, headers=None):
        calls.append(url)
        return json.dumps(payload).encode("utf-8")

    return _get


def fixed_now():
    return dt.datetime(2026, 11, 16, 21, 0, 0, tzinfo=dt.timezone.utc)


def make_watcher(streamers, http_get, notifier=None):
    return Watcher(
        streamers,
        http_get=http_get,
        notifier=notifier or Recorder(),
        now=fixed_now,
    )


# --- the allow-list gate ----------------------------------------------------


def test_non_allowed_streamers_are_never_polled():
    calls: list = []
    watcher = make_watcher(
        [
            make_streamer("allowed", status="verified"),
            make_streamer("pending", status="pending"),
            make_streamer("unverified", status="unverified"),
            make_streamer("rejected", status="rejected"),
            make_streamer("noproof", status="verified", proof=False),
        ],
        kick_get(False, calls),
    )
    watcher.poll_once()
    assert calls == ["https://kick.com/api/v2/channels/allowed"]


def test_the_watcher_keeps_no_state_for_refused_streamers():
    watcher = make_watcher([make_streamer("pending", status="pending")], kick_get(False, []))
    watcher.poll_once()
    assert watcher.state == {}


# --- transitions -------------------------------------------------------------


def test_a_transition_emits_exactly_one_event():
    calls: list = []
    notifier = Recorder()
    watcher = make_watcher([make_streamer("x")], kick_get(True, calls), notifier)

    events = watcher.poll_once()
    assert len(events) == 1
    assert events[0].kind == "went_live"
    assert events[0].streamer_id == "x"
    assert events[0].title == "Ranked grind"
    assert events[0].at == "2026-11-16T21:00:00+00:00"
    assert len(notifier.events) == 1


def test_polling_again_without_a_change_emits_nothing():
    calls: list = []
    notifier = Recorder()
    watcher = make_watcher([make_streamer("x")], kick_get(True, calls), notifier)

    watcher.poll_once()
    watcher.poll_once()
    watcher.poll_once()
    assert len(notifier.events) == 1


def test_going_offline_emits_one_more_event():
    calls: list = []
    notifier = Recorder()
    live = {"value": True}

    def _get(url, headers=None):
        calls.append(url)
        payload = {
            "livestream": {
                "session_title": "Ranked grind",
                "start_time": "2026-11-16T20:00:00Z",
                "is_live": True,
            }
        }
        if not live["value"]:
            payload = {"livestream": None}
        return json.dumps(payload).encode("utf-8")

    watcher = make_watcher([make_streamer("x")], _get, notifier)
    watcher.poll_once()
    live["value"] = False
    events = watcher.poll_once()
    assert len(events) == 1
    assert events[0].kind == "went_offline"
    assert len(notifier.events) == 2


def test_an_offline_streamer_emits_nothing():
    calls: list = []
    notifier = Recorder()
    watcher = make_watcher([make_streamer("x")], kick_get(False, calls), notifier)
    assert watcher.poll_once() == []
    assert watcher.poll_once() == []
    assert notifier.events == []


# --- errors ------------------------------------------------------------------


def test_a_missing_credential_is_reported_and_others_still_run(monkeypatch):
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_TOKEN", raising=False)
    notifier = Recorder()
    watcher = make_watcher(
        [make_streamer("tw", platform="twitch"), make_streamer("x", platform="kick")],
        kick_get(False, []),
        notifier,
    )
    watcher.poll_once()
    assert len(watcher.errors) == 1
    assert "TWITCH_CLIENT_ID" in watcher.errors[0]
    # The Kick streamer was still polled.
    assert watcher.state["x"].live is False


# --- the notifier ------------------------------------------------------------


def test_the_default_notifier_appends_to_the_events_file(tmp_path):
    events_path = tmp_path / "events.jsonl"
    notifier = build_notifier(events_path, environ={})
    notifier.notify(
        __import__("clipper.watch", fromlist=["Event"]).Event(
            kind="went_live",
            streamer_id="x",
            streamer_name="X",
            platform="kick",
            channel="x",
            title="Ranked grind",
            started_at="2026-11-16T20:00:00Z",
            stream_url="https://kick.com/x",
            at="2026-11-16T21:00:00+00:00",
        )
    )
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["kind"] == "went_live"
    assert payload["title"] == "Ranked grind"


def test_the_ntfy_notifier_is_off_unless_the_url_is_set(tmp_path):
    events_path = tmp_path / "events.jsonl"
    notifier = build_notifier(events_path, environ={})
    names = [type(n).__name__ for n in notifier.notifiers]
    assert names == ["PrintNotifier"]

    notifier = build_notifier(events_path, environ={"NTFY_URL": "https://ntfy.test/x"})
    names = [type(n).__name__ for n in notifier.notifiers]
    assert names == ["PrintNotifier", "NtfyNotifier"]
