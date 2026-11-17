"""`clipper run`.

The platform, the recorder, chat and listen are fakes; the recorder writes a
tiny ffmpeg test file, and the kits are cut for real from it.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from clipper import audio, run
from clipper.allowlist import Conditions, Proof, Streamer
from clipper.chat import ChatMessages
from clipper.config import Config

SECONDS = 12.0


def make_streamer(status="verified", proof=True) -> Streamer:
    return Streamer(
        id="x", name="X", platform="kick", channel="x", status=status,
        proof=((Proof(url="https://example.test/b", kind="program-brief", checked="2026-11-16"),) if proof else ()),
        conditions=Conditions(),
    )


def make_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.run([
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=10:duration={SECONDS}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={SECONDS}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ])
    return path


class World:
    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.config = Config(
            raw={"data_dir": str(tmp_path / "data"), "review_port": 8297,
                 "recordings": {"root": str(tmp_path / "rec")}},
            path=tmp_path / "c.yaml",
        )
        self.recorded: list = []
        self.events: list = []
        self.http_calls: list = []
        self.chat_calls: list = []
        self.analysed: list = []

    # fakes
    def http_get(self, url, headers=None):
        self.http_calls.append(url)
        return json.dumps({"id": 7, "livestream": {"is_live": True, "session_title": "t",
                                                     "start_time": "2026-11-17T00:00:00Z"}}).encode()

    def record_fn(self, plan, max_gb):
        self.recorded.append(plan)
        make_video(plan.output_dir / "000.ts")
        return 0

    def chat_fn(self, streamer, channel_id, start, end, http_get):
        self.chat_calls.append((channel_id, start, end))
        messages = ChatMessages([{"t": 3.0, "user": "a", "text": "wow"}])
        return messages

    def analyse_fn(self, source, out_dir, config, chat_path=None, top_n=None):
        self.analysed.append((source, chat_path))
        payload = {"source": str(source), "duration": SECONDS, "moments": [
            {"rank": 1, "mmss": "00:06", "time": 6.0, "score": 3.0, "signals": {}}]}
        (out_dir / "moments.json").write_text(json.dumps(payload))
        return payload

    def notify(self, event):
        self.events.append(event)

    @property
    def notifier(self):
        return self

    def kwargs(self):
        return dict(chat_fn=self.chat_fn, analyse_fn=self.analyse_fn, notifier=self)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def go(world, streamer, **extra):
    return run.run_streamer(
        streamer, world.config, http_get=world.http_get, once=True,
        record_fn=world.record_fn, sleep=lambda _s: None,
        now=lambda: dt.datetime(2026, 11, 17, 1, 2, 3, tzinfo=dt.timezone.utc),
        **world.kwargs(), **extra,
    )


def test_a_live_run_has_the_folder_layout_kits_and_one_notification(world):
    [run_dir] = go(world, make_streamer())
    assert run_dir == world.config.data_dir / "runs" / "x-20261117T010203Z"
    assert (run_dir / "recording" / "000.ts").exists()
    assert (run_dir / "chat.jsonl").exists()
    assert (run_dir / "stream.json").exists()
    assert (run_dir / "moments.json").exists()
    kits = list((run_dir / "kits").glob("*/kit.json"))
    assert len(kits) == 1
    assert (kits[0].parent / "vertical.mp4").exists()
    assert len(world.events) == 1
    assert world.events[0].title.startswith("1 kits ready to review")
    assert "8297" in world.events[0].title
    # chat was fetched for the channel id and handed to listen
    assert world.chat_calls[0][0] == "7"
    assert world.analysed[0][1] == run_dir / "chat.jsonl"


def test_a_disallowed_streamer_never_reaches_the_recorder_or_the_network(world):
    for streamer in (make_streamer(status="pending"), make_streamer(proof=False)):
        with pytest.raises(run.RunRefused):
            go(world, streamer)
    assert world.recorded == []
    assert world.http_calls == []
    assert world.chat_calls == []
    assert world.events == []


def test_not_live_records_nothing(world):
    world.http_get = lambda url, headers=None: json.dumps({"id": 7, "livestream": None}).encode()
    assert go(world, make_streamer()) == []
    assert world.recorded == []
    assert world.events == []


def test_max_hours_limits_the_ffmpeg_output(world):
    go(world, make_streamer(), max_hours=2)
    argv = world.recorded[0].ffmpeg_argv
    assert argv[-3:-1] == ("-t", "7200")
    assert argv[-1].endswith("%03d.ts")


def test_from_file_runs_the_second_half_with_a_chat_file(world, tmp_path):
    video = make_video(tmp_path / "in" / "v.mp4")
    chat_file = tmp_path / "in" / "chat.jsonl"
    chat_file.write_text('{"t": 1, "user": "a", "text": "hi"}\n')
    run_dir = tmp_path / "data" / "runs" / "x-manual"
    kits = run.process_recording(
        video, run_dir, make_streamer(), world.config, chat_path=chat_file, **world.kwargs()
    )
    assert len(kits) == 1
    assert (run_dir / "chat.jsonl").read_text() == chat_file.read_text()
    assert world.chat_calls == []
    assert len(world.events) == 1


def test_from_file_refuses_a_disallowed_streamer(world, tmp_path):
    with pytest.raises(run.RunRefused):
        run.process_recording(tmp_path / "v.mp4", tmp_path / "r", make_streamer(status="pending"),
                              world.config, **world.kwargs())
    assert world.analysed == []


def test_a_failed_recording_returns_to_watching_and_sleeps(world):
    sleeps = []

    def stop_after_two(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    def failing_record(plan, max_gb):
        world.recorded.append(plan)
        plan.output_dir.mkdir(parents=True, exist_ok=True)  # streamlink died: no chunks
        return 1

    with pytest.raises(KeyboardInterrupt):
        run.run_streamer(
            make_streamer(), world.config, http_get=world.http_get,
            record_fn=failing_record, sleep=stop_after_two, interval=7.0,
            now=lambda: dt.datetime(2026, 11, 17, 1, 2, 3, tzinfo=dt.timezone.utc),
            **world.kwargs(),
        )
    assert len(world.recorded) == 2  # it polled again; it did not die
    assert sleeps == [7.0, 7.0]
    assert world.events == []


def test_a_processing_error_does_not_kill_the_loop(world):
    def boom(*a, **k):
        raise audio.MediaError("kaboom")

    kwargs = world.kwargs()
    kwargs["analyse_fn"] = boom
    slept = []
    runs = run.run_streamer(
        make_streamer(), world.config, http_get=world.http_get, once=True,
        record_fn=world.record_fn, sleep=slept.append,
        now=lambda: dt.datetime(2026, 11, 17, 1, 2, 3, tzinfo=dt.timezone.utc),
        **kwargs,
    )
    assert len(runs) == 1 and world.events == []
