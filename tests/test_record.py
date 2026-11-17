"""The recorder.

Rules under test:
  * a streamer who is not allowed is never recorded;
  * the command is the documented streamlink | ffmpeg pipeline;
  * the disk budget stops the recording;
  * a finished recording is trashed with `gio trash` and a log line — never `rm`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path

import pytest

from clipper.allowlist import Conditions, Proof, Streamer
from clipper import config, record
from clipper.record import (
    EXIT_REFUSED,
    RecordRefused,
    budget_exceeded,
    build_plan,
    channel_url,
    disk_usage_bytes,
    trash_recording,
)


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


def write_allowlist(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "streamers.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# --- the allow-list gate -----------------------------------------------------


def test_build_plan_refuses_a_pending_streamer(tmp_path):
    with pytest.raises(RecordRefused, match="not allowed to be recorded"):
        build_plan(make_streamer(status="pending"), tmp_path / "out", 600)


def test_build_plan_refuses_a_streamer_without_proof(tmp_path):
    with pytest.raises(RecordRefused, match="not allowed to be recorded"):
        build_plan(make_streamer(status="verified", proof=False), tmp_path / "out", 600)


def test_build_plan_makes_the_documented_pipeline(tmp_path):
    plan = build_plan(make_streamer("x", "kick"), tmp_path / "out", 600)
    assert plan.streamlink_argv == (
        "streamlink",
        "--hls-live-restart",
        "-o",
        "-",
        "https://kick.com/x",
        "best",
    )
    assert plan.ffmpeg_argv == (
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        "-",
        "-f",
        "segment",
        "-segment_time",
        "600",
        "-c",
        "copy",
        "-segment_format",
        "mpegts",
        str(tmp_path / "out" / "%03d.ts"),
    )
    assert " | " in plan.command
    assert plan.command.startswith("streamlink --hls-live-restart -o - ")


def test_the_url_can_be_overridden(tmp_path):
    plan = build_plan(make_streamer(), tmp_path / "out", 600, url="https://example.test/live.m3u8")
    assert plan.url == "https://example.test/live.m3u8"
    assert "https://example.test/live.m3u8" in plan.streamlink_argv


def test_channel_urls():
    assert channel_url(make_streamer("x", "kick")) == "https://kick.com/x"
    assert channel_url(make_streamer("y", "twitch")) == "https://www.twitch.tv/y"
    assert channel_url(make_streamer("z", "youtube")) == "https://www.youtube.com/@z"


# --- the CLI -----------------------------------------------------------------


def test_the_cli_refuses_a_pending_streamer(tmp_path, capsys):
    allowlist_path = write_allowlist(
        tmp_path,
        """
        streamers:
          - id: examplestreamer
            name: ExampleStreamer
            platform: kick
            channel: examplestreamer
            status: pending
            proof: []
        """,
    )
    args = argparse.Namespace(
        id="examplestreamer",
        dry_run=True,
        url=None,
        out=None,
        segment_seconds=None,
        max_gb=None,
        file=str(allowlist_path),
    )
    rc = record.run_record(args)
    assert rc == EXIT_REFUSED
    captured = capsys.readouterr()
    assert "refused" in captured.err
    assert "not allowed" in captured.err


def test_the_cli_refuses_an_unknown_streamer(tmp_path, capsys):
    allowlist_path = write_allowlist(
        tmp_path,
        """
        streamers:
          - id: examplestreamer
            name: ExampleStreamer
            platform: kick
            channel: examplestreamer
            status: pending
            proof: []
        """,
    )
    args = argparse.Namespace(
        id="nobody",
        dry_run=True,
        url=None,
        out=None,
        segment_seconds=None,
        max_gb=None,
        file=str(allowlist_path),
    )
    rc = record.run_record(args)
    assert rc == EXIT_REFUSED
    assert "no streamer with id" in capsys.readouterr().err


def test_the_cli_prints_the_command_for_an_allowed_streamer(tmp_path, capsys):
    allowlist_path = write_allowlist(
        tmp_path,
        """
        streamers:
          - id: x
            name: X
            platform: kick
            channel: x
            status: verified
            proof:
              - url: https://example.test/brief
                kind: program-brief
                checked: 2026-11-16
        """,
    )
    args = argparse.Namespace(
        id="x",
        dry_run=True,
        url=None,
        out=str(tmp_path / "out"),
        segment_seconds=600,
        max_gb=200,
        file=str(allowlist_path),
    )
    rc = record.run_record(args)
    assert rc == 0
    printed = capsys.readouterr().out.strip()
    assert printed.startswith("streamlink --hls-live-restart -o - https://kick.com/x best | ffmpeg")


# --- the disk budget ---------------------------------------------------------


def test_budget_exceeded():
    assert budget_exceeded(200 * (1024 ** 3), 200) is True
    assert budget_exceeded(199 * (1024 ** 3), 200) is False
    assert budget_exceeded(1, 0) is True


def test_disk_usage_counts_the_files(tmp_path):
    (tmp_path / "a.ts").write_bytes(b"x" * 100)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.ts").write_bytes(b"y" * 50)
    assert disk_usage_bytes(tmp_path) == 150
    assert disk_usage_bytes(tmp_path / "missing") == 0


def test_the_cli_refuses_when_the_budget_is_used_up(tmp_path, monkeypatch, capsys):
    allowlist_path = write_allowlist(
        tmp_path,
        """
        streamers:
          - id: x
            name: X
            platform: kick
            channel: x
            status: verified
            proof:
              - url: https://example.test/brief
                kind: program-brief
                checked: 2026-11-16
        """,
    )
    # A recordings root that is already over its budget.
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    (recordings_root / "big.ts").write_bytes(b"x" * 1000)
    cfg_path = tmp_path / "clipper.yaml"
    cfg_path.write_text(
        f"recordings:\n  recordings_max_gb: 0.0000001\n  root: {recordings_root}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(record, "load_config", lambda: config.load(cfg_path))
    args = argparse.Namespace(
        id="x",
        dry_run=True,
        url=None,
        out=str(tmp_path / "out"),
        segment_seconds=600,
        max_gb=None,
        file=str(allowlist_path),
    )
    rc = record.run_record(args)
    assert rc == EXIT_REFUSED
    assert "disk budget" in capsys.readouterr().err


# --- trashing ----------------------------------------------------------------


def test_trash_recording_uses_gio_and_writes_a_log_line(tmp_path):
    recording = tmp_path / "recording.ts"
    recording.write_bytes(b"data")
    log_path = tmp_path / "trash-log.txt"
    calls: list = []

    def fake_run(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    line = trash_recording(
        recording,
        reason="kits cut",
        trash_log=log_path,
        runner=fake_run,
        when=dt.datetime(2026, 11, 16, 21, 0, 0, tzinfo=dt.timezone.utc),
    )
    assert calls == [["gio", "trash", str(recording)]]
    assert log_path.read_text(encoding="utf-8").strip() == line
    assert "gio trash" not in line  # the log names the action, not the raw command
    assert "trashed" in line and str(recording) in line and "kits cut" in line


def test_trash_recording_never_calls_rm(tmp_path):
    recording = tmp_path / "recording.ts"
    recording.write_bytes(b"data")
    calls: list = []

    def fake_run(argv):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    trash_recording(recording, trash_log=tmp_path / "log.txt", runner=fake_run)
    assert all(argv[0] == "gio" for argv in calls)


def test_trash_recording_reports_a_gio_failure(tmp_path):
    recording = tmp_path / "recording.ts"
    recording.write_bytes(b"data")

    def fake_run(argv):
        return subprocess.CompletedProcess(argv, 1, "", "gio: no such file")

    with pytest.raises(RecordRefused, match="gio trash failed"):
        trash_recording(recording, trash_log=tmp_path / "log.txt", runner=fake_run)


def test_trash_recording_refuses_a_missing_path(tmp_path):
    with pytest.raises(RecordRefused, match="nothing to trash"):
        trash_recording(tmp_path / "nope.ts", trash_log=tmp_path / "log.txt")
