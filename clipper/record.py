"""The recorder.

Records an allow-listed stream in 10-minute chunks, keeps a disk budget, and
trashes the recording once the kits have been cut (only
the cut clips are kept).

The command we build is the one the reference system uses — streamlink
writes the live stream to stdout, ffmpeg cuts it into 10-minute MPEG-TS
segments:

    streamlink --hls-live-restart -o - <url> best \
      | ffmpeg -hide_banner -loglevel warning -i - \
          -f segment -segment_time 600 -c copy \
          data/recordings/<id>/<utc>/%03d.ts

`-o -` is streamlink's "write the stream to stdout" (streamlink 8.6.3;
`--output -`). The two processes are connected by a pipe, not by a shell, so
no path is ever handed to a shell.

streamlink is NOT part of this repo's virtualenv. Install it once with:

    pipx install streamlink
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import allowlist
from .allowlist import Streamer
from .config import Config, load as load_config

#: Where the trashed recordings are logged (never `rm`, AGENTS.md).
TRASH_LOG = Path.home() / "archive" / "trash-log.txt"
#: How often the recorder checks the disk budget, seconds.
BUDGET_CHECK_SECONDS = 15.0
#: The exit code used when a streamer may not be recorded.
EXIT_REFUSED = 3


class RecordRefused(Exception):
    """The recording may not happen. The message says why."""


@dataclass(frozen=True)
class RecordPlan:
    """Everything needed to start one recording."""

    streamer_id: str
    streamer_name: str
    platform: str
    channel: str
    url: str
    output_dir: Path
    segment_seconds: int
    streamlink_argv: tuple[str, ...]
    ffmpeg_argv: tuple[str, ...]
    #: The platform's own numeric id of the channel. The chat replay needs
    #: it, so it is kept on the plan.
    channel_id: str = ""

    @property
    def command(self) -> str:
        """The command as one line, for the operator to read."""
        return " ".join(self.streamlink_argv) + " | " + " ".join(self.ffmpeg_argv)


def channel_url(streamer: Streamer) -> str:
    """The page URL for a streamer's channel. streamlink resolves the stream."""
    channel = streamer.channel.strip().lstrip("@")
    if streamer.platform == "kick":
        return f"https://kick.com/{channel}"
    if streamer.platform == "twitch":
        return f"https://www.twitch.tv/{channel.lower()}"
    if streamer.platform == "youtube":
        return f"https://www.youtube.com/@{channel}"
    raise RecordRefused(f"{streamer.id}: unknown platform {streamer.platform!r}")


def build_plan(
    streamer: Streamer,
    output_dir: Path,
    segment_seconds: int,
    url: str | None = None,
    channel_id: str = "",
) -> RecordPlan:
    """Build the record command for an allowed streamer."""
    if not allowlist.allowed(streamer):
        raise RecordRefused(
            f"{streamer.id} ({streamer.name}) is not allowed to be recorded: "
            f"{allowlist.reason(streamer)}"
        )
    target = url or channel_url(streamer)
    output_dir.mkdir(parents=True, exist_ok=True)
    streamlink_argv = (
        "streamlink",
        "--hls-live-restart",
        "-o",
        "-",
        target,
        "best",
    )
    ffmpeg_argv = (
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        "-",
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-c",
        "copy",
        "-segment_format",
        "mpegts",
        str(output_dir / "%03d.ts"),
    )
    return RecordPlan(
        streamer_id=streamer.id,
        streamer_name=streamer.name,
        platform=streamer.platform,
        channel=streamer.channel,
        url=target,
        output_dir=output_dir,
        segment_seconds=segment_seconds,
        streamlink_argv=streamlink_argv,
        ffmpeg_argv=ffmpeg_argv,
        channel_id=str(channel_id or ""),
    )


# --- disk budget -------------------------------------------------------------


def disk_usage_bytes(path: Path) -> int:
    """The size of everything under `path` (0 when it does not exist)."""
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def budget_exceeded(usage_bytes: int, max_bytes: float) -> bool:
    """Has the recordings disk budget been used up?"""
    if max_bytes <= 0:
        return True
    return usage_bytes >= max_bytes * (1024 ** 3)


# --- trashing a recording ----------------------------------------------------


def trash_log_line(path: Path, reason: str, when: dt.datetime | None = None) -> str:
    """The line we append to the trash log."""
    stamp = (when or dt.datetime.now().astimezone()).isoformat(timespec="seconds")
    return f"{stamp} trashed {path} ({reason}) (StreamClipper)"


def trash_recording(
    path: Path,
    reason: str = "kits cut; only the clips are kept",
    trash_log: Path = TRASH_LOG,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess] | None = None,
    when: dt.datetime | None = None,
) -> str:
    """Move a recording to the trash with `gio trash` and log the line.

    Files are never removed with `rm` (AGENTS.md).
    """
    run = runner or (lambda argv: subprocess.run(argv, capture_output=True, text=True))
    if not path.exists():
        raise RecordRefused(f"nothing to trash: {path}")
    proc = run(["gio", "trash", str(path)])
    returncode = getattr(proc, "returncode", 0)
    if returncode != 0:
        stderr = (getattr(proc, "stderr", "") or "").strip()
        raise RecordRefused(f"gio trash failed (rc={returncode}): {stderr}")
    line = trash_log_line(path, reason, when)
    trash_log.parent.mkdir(parents=True, exist_ok=True)
    with trash_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return line


# --- running -----------------------------------------------------------------


def streamlink_installed() -> bool:
    return shutil.which("streamlink") is not None


class _BudgetMonitor(threading.Thread):
    """Stops the recording when the disk budget is used up."""

    def __init__(self, directory: Path, max_bytes: float, interval: float = BUDGET_CHECK_SECONDS):
        super().__init__(daemon=True)
        self.directory = directory
        self.max_bytes = max_bytes
        self.interval = interval
        self.stop_event = threading.Event()
        self.exceeded = False

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            if budget_exceeded(disk_usage_bytes(self.directory), self.max_bytes):
                self.exceeded = True
                return


def run_pipeline(plan: RecordPlan, max_gb: float) -> int:
    """Run the record pipeline until the stream ends or the budget runs out."""
    streamlink = subprocess.Popen(
        list(plan.streamlink_argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert streamlink.stdout is not None
    ffmpeg = subprocess.Popen(
        list(plan.ffmpeg_argv),
        stdin=streamlink.stdout,
        stderr=subprocess.PIPE,
    )
    streamlink.stdout.close()
    monitor = _BudgetMonitor(plan.output_dir, max_gb)
    monitor.start()
    ffmpeg_rc = ffmpeg.wait()
    monitor.stop_event.set()
    streamlink_rc = streamlink.wait()
    if monitor.exceeded:
        print(
            f"disk budget of {max_gb:.0f} GB reached — recording stopped",
            file=sys.stderr,
        )
    return int(ffmpeg_rc or streamlink_rc or 0)


# --- CLI ---------------------------------------------------------------------


def add_record_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("id", help="the allow-list id of the streamer to record")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command that would run and stop",
    )
    parser.add_argument(
        "--url", default=None, help="record this URL instead of the channel page"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="the output directory (default data/recordings/<id>/<utc>)",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=None,
        help="chunk length in seconds (default from config: 600)",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=None,
        help="the recordings disk budget in GB (default from config)",
    )
    parser.add_argument(
        "--file", default=None, help="path to streamers.yaml (default: config/streamers.yaml)"
    )


def run_record(args: argparse.Namespace) -> int:
    config = load_config()
    try:
        streamers = allowlist.load(args.file)
    except allowlist.AllowlistError as exc:
        print(f"allow-list error: {exc}", file=sys.stderr)
        return 2
    match = next((s for s in streamers if s.id == args.id), None)
    if match is None:
        print(
            f"no streamer with id {args.id!r} on the allow-list — "
            "an unknown streamer is never recorded",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    max_gb = args.max_gb if args.max_gb is not None else config.recordings_max_gb
    segment_seconds = (
        args.segment_seconds if args.segment_seconds is not None else config.segment_seconds
    )
    if args.out:
        output_dir = Path(args.out)
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = config.recordings_root / match.id / stamp

    try:
        plan = build_plan(match, output_dir, segment_seconds, args.url)
    except RecordRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    usage = disk_usage_bytes(config.recordings_root)
    if budget_exceeded(usage, max_gb):
        print(
            f"refused: the recordings disk budget of {max_gb:.0f} GB is already used "
            f"({usage / (1024 ** 3):.1f} GB in {config.recordings_root})",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    if args.dry_run:
        print(plan.command)
        return 0

    if not streamlink_installed():
        print(
            "streamlink is not installed. Install it once with:\n"
            "    pipx install streamlink",
            file=sys.stderr,
        )
        return 4

    print(f"recording {plan.streamer_name} into {plan.output_dir}")
    rc = run_pipeline(plan, max_gb)
    print(f"recording finished (rc={rc}); chunks in {plan.output_dir}")
    return rc
