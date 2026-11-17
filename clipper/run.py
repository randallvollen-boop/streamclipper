"""The command: `python -m clipper run <streamer>`.

One command that ties the pieces together: watch an allow-listed streamer,
record the stream, then fetch the chat, listen, build the kits and tell
somebody they are ready to review.

    data/runs/<streamer>-<UTC date-time>/
        recording/   the recorder's chunks   (live runs only)
        chat.jsonl   the chat of the recorded window
        stream.json  who, when, and how the chat fetch went
        moments.json, cache/, kits/          from listen and the kit builder

`python -m clipper run --from-file <video> --streamer <id> [--chat <jsonl>]`
runs the second half on a recording that is already on disk.

The allow-list is checked first, here, and again by every piece it calls.
Every outside effect (HTTP, the recorder, listen, the kit builder, the
notifier, the clock) is a parameter, so the tests never touch a platform.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import allowlist, chat, kit, listen, record
from .allowlist import Streamer
from .config import Config, load as load_config
from .platforms import ADAPTERS, HttpGet, LiveStatus, PlatformError
from .watch import DEFAULT_INTERVAL, Event, Notifier, build_notifier

EXIT_REFUSED = record.EXIT_REFUSED


class RunRefused(Exception):
    """The run may not happen. The message says why."""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def run_folder(config: Config, streamer: Streamer, when: dt.datetime) -> Path:
    stamp = when.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config.data_dir / "runs" / f"{streamer.id}-{stamp}"


def require_allowed(streamer: Streamer) -> None:
    if not allowlist.allowed(streamer):
        raise RunRefused(
            f"{streamer.id} ({streamer.name}) is not allowed to be run: "
            f"{allowlist.reason(streamer)}"
        )


def record_stream(plan: record.RecordPlan, max_gb: float) -> int:
    """The default recorder: the existing streamlink | ffmpeg pipeline."""
    if not record.streamlink_installed():
        print("streamlink is not installed (pipx install streamlink)", file=sys.stderr)
        return 4
    return record.run_pipeline(plan, max_gb)


def cap_plan(plan: record.RecordPlan, max_hours: float | None) -> record.RecordPlan:
    """Stop the recording after `max_hours` by limiting the ffmpeg output."""
    if not max_hours:
        return plan
    argv = list(plan.ffmpeg_argv)
    argv[-1:-1] = ["-t", str(int(max_hours * 3600))]  # an output option: before the file
    return dataclasses.replace(plan, ffmpeg_argv=tuple(argv))


def process_recording(
    source: Path,
    run_dir: Path,
    streamer: Streamer,
    config: Config,
    *,
    chat_path: Path | None = None,
    window: tuple[dt.datetime, dt.datetime] | None = None,
    channel_id: str = "",
    http_get: HttpGet | None = None,
    chat_fn: chat.ChatFetcher = chat.fetch_chat,
    analyse_fn: Callable[..., dict[str, Any]] = listen.analyse,
    kits_fn: Callable[..., list[dict[str, Any]]] = kit.build_kits,
    notifier: Notifier | None = None,
    now: Callable[[], dt.datetime] = _utcnow,
) -> list[dict[str, Any]]:
    """Steps 4 and 5: chat, listen, kits, one notification."""
    require_allowed(streamer)
    run_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"streamer_id": streamer.id, "source": str(source)}

    chat_file: Path | None = None
    if chat_path is not None:
        chat_file = chat.copy_chat(chat_path, run_dir / "chat.jsonl")
    elif window is not None and channel_id and http_get is not None:
        info["window"] = {"start": chat.iso_z(window[0]), "end": chat.iso_z(window[1])}
        try:
            messages = chat_fn(streamer, channel_id, window[0], window[1], http_get)
        except chat.ChatError as exc:
            info["chat_note"] = str(exc)
            print(f"chat unavailable, carrying on without it: {exc}", file=sys.stderr)
        else:
            info["chat_messages"] = len(messages)
            info["chat_complete"] = messages.complete
            if messages.notes:
                info["chat_note"] = messages.notes
            if messages:
                chat_file = chat.write_chat(messages, run_dir / "chat.jsonl")
    (run_dir / "stream.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    joined = listen.resolve_source(source, run_dir)
    analyse_fn(joined, run_dir, config, chat_path=chat_file)
    kits = kits_fn(run_dir, streamer.id, config, streamer)

    page = f"http://127.0.0.1:{config.review_port}/"
    (notifier or build_notifier()).notify(
        Event(
            kind="kits_ready",
            streamer_id=streamer.id,
            streamer_name=streamer.name,
            platform=streamer.platform,
            channel=streamer.channel,
            title=f"{len(kits)} kits ready to review: {page} ({run_dir})",
            at=now().isoformat(timespec="seconds"),
        )
    )
    return kits


def run_streamer(
    streamer: Streamer,
    config: Config,
    *,
    http_get: HttpGet,
    once: bool = False,
    interval: float = DEFAULT_INTERVAL,
    max_hours: float | None = None,
    record_fn: Callable[[record.RecordPlan, float], int] = record_stream,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], dt.datetime] = _utcnow,
    **process_kwargs: Any,
) -> list[Path]:
    """Poll, record, process; back to watching unless `once`. Returns the run folders."""
    require_allowed(streamer)  # nothing is polled for a disallowed streamer
    adapter = ADAPTERS.get(streamer.platform)
    if adapter is None:
        raise RunRefused(f"{streamer.id}: unknown platform {streamer.platform!r}")
    runs: list[Path] = []
    while True:
        status = LiveStatus(live=False)
        try:
            status = adapter(streamer.channel, http_get)
        except PlatformError as exc:
            print(f"{streamer.id}: {exc}", file=sys.stderr)
        if status.live:
            started = now()
            run_dir = run_folder(config, streamer, started)
            plan = record.build_plan(
                streamer,
                run_dir / "recording",
                config.segment_seconds,
                status.stream_url or None,
                status.channel_id,
            )
            try:
                rc = record_fn(cap_plan(plan, max_hours), config.recordings_max_gb)
                print(f"recording finished (rc={rc}); chunks in {plan.output_dir}")
                if not any(plan.output_dir.glob("*.ts")):
                    print(f"{streamer.id}: no chunks recorded (rc={rc}); back to watching",
                          file=sys.stderr)
                else:
                    runs.append(run_dir)
                    process_recording(
                        plan.output_dir,
                        run_dir,
                        streamer,
                        config,
                        window=(started, now()),
                        channel_id=status.channel_id,
                        http_get=http_get,
                        now=now,
                        **process_kwargs,
                    )
            except (RunRefused, record.RecordRefused):
                raise
            except Exception as exc:  # one bad recording must not end the watch
                print(f"{streamer.id}: recording or processing failed, back to watching: {exc}",
                      file=sys.stderr)
        if once:
            return runs
        sleep(interval)


# --- CLI ----------------------------------------------------------------------


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("id", nargs="?", help="the allow-list id of the streamer to run")
    parser.add_argument("--once", action="store_true", help="check once and exit")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"seconds between polls (default {DEFAULT_INTERVAL:.0f})")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop recording after this many hours")
    parser.add_argument("--from-file", default=None,
                        help="skip watching and recording: process this recording")
    parser.add_argument("--streamer", default=None, help="the streamer id, with --from-file")
    parser.add_argument("--chat", default=None, help="a chat jsonl file, with --from-file")
    parser.add_argument("--file", default=None,
                        help="path to streamers.yaml (default: config/streamers.yaml)")


def run_run(args: argparse.Namespace) -> int:
    from .platforms import urllib_get

    streamer_id = args.streamer if args.from_file else args.id
    if not streamer_id:
        print("give a streamer id (or --from-file with --streamer)", file=sys.stderr)
        return 2
    config = load_config()
    try:
        streamers = allowlist.load(args.file)
    except allowlist.AllowlistError as exc:
        print(f"allow-list error: {exc}", file=sys.stderr)
        return 2
    streamer = next((s for s in streamers if s.id == streamer_id), None)
    if streamer is None:
        print(f"no streamer with id {streamer_id!r} on the allow-list", file=sys.stderr)
        return EXIT_REFUSED
    try:
        if args.from_file:
            source = Path(args.from_file).expanduser()
            kits = process_recording(
                source,
                run_folder(config, streamer, _utcnow()),
                streamer,
                config,
                chat_path=Path(args.chat).expanduser() if args.chat else None,
            )
            print(f"{len(kits)} kit(s) built")
        else:
            run_streamer(
                streamer,
                config,
                http_get=urllib_get,
                once=args.once,
                interval=args.interval,
                max_hours=args.max_hours,
            )
    except (RunRefused, record.RecordRefused) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (kit.KitError, FileNotFoundError) as exc:
        print(f"run error: {exc}", file=sys.stderr)
        return 2
    return 0
