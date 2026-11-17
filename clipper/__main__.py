"""The CLI: `python -m clipper <command>`.

Commands:
  streamers  the allow-list and why each entry is or is not allowed
  watch     poll the allow-listed streamers for "went live"
  record    record an allow-listed stream (streamlink)
  listen    rank the moments in a recording
  kits      build the remix kits for a finished run
  run       watch, record, listen and build kits for one streamer
  review    serve the review page on 127.0.0.1
"""
from __future__ import annotations

import argparse
import sys

from . import allowlist


def _cmd_streamers(args: argparse.Namespace) -> int:
    try:
        streamers = allowlist.load(args.file)
    except allowlist.AllowlistError as exc:
        print(f"allow-list error: {exc}", file=sys.stderr)
        return 2

    columns = [
        ("ID", 12),
        ("NAME", 12),
        ("PLATFORM", 9),
        ("CHANNEL", 16),
        ("STATUS", 10),
        ("ALLOWED", 7),
    ]
    header = "  ".join(name.ljust(width) for name, width in columns)
    print(header)
    print("  ".join("-" * width for _, width in columns))
    for streamer in streamers:
        is_allowed = allowlist.allowed(streamer)
        cells = [
            streamer.id.ljust(columns[0][1]),
            streamer.name.ljust(columns[1][1]),
            streamer.platform.ljust(columns[2][1]),
            streamer.channel.ljust(columns[3][1]),
            streamer.status.ljust(columns[4][1]),
            ("yes" if is_allowed else "no").ljust(columns[5][1]),
        ]
        print("  ".join(cells).rstrip())
        print("  " * 6 + f"why: {allowlist.reason(streamer)}")
        if streamer.conditions.notes:
            print("  " * 6 + f"notes: {streamer.conditions.notes}")
    allowed_count = len(allowlist.only_allowed(streamers))
    print()
    print(f"{allowed_count} of {len(streamers)} entries may be watched, recorded or clipped.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipper", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    streamers = sub.add_parser("streamers", help="show the allow-list")
    streamers.add_argument(
        "--file", default=None, help="path to streamers.yaml (default: config/streamers.yaml)"
    )
    streamers.set_defaults(func=_cmd_streamers)

    # The other commands live in their own modules. They are registered one
    # at a time so a slice that has not landed yet does not break the CLI.
    _add_watch(sub)
    _add_record(sub)
    _add_listen(sub)
    _add_kits(sub)
    _add_run(sub)
    _add_review(sub)

    return parser


def _add_watch(sub: argparse._SubParsersAction) -> None:
    try:
        from .watch import add_watch_arguments, run_watch
    except ImportError:
        return
    parser = sub.add_parser("watch", help="poll allow-listed streamers for live state")
    add_watch_arguments(parser)
    parser.set_defaults(func=lambda args: run_watch(args))


def _add_record(sub: argparse._SubParsersAction) -> None:
    try:
        from .record import add_record_arguments, run_record
    except ImportError:
        return
    parser = sub.add_parser("record", help="record an allow-listed stream")
    add_record_arguments(parser)
    parser.set_defaults(func=lambda args: run_record(args))


def _add_listen(sub: argparse._SubParsersAction) -> None:
    try:
        from .listen import add_listen_arguments, run_listen
    except ImportError:
        return
    parser = sub.add_parser("listen", help="rank the moments in a video")
    add_listen_arguments(parser)
    parser.set_defaults(func=lambda args: run_listen(args))


def _add_kits(sub: argparse._SubParsersAction) -> None:
    try:
        from .kit import add_kit_arguments, run_kits
    except ImportError:
        return
    parser = sub.add_parser("kits", help="build remix kits for a run")
    add_kit_arguments(parser)
    parser.set_defaults(func=lambda args: run_kits(args))


def _add_run(sub: argparse._SubParsersAction) -> None:
    try:
        from .run import add_run_arguments, run_run
    except ImportError:
        return
    parser = sub.add_parser("run", help="watch, record, listen and build kits for one streamer")
    add_run_arguments(parser)
    parser.set_defaults(func=lambda args: run_run(args))


def _add_review(sub: argparse._SubParsersAction) -> None:
    try:
        from .review.server import add_review_arguments, run_review
    except ImportError:
        return
    parser = sub.add_parser("review", help="serve the review page on 127.0.0.1")
    add_review_arguments(parser)
    parser.set_defaults(func=lambda args: run_review(args))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
