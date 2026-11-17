"""The live watcher.

Polls the allow-listed streamers every N seconds and tells us when one goes
live. Only ALLOWED streamers are polled — the allow-list is the hard rule,
and a streamer we may not clip is a streamer we do not watch either.

Notifications carry a title and a timestamp. No media ever leaves the host.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from . import allowlist
from .allowlist import Streamer
from .config import REPO_ROOT
from .platforms import ADAPTERS, HttpGet, LiveStatus, MissingCredentialError, PlatformError
from .platforms.base import USER_AGENT

#: Where events are appended when nothing else is configured.
EVENTS_PATH = REPO_ROOT / "data" / "events.jsonl"
#: The default gap between polls, seconds.
DEFAULT_INTERVAL = 120.0


@dataclass(frozen=True)
class Event:
    """One state change worth telling somebody about."""

    kind: str  # "went_live" | "went_offline"
    streamer_id: str
    streamer_name: str
    platform: str
    channel: str
    title: str = ""
    started_at: str = ""
    stream_url: str = ""
    at: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class Notifier(Protocol):
    """Something that can announce an event."""

    def notify(self, event: Event) -> None:  # pragma: no cover - protocol
        ...


class PrintNotifier:
    """Print the event and append it to `data/events.jsonl`."""

    def __init__(self, events_path: str | Path = EVENTS_PATH) -> None:
        self.events_path = Path(events_path)

    def notify(self, event: Event) -> None:
        line = (
            f"[{event.at}] {event.kind}: {event.streamer_name} "
            f"({event.platform}/{event.channel}) {event.title}".rstrip()
        )
        print(line)
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"could not write {self.events_path}: {exc}", file=sys.stderr)


class NtfyNotifier:
    """Post the event to an ntfy server. Off unless `NTFY_URL` is set."""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self.url = url
        self.timeout = timeout

    def notify(self, event: Event) -> None:
        title = "StreamClipper: went live" if event.kind == "went_live" else "StreamClipper: went offline"
        payload = json.dumps(
            {
                "topic": title,
                "message": f"{event.streamer_name}: {event.title}".rstrip(": "),
                "tags": ["loud_sound"] if event.kind == "went_live" else [],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
        except (urllib.error.URLError, OSError) as exc:
            print(f"ntfy notification failed: {exc}", file=sys.stderr)


class CompositeNotifier:
    """The print+file notifier, plus ntfy when it is configured."""

    def __init__(self, notifiers: Iterable[Notifier]) -> None:
        self.notifiers = list(notifiers)

    def notify(self, event: Event) -> None:
        for notifier in self.notifiers:
            notifier.notify(event)


def build_notifier(events_path: str | Path = EVENTS_PATH, environ: dict | None = None) -> Notifier:
    """The default notifier: print + append, plus ntfy when `NTFY_URL` is set."""
    env = os.environ if environ is None else environ
    notifiers: list[Notifier] = [PrintNotifier(events_path)]
    ntfy_url = env.get("NTFY_URL")
    if ntfy_url:
        notifiers.append(NtfyNotifier(ntfy_url))
    return CompositeNotifier(notifiers)


class Watcher:
    """Polls the allow-listed streamers and reports the transitions."""

    def __init__(
        self,
        streamers: Iterable[Streamer],
        *,
        http_get: HttpGet,
        notifier: Notifier | None = None,
        interval: float = DEFAULT_INTERVAL,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], dt.datetime] | None = None,
    ) -> None:
        # The allow-list gate happens here, once: nothing else polls.
        self.streamers = [s for s in streamers if allowlist.allowed(s)]
        self.http_get = http_get
        self.notifier = notifier or build_notifier()
        self.interval = interval
        self.sleep = sleep
        self.now = now or (lambda: dt.datetime.now(dt.timezone.utc))
        self.state: dict[str, LiveStatus] = {}
        self.errors: list[str] = []

    def poll_once(self) -> list[Event]:
        """Poll every allowed streamer once. Returns the events emitted."""
        events: list[Event] = []
        self.errors = []
        for streamer in self.streamers:
            adapter = ADAPTERS.get(streamer.platform)
            if adapter is None:
                self.errors.append(f"{streamer.id}: unknown platform {streamer.platform!r}")
                continue
            try:
                status = adapter(streamer.channel, self.http_get)
            except MissingCredentialError as exc:
                # One platform without credentials must not stop the others.
                self.errors.append(f"{streamer.id}: {exc}")
                continue
            except PlatformError as exc:
                self.errors.append(f"{streamer.id}: {exc}")
                continue
            previous = self.state.get(streamer.id) or LiveStatus(live=False)
            self.state[streamer.id] = status
            if status.live == previous.live:
                continue
            event = Event(
                kind="went_live" if status.live else "went_offline",
                streamer_id=streamer.id,
                streamer_name=streamer.name,
                platform=streamer.platform,
                channel=streamer.channel,
                title=status.title,
                started_at=status.started_at,
                stream_url=status.stream_url,
                at=self.now().isoformat(timespec="seconds"),
            )
            self.notifier.notify(event)
            events.append(event)
        return events

    def run(self, once: bool = False) -> None:
        """Poll, then sleep, until interrupted (or once)."""
        while True:
            self.poll_once()
            for message in self.errors:
                print(f"watcher error: {message}", file=sys.stderr)
            if once:
                return
            self.sleep(self.interval)


def add_watch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--once", action="store_true", help="poll once and exit (default: loop forever)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"seconds between polls (default {DEFAULT_INTERVAL:.0f})",
    )
    parser.add_argument(
        "--events",
        default=str(EVENTS_PATH),
        help="path of the events jsonl file (default data/events.jsonl)",
    )
    parser.add_argument(
        "--file", default=None, help="path to streamers.yaml (default: config/streamers.yaml)"
    )


def run_watch(args: argparse.Namespace) -> int:
    from .platforms import urllib_get

    try:
        streamers = allowlist.load(args.file)
    except allowlist.AllowlistError as exc:
        print(f"allow-list error: {exc}", file=sys.stderr)
        return 2
    allowed = allowlist.only_allowed(streamers)
    if not allowed:
        print(
            "no allowed streamers to watch — every entry on the allow-list is "
            "pending, unverified or rejected.",
            file=sys.stderr,
        )
    watcher = Watcher(
        streamers,
        http_get=urllib_get,
        notifier=build_notifier(args.events),
        interval=args.interval,
    )
    watcher.run(once=args.once)
    return 0
