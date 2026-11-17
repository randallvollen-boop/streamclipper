"""The chat replay fetcher.

A stream's chat is a signal of its own: the viewers react a few seconds
after the moment, and what they say is half the reason the clip is worth
cutting. Kick keeps the chat of a past stream, so a recording can be read
against what the viewers said while it was on.

The HTTP call is injected (`http_get`), so the tests never touch Kick —
the same rule the platform adapters follow.

    from clipper.chat import fetch_chat
    messages = fetch_chat(streamer, channel_id, start, end, http_get)

`fetch_chat` is the gated entry point: a streamer the allow-list refuses is
refused there, before a single request is made. `fetch_kick_chat` is the raw
fetcher underneath it.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .allowlist import Streamer
from .allowlist import allowed as streamer_allowed
from .allowlist import reason as streamer_reason
from .platforms.base import USER_AGENT, HttpGet

#: The Kick chat replay endpoint.
KICK_MESSAGES_URL = "https://kick.com/api/v2/channels/{channel_id}/messages"
#: How much chat one call is expected to return, seconds.
PAGE_SECONDS = 5.0
#: The smallest step we believe the API moved forward by, seconds.
MIN_STEP_SECONDS = 0.25
#: How many calls in a row may move almost not at all before we jump ahead.
STALL_CALLS = 10


class ChatError(Exception):
    """The chat could not be fetched."""


class ChatRefused(ChatError):
    """The chat may not be fetched: the streamer is not allowed."""


class ChatMessages(list):
    """The messages of a window, and how the fetch went.

    It is a plain list of `{t, user, text}` — the format
    `listen.load_chat` reads — with two extra attributes: `complete`
    (False when the fetch was cut short) and `notes` (why).
    """

    def __init__(self, messages: Iterable[dict[str, Any]] = ()) -> None:
        super().__init__(messages)
        self.complete = True
        self.notes = ""


# --- the small helpers --------------------------------------------------------


def as_utc(value: dt.datetime) -> dt.datetime:
    """A datetime in UTC. A naive one is taken to be UTC already."""
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def iso_z(value: dt.datetime) -> str:
    """`2026-11-14T01:00:00.000Z` — the form the API wants."""
    return as_utc(value).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_time(value: Any) -> dt.datetime | None:
    """An API timestamp as a UTC datetime, or None when it is not one."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return as_utc(parsed)


def parse_page(raw: bytes | str) -> list[dict[str, Any]]:
    """One page of the messages endpoint: `[{id, at, user, text}]`.

    A message without a timestamp is dropped: we cannot place it in the
    window without one.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ChatError(f"the chat response is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ChatError("the chat response is not a JSON object")
    entries = payload.get("data")
    if not isinstance(entries, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        at = parse_time(entry.get("created_at"))
        if at is None:
            continue
        sender = entry.get("sender")
        user = ""
        if isinstance(sender, dict):
            user = str(sender.get("username") or sender.get("slug") or "")
        text = str(entry.get("content") or "")
        out.append(
            {
                "id": str(entry.get("id") or f"{at.isoformat()}|{text}"),
                "at": at,
                "user": user,
                "text": text,
            }
        )
    return out


# --- the fetcher --------------------------------------------------------------


def fetch_kick_chat(
    channel_id: str | int,
    start: dt.datetime,
    end: dt.datetime,
    http_get: HttpGet,
    pause: float = 0.25,
) -> ChatMessages:
    """The chat of a Kick channel over a wall-clock window.

    The API answers with about five seconds of chat per call, so
    `start_time` steps forward through the window until the end of it. The
    boundary between two calls can repeat a message, so the messages are
    deduplicated by id.

    `t` is the seconds from `start`, which is what the chat signal wants.

    On an HTTP error the call is retried once. If it fails again the fetch
    stops and the messages fetched so far are kept — with `complete` False,
    so the run can record that the chat is short.
    """
    start = as_utc(start)
    end = as_utc(end)
    messages = ChatMessages()
    if end <= start:
        return messages
    channel = str(channel_id).strip()
    if not channel:
        raise ChatError("no numeric channel id: the live check did not give one")

    url_base = KICK_MESSAGES_URL.format(channel_id=channel)
    headers = {"User-Agent": USER_AGENT}
    seen: set[str] = set()
    cursor = start
    stall = 0
    first = True

    while cursor < end:
        if not first and pause:
            time.sleep(pause)
        first = False
        url = f"{url_base}?start_time={iso_z(cursor)}"
        try:
            raw = http_get(url, headers)
        except Exception as exc:  # a real HTTP error, or a fake one
            if pause:
                time.sleep(pause)
            try:
                raw = http_get(url, headers)
            except Exception as exc2:
                messages.complete = False
                messages.notes = (
                    f"the chat fetch stopped at {iso_z(cursor)}: {exc2}"
                )
                break
        try:
            page = parse_page(raw)
        except ChatError as exc:
            messages.complete = False
            messages.notes = str(exc)
            break

        fresh = [entry for entry in page if entry["id"] not in seen]
        for entry in fresh:
            seen.add(entry["id"])
        for entry in fresh:
            if entry["at"] < start or entry["at"] > end:
                continue
            messages.append(
                {
                    "t": round((entry["at"] - start).total_seconds(), 3),
                    "user": entry["user"],
                    "text": entry["text"],
                }
            )

        # Step `start_time` forward. The API answers with the messages from
        # that moment on, so the last fresh message is where the next call
        # starts — a millisecond later, so the message itself is not asked
        # for twice.
        last = max((entry["at"] for entry in fresh), default=None)
        if last is None or last <= cursor:
            cursor = cursor + dt.timedelta(seconds=PAGE_SECONDS)
            stall = 0
            continue
        if (last - cursor).total_seconds() < MIN_STEP_SECONDS:
            stall += 1
            if stall >= STALL_CALLS:
                # The API is not moving forward. Do not crawl: jump a page.
                cursor = cursor + dt.timedelta(seconds=PAGE_SECONDS)
                stall = 0
                continue
        else:
            stall = 0
        cursor = last + dt.timedelta(milliseconds=1)

    messages.sort(key=lambda message: message["t"])
    return messages


def fetch_chat(
    streamer: Streamer,
    channel_id: str | int,
    start: dt.datetime,
    end: dt.datetime,
    http_get: HttpGet,
    pause: float = 0.25,
) -> ChatMessages:
    """The chat of a streamer over a window — if the allow-list allows it.

    This is the gated entry point. A streamer the allow-list refuses is
    refused here, before a single request is made.

    Twitch and YouTube have no chat replay yet: they get an empty list and
    a note, so the run carries on without chat.
    """
    if not streamer_allowed(streamer):
        raise ChatRefused(
            f"{streamer.id} ({streamer.name}) may not have its chat fetched: "
            f"{streamer_reason(streamer)}"
        )
    if streamer.platform != "kick":
        messages = ChatMessages()
        messages.notes = (
            f"chat replay not supported yet for {streamer.platform} — "
            "the run carries on without chat"
        )
        return messages
    return fetch_kick_chat(channel_id, start, end, http_get, pause=pause)


# --- the run folder ------------------------------------------------------------


def write_chat(messages: Sequence[dict[str, Any]], path: Path) -> Path:
    """Write the messages as the jsonl `listen.load_chat` reads."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(
                json.dumps(
                    {
                        "t": round(float(message.get("t", 0.0)), 3),
                        "user": str(message.get("user", "")),
                        "text": str(message.get("text", "")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return target


def copy_chat(source: Path, path: Path) -> Path:
    """Put a chat file that is already on disk into the run folder."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
    return target


#: The type of a chat fetcher the runner uses.
ChatFetcher = Callable[..., ChatMessages]
