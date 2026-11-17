"""The allow-list: the hard rule of this project.

A streamer may be watched, recorded or clipped ONLY when `allowed()` says so.
`allowed()` is deliberately narrow:

    status == "verified"  AND  >= 1 proof entry with a url AND a checked date

Being on the list is not enough. A `pending` streamer (program not joined
yet) and a `verified` streamer whose proof was never saved are both refused,
and `reason()` says why in words so the operator can see what to fix.

Schema errors are loud: `load()` raises `AllowlistError` rather than silently
dropping a streamer, because a silently-dropped streamer is a streamer we
might clip by accident elsewhere.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

#: The platforms we can watch.
PLATFORMS = ("kick", "twitch", "youtube")
#: The lifecycle of an allow-list entry.
STATUSES = ("verified", "pending", "unverified", "rejected")
#: What a proof entry is allowed to be.
PROOF_KINDS = ("program-brief", "own-statement", "campaign")

#: The repository's own config directory.
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
#: The default allow-list file. Copy `config/streamers.example.yaml` to this
#: name and edit it; the environment variable STREAMCLIPPER_ALLOWLIST points
#: somewhere else.
STREAMERS_PATH = Path(os.environ.get("STREAMCLIPPER_ALLOWLIST") or CONFIG_DIR / "streamers.yaml")


class AllowlistError(Exception):
    """Raised when the allow-list file is malformed. Loud on purpose."""


@dataclass(frozen=True)
class Proof:
    """Evidence that the streamer allows clipping."""

    url: str
    kind: str
    checked: str

    @property
    def checked_date(self) -> dt.date | None:
        """The checked date as a date, or None when it is not a real date."""
        try:
            return dt.date.fromisoformat(self.checked)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class Conditions:
    """What the streamer requires of a clipper."""

    credit_format: str | None = None
    monetisation: str | None = None
    banned: tuple[str, ...] = ()
    notes: str = ""

    @property
    def credit_line(self) -> str:
        """The credit line to burn in / write in the caption."""
        return self.credit_format or ""


@dataclass(frozen=True)
class Streamer:
    """One allow-list entry."""

    id: str
    name: str
    platform: str
    channel: str
    status: str
    proof: tuple[Proof, ...] = ()
    conditions: Conditions = field(default_factory=Conditions)

    @property
    def proof_with_source(self) -> tuple[Proof, ...]:
        """Proof entries that carry both a url and a checked date."""
        return tuple(p for p in self.proof if p.url.strip() and p.checked_date)


def _require(mapping: Any, key: str, where: str) -> Any:
    if not isinstance(mapping, dict):
        raise AllowlistError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    if key not in mapping:
        raise AllowlistError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _as_str(value: Any, key: str, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise AllowlistError(f"{where}: {key!r} must be a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise AllowlistError(f"{where}: {key!r} must not be empty")
    return value.strip()


def _as_date_string(value: Any, where: str) -> str:
    """A date, as an ISO string.

    YAML parses an unquoted `2026-11-16` into a `datetime.date`, which is the
    natural way to write it, so that is accepted and normalised here.
    """
    if isinstance(value, dt.datetime):
        value = value.date()
    if isinstance(value, dt.date):
        return value.isoformat()
    if not isinstance(value, str):
        raise AllowlistError(
            f"{where}: checked must be a date (YYYY-MM-DD), got {type(value).__name__}"
        )
    text = value.strip()
    try:
        dt.date.fromisoformat(text)
    except ValueError:
        raise AllowlistError(
            f"{where}: checked {text!r} is not an ISO date (YYYY-MM-DD)"
        ) from None
    return text


def _parse_proof(raw: Any, where: str) -> Proof:
    if not isinstance(raw, list):
        raise AllowlistError(f"{where}.proof: expected a list, got {type(raw).__name__}")
    proofs: list[Proof] = []
    for index, entry in enumerate(raw):
        at = f"{where}.proof[{index}]"
        url = _as_str(_require(entry, "url", at), "url", at)
        kind = _as_str(_require(entry, "kind", at), "kind", at)
        if kind not in PROOF_KINDS:
            raise AllowlistError(
                f"{at}: kind {kind!r} is not one of {', '.join(PROOF_KINDS)}"
            )
        checked = _as_date_string(_require(entry, "checked", at), at)
        proofs.append(Proof(url=url, kind=kind, checked=checked))
    return tuple(proofs)  # type: ignore[return-value]


def _parse_conditions(raw: Any, where: str) -> Conditions:
    if raw is None:
        return Conditions()
    if not isinstance(raw, dict):
        raise AllowlistError(
            f"{where}.conditions: expected a mapping, got {type(raw).__name__}"
        )
    credit = raw.get("credit_format")
    if credit is not None and not isinstance(credit, str):
        raise AllowlistError(f"{where}.conditions.credit_format: must be a string or null")
    monetisation = raw.get("monetisation")
    if monetisation is not None and not isinstance(monetisation, str):
        raise AllowlistError(f"{where}.conditions.monetisation: must be a string or null")
    banned_raw = raw.get("banned") or []
    if not isinstance(banned_raw, list) or not all(isinstance(b, str) for b in banned_raw):
        raise AllowlistError(f"{where}.conditions.banned: must be a list of strings")
    notes = raw.get("notes") or ""
    if not isinstance(notes, str):
        raise AllowlistError(f"{where}.conditions.notes: must be a string")
    return Conditions(
        credit_format=credit,
        monetisation=monetisation,
        banned=tuple(banned_raw),
        notes=notes.strip(),
    )


def _parse_streamer(raw: Any, where: str) -> Streamer:
    if not isinstance(raw, dict):
        raise AllowlistError(f"{where}: expected a mapping, got {type(raw).__name__}")
    streamer_id = _as_str(_require(raw, "id", where), "id", where)
    where = f"{where} (id={streamer_id})"
    name = _as_str(_require(raw, "name", where), "name", where)
    platform = _as_str(_require(raw, "platform", where), "platform", where)
    if platform not in PLATFORMS:
        raise AllowlistError(
            f"{where}: platform {platform!r} is not one of {', '.join(PLATFORMS)}"
        )
    channel = _as_str(_require(raw, "channel", where), "channel", where)
    status = _as_str(_require(raw, "status", where), "status", where)
    if status not in STATUSES:
        raise AllowlistError(
            f"{where}: status {status!r} is not one of {', '.join(STATUSES)}"
        )
    proof = _parse_proof(raw.get("proof") or [], where)
    conditions = _parse_conditions(raw.get("conditions"), where)
    return Streamer(
        id=streamer_id,
        name=name,
        platform=platform,
        channel=channel,
        status=status,
        proof=proof,
        conditions=conditions,
    )


def load(path: str | Path | None = None) -> list[Streamer]:
    """Load and validate the allow-list. Raises `AllowlistError` on any fault."""
    allowlist_path = Path(path) if path is not None else STREAMERS_PATH
    try:
        text = allowlist_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AllowlistError(f"cannot read allow-list {allowlist_path}: {exc}") from exc
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise AllowlistError(f"{allowlist_path} is not valid YAML: {exc}") from exc
    if doc is None:
        raise AllowlistError(f"{allowlist_path} is empty")
    if isinstance(doc, list):
        entries: Any = doc
    elif isinstance(doc, dict) and "streamers" in doc:
        entries = doc["streamers"]
        if not isinstance(entries, list):
            raise AllowlistError(f"{allowlist_path}: 'streamers' must be a list")
    else:
        raise AllowlistError(f"{allowlist_path}: expected a 'streamers' list")

    streamers = [
        _parse_streamer(entry, f"{allowlist_path}[{index}]")
        for index, entry in enumerate(entries)
    ]
    seen: set[str] = set()
    for streamer in streamers:
        if streamer.id in seen:
            raise AllowlistError(
                f"{allowlist_path}: duplicate streamer id {streamer.id!r}"
            )
        seen.add(streamer.id)
    return streamers


def allowed(streamer: Streamer) -> bool:
    """True only for a verified streamer with proof that names a source and a date."""
    return streamer.status == "verified" and bool(streamer.proof_with_source)


def reason(streamer: Streamer) -> str:
    """Why this streamer is or is not allowed, in words."""
    if streamer.status != "verified":
        return f"status is {streamer.status!r}, not 'verified'"
    if not streamer.proof_with_source:
        if not streamer.proof:
            return "no proof recorded (save the program brief or their own statement)"
        return "no proof with both a url and a checked date"
    return "verified with proof"


def only_allowed(streamers: Iterable[Streamer]) -> list[Streamer]:
    """Filter down to the streamers we may act on."""
    return [s for s in streamers if allowed(s)]
