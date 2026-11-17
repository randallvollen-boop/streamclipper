"""The kit library: what a kit is and what a decision does to it.

The review page (HTTP), the CLI and the MCP server all call the functions in
this module. None of them re-implements them.

The layout:

    data/runs/<run>/kits/<kit>/kit.json     what the kit is
    data/decisions.jsonl                    one line per decision
    data/handoff/<account>/<run>-<kit>/     what the person uploads

A decision is append-only. The last approve or reject for a kit decides its
status (new / approved / rejected); a trim or a crop is recorded but does not
change the status.

Nothing is ever deleted. A re-cut clip moves the old file to `prev/` inside
the kit, and an approved clip is copied into the hand-off folder.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import audio
from . import finish as finisher
from . import kit as kit_builder
from .config import REPO_ROOT, Config

#: The status of a kit.
STATUS_NEW = "new"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

#: The decision log.
DECISIONS_NAME = "decisions.jsonl"

#: The files of a kit the review page may play.
MEDIA_FILES = ("clip.mp4", "vertical.mp4")

#: A run or kit name: no slashes, no dots, no spaces.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class LibraryError(Exception):
    """The kit, the run or the decision cannot be used."""


@dataclass(frozen=True)
class KitRef:
    """Where one kit lives."""

    run: str
    kit: str
    folder: Path


# --- names and paths ---------------------------------------------------------


def safe_name(name: str) -> str:
    """A run or kit name that cannot escape its folder."""
    text = str(name or "")
    if not text or not SAFE_NAME.match(text) or text in {".", ".."}:
        raise LibraryError(f"not a usable name: {name!r}")
    return text


def runs_root(data_dir: Path) -> Path:
    return Path(data_dir) / "runs"


def decisions_path(data_dir: Path) -> Path:
    return Path(data_dir) / DECISIONS_NAME


def handoff_folder(config: Config, run: str, kit: str) -> Path:
    """The hand-off folder of one kit: `<handoff_dir>/<account>/<run>-<kit>`."""
    return Path(config.handoff_dir) / config.account / f"{run}-{kit}"


def kit_folder(data_dir: Path, run: str, kit: str) -> Path:
    """The folder of one kit. Raises `LibraryError` when it is not there."""
    run_name = safe_name(run)
    kit_name = safe_name(kit)
    folder = runs_root(data_dir) / run_name / "kits" / kit_name
    if not folder.is_dir():
        raise LibraryError(f"there is no kit {run_name}/{kit_name}")
    return folder


def find_runs(data_dir: Path) -> list[Path]:
    """The run folders, newest first."""
    root = runs_root(data_dir)
    if not root.is_dir():
        return []
    runs = [path for path in root.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return runs


# --- the decision log --------------------------------------------------------


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_decisions(data_dir: Path) -> list[dict[str, Any]]:
    """Every decision on the log, oldest first. A broken line is skipped."""
    path = decisions_path(data_dir)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def append_decision(
    data_dir: Path,
    run: str,
    kit: str,
    action: str,
    who: str = "local",
    **values: Any,
) -> dict[str, Any]:
    """Add one decision to the log and return it."""
    record = {
        "time": now_iso(),
        "run": str(run),
        "kit": str(kit),
        "action": str(action),
        "who": str(who or "local"),
        "values": values,
    }
    path = decisions_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def status_for(decisions: Sequence[dict[str, Any]], run: str, kit: str) -> str:
    """The status of a kit: the last approve or reject wins."""
    status = STATUS_NEW
    for record in decisions:
        if record.get("run") != run or record.get("kit") != kit:
            continue
        action = record.get("action")
        if action == "approve":
            status = STATUS_APPROVED
        elif action == "reject":
            status = STATUS_REJECTED
    return status


# --- reading the kits --------------------------------------------------------


def read_kit(folder: Path) -> dict[str, Any]:
    """The `kit.json` of a kit folder."""
    path = Path(folder) / "kit.json"
    if not path.exists():
        raise LibraryError(f"{folder} has no kit.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LibraryError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LibraryError(f"{path} is not a kit")
    return payload


def write_kit(folder: Path, payload: dict[str, Any]) -> None:
    """Write the `kit.json` of a kit folder."""
    (Path(folder) / "kit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def read_signals(folder: Path) -> dict[str, Any]:
    """The `signals.json` of a kit folder (the signals that fired)."""
    path = Path(folder) / "signals.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def signal_chips(signals: dict[str, Any]) -> list[str]:
    """The signals as short words, for the chip row on the review card."""
    chips: list[str] = []
    loudness = signals.get("loudness")
    if isinstance(loudness, dict):
        chips.append(f"louder by {loudness.get('jump_lu', 0)} LU")
    sound = signals.get("sound")
    if isinstance(sound, dict):
        labels = [
            str(entry.get("label"))
            for entry in (sound.get("classes") or [])
            if isinstance(entry, dict)
        ][:2]
        if labels:
            chips.append(" and ".join(labels))
    if signals.get("transcript"):
        chips.append("the words land")
    chat = signals.get("chat")
    if isinstance(chat, dict):
        chips.append(f"chat {chat.get('messages_in_window', 0)} messages")
    if not chips:
        chips.append("quiet but real")
    return chips


def kit_summary(
    data_dir: Path,
    run: str,
    folder: Path,
    decisions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Everything the review card and the MCP need to know about one kit."""
    payload = read_kit(folder)
    signals = read_signals(folder)
    times = payload.get("times") or {}
    crop = payload.get("vertical_crop") or {}
    clip = Path(folder) / "clip.mp4"
    vertical = Path(folder) / "vertical.mp4"
    hooks = [str(hook) for hook in (payload.get("hooks") or [])]
    return {
        "run": str(run),
        "kit": Path(folder).name,
        "folder": str(Path(folder)),
        "rank": payload.get("rank"),
        "mmss": payload.get("mmss"),
        "peak": times.get("peak"),
        "score": payload.get("score"),
        "times": times,
        "signals": signals,
        "chips": signal_chips(signals),
        "hooks": hooks,
        "credit": str(payload.get("credit") or ""),
        "streamer": {
            "id": payload.get("streamer_id"),
            "name": payload.get("streamer_name"),
        },
        "status": status_for(decisions, run, Path(folder).name),
        "has_clip": clip.exists(),
        "has_vertical": vertical.exists(),
        "finished": (Path(folder) / "final" / "short.mp4").exists(),
        "clip_bytes": clip.stat().st_size if clip.exists() else 0,
        "vertical_bytes": vertical.stat().st_size if vertical.exists() else 0,
        "version": int(vertical.stat().st_mtime) if vertical.exists() else 0,
        "crop_x": float(crop.get("crop_x", 0.0) or 0.0),
        "media": {
            "clip": f"/media/{run}/{Path(folder).name}/clip.mp4",
            "vertical": f"/media/{run}/{Path(folder).name}/vertical.mp4",
        },
    }


def list_kits(data_dir: Path, status: str | None = None) -> list[dict[str, Any]]:
    """Every kit, newest run first and by rank inside a run."""
    decisions = read_decisions(data_dir)
    out: list[dict[str, Any]] = []
    for run_dir in find_runs(data_dir):
        kits_dir = run_dir / "kits"
        if not kits_dir.is_dir():
            continue
        summaries: list[dict[str, Any]] = []
        for folder in sorted(path for path in kits_dir.iterdir() if path.is_dir()):
            try:
                summaries.append(kit_summary(data_dir, run_dir.name, folder, decisions))
            except LibraryError:
                continue
        summaries.sort(
            key=lambda item: (
                item.get("rank") is None,
                item.get("rank") or 0,
                str(item.get("kit")),
            )
        )
        out.extend(summaries)
    if status:
        wanted = str(status).strip().lower()
        out = [kit for kit in out if kit["status"] == wanted]
    return out


def get_kit(data_dir: Path, run: str, kit: str) -> dict[str, Any]:
    """One kit, with its status."""
    folder = kit_folder(data_dir, run, kit)
    decisions = read_decisions(data_dir)
    return kit_summary(data_dir, safe_name(run), folder, decisions)


# --- the decisions -----------------------------------------------------------


def keep_previous(folder: Path, name: str) -> Path | None:
    """Move a kit file to `prev/` inside the kit. Never deletes it."""
    source = Path(folder) / name
    if not source.exists():
        return None
    prev = Path(folder) / "prev"
    prev.mkdir(parents=True, exist_ok=True)
    target = prev / name
    index = 2
    while target.exists():
        target = prev / f"{Path(name).stem}-{index}{Path(name).suffix}"
        index += 1
    shutil.move(str(source), str(target))
    return target


def resolve_source_path(source: Any) -> Path:
    """The source video of a kit, as an absolute path."""
    text = str(source or "")
    if not text:
        raise LibraryError("this kit has no source video")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise LibraryError(f"the source video is gone: {path}")
    return path


def refresh_transcript(
    data_dir: Path,
    run: str,
    folder: Path,
    source: Path,
    window: kit_builder.ClipWindow,
) -> int:
    """Re-slice the kit's transcript for a new clip window."""
    run_dir = runs_root(data_dir) / safe_name(run)
    words: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    moments_path = run_dir / "moments.json"
    if moments_path.exists():
        try:
            payload = json.loads(moments_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    if isinstance(payload, dict):
        words = kit_builder.clip_relative_words(
            kit_builder.load_transcript(run_dir, payload), window
        )
    (Path(folder) / "transcript.json").write_text(
        json.dumps(
            {"source": str(source), "clip": window.as_dict(), "words": words},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return len(words)


def apply_trim(
    data_dir: Path,
    run: str,
    kit: str,
    in_s: float,
    out_s: float,
    config: Config,
    who: str = "local",
) -> dict[str, Any]:
    """Re-cut the clip and the vertical from the source, inside the new window.

    The old files are moved to `prev/` inside the kit; nothing is deleted.
    """
    folder = kit_folder(data_dir, run, kit)
    payload = read_kit(folder)
    source = resolve_source_path(payload.get("source"))
    try:
        start = float(in_s)
        end = float(out_s)
    except (TypeError, ValueError):
        raise LibraryError("the in and out points must be numbers") from None
    if not (0.0 <= start < end):
        raise LibraryError("the in point must be before the out point, and both at or after zero")
    duration = audio.probe_duration(source)
    if end > duration + 0.5:
        raise LibraryError(
            f"the out point {end:.1f} s is past the end of the source ({duration:.1f} s)"
        )
    end = min(end, duration)
    times = payload.get("times") or {}
    try:
        peak = float(times.get("peak", start))
    except (TypeError, ValueError):
        peak = start
    peak = min(max(peak, start), end)
    window = kit_builder.ClipWindow(start=start, end=end, peak=peak)

    kept = [
        str(path)
        for path in (
            keep_previous(folder, "clip.mp4"),
            keep_previous(folder, "vertical.mp4"),
        )
        if path is not None
    ]
    kit_builder.cut_clip(source, window, Path(folder) / "clip.mp4", config)
    crop = payload.get("vertical_crop") or {}
    crop_x = float(crop.get("crop_x", 0.0) or 0.0)
    kit_builder.make_vertical(
        Path(folder) / "clip.mp4", Path(folder) / "vertical.mp4", config, crop_x=crop_x
    )
    words = refresh_transcript(data_dir, run, folder, source, window)

    payload["times"] = window.as_dict()
    payload["clip"] = str(Path(folder) / "clip.mp4")
    payload["vertical"] = str(Path(folder) / "vertical.mp4")
    payload["words"] = words
    write_kit(folder, payload)

    record = append_decision(
        data_dir,
        run,
        kit,
        "trim",
        who=who,
        in_s=round(start, 3),
        out_s=round(end, 3),
        duration=round(window.duration, 3),
    )
    return {
        "run": str(run),
        "kit": str(kit),
        "times": window.as_dict(),
        "kept_previous": kept,
        "decision": record,
    }


def apply_crop(
    data_dir: Path,
    run: str,
    kit: str,
    crop_x: float,
    config: Config,
    who: str = "local",
) -> dict[str, Any]:
    """Re-render the vertical with the picture nudged sideways.

    `crop_x` runs from -1 (as far left as the frame allows) to 1 (as far
    right). The old vertical is moved to `prev/` inside the kit.
    """
    folder = kit_folder(data_dir, run, kit)
    payload = read_kit(folder)
    clip = Path(folder) / "clip.mp4"
    if not clip.exists():
        raise LibraryError(f"there is no clip.mp4 in {run}/{kit} to crop")
    try:
        value = float(crop_x)
    except (TypeError, ValueError):
        raise LibraryError("the crop offset must be a number") from None
    if not (-1.0 <= value <= 1.0):
        raise LibraryError("the crop offset must be between -1 and 1")
    kept = keep_previous(folder, "vertical.mp4")
    filter_text = kit_builder.make_vertical(
        clip, Path(folder) / "vertical.mp4", config, crop_x=value
    )
    crop = dict(payload.get("vertical_crop") or {})
    crop.update(
        {
            "mode": "offset",
            "crop_x": value,
            "filter": filter_text,
            "size": f"{kit_builder.VERTICAL_WIDTH}x{kit_builder.VERTICAL_HEIGHT}",
        }
    )
    payload["vertical_crop"] = crop
    payload["vertical"] = str(Path(folder) / "vertical.mp4")
    write_kit(folder, payload)
    record = append_decision(data_dir, run, kit, "crop", who=who, crop_x=value)
    return {
        "run": str(run),
        "kit": str(kit),
        "crop_x": value,
        "filter": filter_text,
        "kept_previous": [str(kept)] if kept else [],
        "decision": record,
    }


def approve_kit(
    data_dir: Path,
    run: str,
    kit: str,
    hook_index: int = 0,
    config: Config | None = None,
    who: str = "local",
) -> dict[str, Any]:
    """Approve a kit: make the finished Short, copy it and the caption to the hand-off."""
    if config is None:
        raise LibraryError("approving a kit needs the config (for the hand-off folder)")
    folder = kit_folder(data_dir, run, kit)
    payload = read_kit(folder)
    hooks = [str(hook) for hook in (payload.get("hooks") or [])]
    if not hooks:
        raise LibraryError(f"{run}/{kit} has no hook lines to approve")
    try:
        index = int(hook_index)
    except (TypeError, ValueError):
        raise LibraryError("the hook number must be a number") from None
    if index < 0 or index >= len(hooks):
        raise LibraryError(f"{run}/{kit} has {len(hooks)} hook lines, not {index + 1}")
    vertical = Path(folder) / "vertical.mp4"
    if not vertical.exists():
        raise LibraryError(f"{run}/{kit} has no vertical.mp4 to hand off")

    hook = hooks[index]
    credit = str(payload.get("credit") or "")
    try:
        short = finisher.finish(folder, hook, credit)
    except audio.MediaError as exc:
        raise LibraryError(f"{run}/{kit} could not be finished: {exc}") from exc
    target = handoff_folder(config, run, kit)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(short, target / "short.mp4")
    (target / "caption.txt").write_text(f"{hook}\n{credit}\n", encoding="utf-8")
    record = append_decision(
        data_dir,
        run,
        kit,
        "approve",
        who=who,
        hook_index=index,
        handoff=str(target),
    )
    return {
        "run": str(run),
        "kit": str(kit),
        "status": STATUS_APPROVED,
        "handoff": str(target),
        "files": [str(target / "short.mp4"), str(target / "caption.txt")],
        "caption": [hook, credit],
        "decision": record,
    }


def reject_kit(
    data_dir: Path,
    run: str,
    kit: str,
    reason: str = "",
    config: Config | None = None,
    who: str = "local",
) -> dict[str, Any]:
    """Reject a kit. Nothing is deleted; the files stay where they are."""
    kit_folder(data_dir, run, kit)
    record = append_decision(
        data_dir, run, kit, "reject", who=who, reason=str(reason or "")
    )
    return {
        "run": str(run),
        "kit": str(kit),
        "status": STATUS_REJECTED,
        "reason": str(reason or ""),
        "decision": record,
    }


def handoff_list(config: Config) -> list[dict[str, Any]]:
    """Everything in the hand-off folder, one entry per approved kit."""
    root = Path(config.handoff_dir)
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for account_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for folder in sorted(path for path in account_dir.iterdir() if path.is_dir()):
            files = [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                }
                for path in sorted(folder.iterdir())
                if path.is_file()
            ]
            out.append(
                {
                    "account": account_dir.name,
                    "folder": folder.name,
                    "path": str(folder),
                    "files": files,
                }
            )
    return out
