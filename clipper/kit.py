"""The remix kits.

A kit is the source material a creator remixes: the cut clip, the vertical
crop, the transcript with word times, the credit line, three hook lines and
the signals that fired. Nothing is posted and nothing is styled — the kit is
input, not a finished post.

The clip window follows these rules: 30-45 s of build-up before the
peak, 10-15 s after it, 45-60 s in total, clamped to the file's bounds. The
peak is always inside the clip; if the peak sits near the start or the end
of the file, the window slides rather than getting shorter.

The vertical crop is a 9:16 centre crop at 1080x1920 — no letterbox, no
padding. A face/action-aware crop is a later slice; the TODO is in the code
and in `kit.json`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import audio
from .allowlist import Streamer
from .config import REPO_ROOT, Config, load as load_config

#: The pseudo-streamer for our own footage.
SELF_ID = "self"
#: The credit line used for our own footage.
SELF_CREDIT = "me"
#: The vertical output size.
VERTICAL_WIDTH = 1080
VERTICAL_HEIGHT = 1920


class KitError(Exception):
    """The kit could not be built."""


@dataclass(frozen=True)
class ClipWindow:
    """The in and out points of one clip."""

    start: float
    end: float
    peak: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_dict(self) -> dict[str, float]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "peak": round(self.peak, 2),
            "duration": round(self.duration, 2),
        }


def clip_window(
    peak: float,
    duration: float,
    build_up: float,
    tail: float,
    min_total: float,
    max_total: float,
) -> ClipWindow:
    """The window around a peak: build-up before it, tail after it.

    The total is kept inside [min_total, max_total] when the file is long
    enough, and the peak always stays inside the clip.
    """
    length = build_up + tail
    if duration <= length:
        # The file is shorter than one clip: the whole file is the clip.
        return ClipWindow(start=0.0, end=duration, peak=min(peak, duration))
    length = min(max(length, min_total), max_total, duration)
    start = peak - build_up
    end = start + length
    if start < 0:
        start = 0.0
        end = length
    if end > duration:
        end = duration
        start = max(0.0, end - length)
    if peak < start:
        start = peak
        end = min(duration, start + length)
    if peak > end:
        end = peak
        start = max(0.0, end - length)
    return ClipWindow(start=max(0.0, start), end=min(duration, end), peak=peak)


def crop_filter(width: int, height: int, crop_x: float = 0.0) -> str:
    """The 9:16 crop of a frame, as an ffmpeg `crop` filter.

    `crop_x` nudges the crop sideways: -1 is as far left as the frame allows,
    1 as far right, 0 is the centre. It is a fraction of the room the crop
    has to move in, so the picture slides smoothly. A frame that is not wider
    than 9:16 has no sideways room, so the offset stays at 0 there.
    """
    target = VERTICAL_WIDTH / VERTICAL_HEIGHT
    source = width / height
    if source > target:
        # Too wide: keep the full height, cut the sides.
        crop_width = int(round(height * target))
        crop_width -= crop_width % 2
        crop_width = max(2, min(crop_width, width))
        slack = width - crop_width
        position = max(-1.0, min(1.0, float(crop_x)))
        offset_x = int(round(slack * (position + 1.0) / 2.0))
        offset_x = max(0, min(offset_x, slack))
        offset_x -= offset_x % 2
        return f"crop={crop_width}:{height}:{offset_x}:0"
    # Too tall (or already 9:16): keep the full width, cut the top and bottom.
    crop_height = int(round(width / target))
    crop_height -= crop_height % 2
    crop_height = max(2, min(crop_height, height))
    offset_y = (height - crop_height) // 2
    offset_y -= offset_y % 2
    return f"crop={width}:{crop_height}:0:{offset_y}"


def centre_crop(width: int, height: int) -> str:
    """The 9:16 centre crop of a frame, as an ffmpeg `crop` filter."""
    return crop_filter(width, height, 0.0)


def cut_clip(source: Path, window: ClipWindow, destination: Path, config: Config) -> None:
    """Cut the clip: re-encoded h264/aac, frame accurate."""
    kit = config.kit_settings
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        audio.ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-ss", f"{window.start:.3f}",
        "-i", str(source),
        "-t", f"{window.duration:.3f}",
        "-c:v", str(kit.get("video_codec", "libx264")),
        "-preset", "medium",
        "-crf", str(kit.get("crf", 20)),
        "-pix_fmt", "yuv420p",
        "-c:a", str(kit.get("audio_codec", "aac")),
        "-b:a", str(kit.get("audio_bitrate", "192k")),
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        str(destination),
    ]
    audio.run(argv)


def make_vertical(
    source: Path, destination: Path, config: Config, crop_x: float = 0.0
) -> str:
    """The 9:16 crop at 1080x1920. Letterbox-free.

    `crop_x` nudges the crop sideways (-1 = left, 1 = right, 0 = centre); the
    review page's crop slider is the manual version of it.

    TODO(face-aware): a centre crop cuts the subject out of frame when the
    action is off-centre. A face/action-aware crop is a later slice.
    """
    info = audio.probe_video(source)
    crop = crop_filter(info["width"], info["height"], crop_x)
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        audio.ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-i", str(source),
        "-vf", f"{crop},scale={VERTICAL_WIDTH}:{VERTICAL_HEIGHT}:flags=lanczos,setsar=1",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(destination),
    ]
    audio.run(argv)
    return crop


def credit_line(streamer: Streamer | None, streamer_id: str) -> str:
    """The credit line: the streamer's own format, or the default."""
    if streamer_id == SELF_ID:
        return SELF_CREDIT
    if streamer is None:
        return ""
    if streamer.conditions.credit_format:
        return streamer.conditions.credit_format
    return f"@{streamer.name} on {streamer.platform}"


def strongest_phrase(moment: dict[str, Any]) -> str:
    """The strongest transcript phrase inside the moment."""
    transcript = moment.get("signals", {}).get("transcript", {})
    hits = transcript.get("hits") or []
    if not hits:
        return ""
    best = max(hits, key=lambda hit: float(hit.get("weight", 0.0)))
    text = str(best.get("text") or best.get("keyword") or "").strip()
    return text


def signal_sentence(moment: dict[str, Any]) -> str:
    """A sentence naming the signals that fired. No language model."""
    signals = moment.get("signals", {})
    parts: list[str] = []
    loudness = signals.get("loudness")
    if loudness:
        parts.append(f"the loudness jumps {loudness.get('jump_lu', 0)} LU")
    sound = signals.get("sound")
    if sound:
        labels = [entry["label"] for entry in sound.get("classes", [])][:2]
        if labels:
            parts.append(f"{' and '.join(labels)} hits")
    transcript = signals.get("transcript")
    if transcript:
        parts.append("the words land right on it")
    chat = signals.get("chat")
    if chat:
        parts.append(f"chat spikes to {chat.get('messages_in_window', 0)} messages")
    if not parts:
        return "the moment is quiet but real"
    return ", ".join(parts)


HOOK_LIMIT = 60
FILLERS = {"uh", "um", "uhh", "umm", "er", "erm", "ah", "hmm", "mm"}
REACTION_LEAD = "Chat lost it when he said "


def clean_words(tokens: list[str]) -> list[str]:
    """Drop filler ("uh", "um") and immediate repeats ("no no no" -> "no")."""
    out: list[str] = []
    for token in tokens:
        bare = re.sub(r"[^\w']+", "", token).lower()
        if not bare or bare in FILLERS:
            continue
        if out and re.sub(r"[^\w']+", "", out[-1]).lower() == bare:
            continue
        out.append(token)
    return out


def trim_words(text: str, limit: int) -> str:
    """`text` cut at a word boundary to at most `limit` characters."""
    text = text.strip().rstrip(",;:-")
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] if " " in text[:limit + 1] else text[:limit]
    return cut.strip().rstrip(",;:-")


def peak_quote(words: list[dict[str, Any]] | None, peak: float, limit: int = HOOK_LIMIT) -> str:
    """The sentence (or 6-12 words) around the peak, cleaned, at most `limit` chars."""
    words = [w for w in (words or []) if str(w.get("word", "")).strip()]
    if not words:
        return ""
    centre = min(range(len(words)), key=lambda i: abs(
        (float(words[i].get("start", 0.0)) + float(words[i].get("end", 0.0))) / 2 - peak))
    low, high = max(0, centre - 6), min(len(words), centre + 6)
    # Prefer the sentence: stop at the nearest sentence ends if 6+ words remain.
    for i in range(centre - 1, low - 1, -1):
        if words[i]["word"].rstrip().endswith((".", "?", "!")):
            if high - (i + 1) >= 6:
                low = i + 1
            break
    for i in range(centre, high):
        if words[i]["word"].rstrip().endswith((".", "?", "!")):
            if (i + 1) - low >= 6:
                high = i + 1
            break
    tokens = clean_words([str(w["word"]).strip() for w in words[low:high]])
    text = " ".join(tokens).strip().rstrip(".,;:!?-")
    return trim_words(text, limit)


def build_hooks(
    moment: dict[str, Any], credit: str, words: list[dict[str, Any]] | None = None
) -> list[str]:
    """Three hook lines. No language model.

    1: the quote around the peak. 2: the reaction plus that quote.
    3: the signal sentence. `words` are the transcript words (source times);
    without them the strongest keyword hit stands in for the quote.
    """
    quote = peak_quote(words, float(moment.get("time", 0.0)))
    if not quote:
        phrase = " ".join(clean_words(strongest_phrase(moment).split()))
        quote = trim_words(phrase, HOOK_LIMIT)
    if quote:
        first = quote
        room = HOOK_LIMIT - len(REACTION_LEAD) - 2
        second = f'{REACTION_LEAD}"{trim_words(quote, room)}"'
    else:
        first = "Wait for this — and then it hits."
        second = "Chat lost it at this moment"
    third = f"{signal_sentence(moment)}."
    return [first, second, third]


def safe_name(text: str) -> str:
    """A file-name-safe version of `text` (a colon breaks shell commands)."""
    return re.sub(r"[^0-9A-Za-z._-]+", "-", text).strip("-") or "clip"


def load_run(run_dir: Path) -> dict[str, Any]:
    """The `moments.json` of a run."""
    path = run_dir / "moments.json"
    if not path.exists():
        raise KitError(f"{path} does not exist — run `clipper listen` first")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KitError(f"{path} is not valid JSON: {exc}") from exc


def load_transcript(run_dir: Path, run: dict[str, Any]) -> list[dict[str, Any]]:
    """The words of the run, from the listen cache."""
    key = (run.get("cache") or {}).get("transcript")
    path = None
    if key:
        candidate = run_dir / "cache" / f"{key}.transcript.json"
        if candidate.exists():
            path = candidate
    if path is None:
        found = sorted((run_dir / "cache").glob("*.transcript.json"))
        path = found[-1] if found else None
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload.get("words", [])


def clip_relative_words(words: list[dict[str, Any]], window: ClipWindow) -> list[dict[str, Any]]:
    """The words inside the clip, with times relative to the clip's start."""
    out: list[dict[str, Any]] = []
    for word in words:
        start = float(word.get("start", 0.0))
        end = float(word.get("end", 0.0))
        if end < window.start or start > window.end:
            continue
        out.append(
            {
                "word": word.get("word", ""),
                "start": round(max(0.0, start - window.start), 3),
                "end": round(max(0.0, end - window.start), 3),
                "source_start": round(start, 3),
            }
        )
    return out


def resolve_source(path_text: str) -> Path:
    """The source video of a run, relative to the repo root if it is."""
    path = Path(path_text)
    if path.exists():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    raise KitError(f"the source {path_text} does not exist")


def build_kits(
    run_dir: Path,
    streamer_id: str,
    config: Config,
    streamer: Streamer | None = None,
) -> list[dict[str, Any]]:
    """Build one kit per moment in the run."""
    run = load_run(run_dir)
    source = resolve_source(str(run.get("source", "")))
    duration = float(run.get("duration") or audio.probe_duration(source))
    words = load_transcript(run_dir, run)
    kit_cfg = config.kit_settings
    build_up = float(kit_cfg.get("build_up_seconds", 35))
    tail = float(kit_cfg.get("tail_seconds", 12))
    min_total = float(kit_cfg.get("min_total_seconds", 45))
    max_total = float(kit_cfg.get("max_total_seconds", 60))
    credit = credit_line(streamer, streamer_id)

    kits_dir = run_dir / "kits"
    kits_dir.mkdir(parents=True, exist_ok=True)
    built: list[dict[str, Any]] = []
    for moment in run.get("moments", []):
        rank = int(moment.get("rank", 0))
        peak = float(moment.get("time", 0.0))
        window = clip_window(peak, duration, build_up, tail, min_total, max_total)
        folder = kits_dir / f"{rank}-{safe_name(str(moment.get('mmss', '')))}"
        folder.mkdir(parents=True, exist_ok=True)

        clip_path = folder / "clip.mp4"
        vertical_path = folder / "vertical.mp4"
        cut_clip(source, window, clip_path, config)
        crop = make_vertical(clip_path, vertical_path, config)

        clip_words = clip_relative_words(words, window)
        (folder / "transcript.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "clip": window.as_dict(),
                    "words": clip_words,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (folder / "credit.txt").write_text(credit + "\n", encoding="utf-8")
        hooks = build_hooks(moment, credit, words)
        (folder / "hooks.txt").write_text("\n".join(hooks) + "\n", encoding="utf-8")
        (folder / "signals.json").write_text(
            json.dumps(moment.get("signals", {}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        payload = {
            "source": str(source),
            "source_sha256": run.get("source_sha256"),
            "run": str(run_dir),
            "streamer_id": streamer_id,
            "streamer_name": streamer.name if streamer else (SELF_CREDIT if streamer_id == SELF_ID else streamer_id),
            "credit": credit,
            "rank": rank,
            "mmss": moment.get("mmss"),
            "times": window.as_dict(),
            "score": moment.get("score"),
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "clip": str(clip_path),
            "vertical": str(vertical_path),
            "vertical_crop": {
                "mode": "centre",
                "filter": crop,
                "size": f"{VERTICAL_WIDTH}x{VERTICAL_HEIGHT}",
                "todo": "face/action-aware crop (the centre crop cuts the subject out when the action is off-centre)",
            },
            "hooks": hooks,
            "words": len(clip_words),
        }
        (folder / "kit.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        built.append(payload)
    return built


# --- CLI ----------------------------------------------------------------------


def add_kit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run", help="the run folder, e.g. data/runs/blade")
    parser.add_argument(
        "--streamer",
        required=True,
        help="the allow-list id of the streamer, or 'self' for our own footage",
    )
    parser.add_argument(
        "--file", default=None, help="path to streamers.yaml (default: config/streamers.yaml)"
    )


def run_kits(args: argparse.Namespace) -> int:
    from . import allowlist

    config = load_config()
    run_dir = Path(args.run).expanduser()
    streamer: Streamer | None = None
    if args.streamer != SELF_ID:
        try:
            streamers = allowlist.load(args.file)
        except allowlist.AllowlistError as exc:
            print(f"allow-list error: {exc}", file=sys.stderr)
            return 2
        streamer = next((s for s in streamers if s.id == args.streamer), None)
        if streamer is None:
            print(
                f"no streamer with id {args.streamer!r} on the allow-list",
                file=sys.stderr,
            )
            return 2
    try:
        kits = build_kits(run_dir, args.streamer, config, streamer)
    except KitError as exc:
        print(f"kit error: {exc}", file=sys.stderr)
        return 2
    if not kits:
        print(f"{run_dir} has no moments — nothing to build.")
        return 0
    for kit in kits:
        print(
            f"#{kit['rank']} {kit['mmss']} score={kit['score']} "
            f"{kit['times']['start']:.1f}-{kit['times']['end']:.1f}s "
            f"({kit['times']['duration']:.1f}s) -> {Path(kit['clip']).parent}"
        )
    print(f"{len(kits)} kit(s) in {run_dir / 'kits'}")
    return 0
