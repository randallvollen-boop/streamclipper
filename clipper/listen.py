"""The listening agent: the moments people react to.

Four signals, each a time series on the same 0.5 s grid:

  sound      PANNs sound events (laughter, cheering, screaming, ...) with a
             weight per class. Music is NEGATIVE — copyrighted music inside a
             stream is a claim risk, so a music-heavy moment scores down.
  loudness   the ebur128 momentary loudness, as a jump over a rolling 60 s
             baseline of the file's own loudness.
  transcript faster-whisper words with their times, scored for hype words.
  chat       optional: messages per 10 s against the file's own baseline, plus
             a boost for hype reactions.

A moment's score is the weighted sum over the signals of each signal's peak
value inside a +/-2 s window. Peaks above the threshold are kept, peaks within
20 s of each other are merged into one moment, and the top N survive
(up to 5).

The expensive steps (the sound events, the transcript) are cached by the
hash of the input, so a second run over the same recording is fast.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import audio, sed
from .config import Config, load as load_config


@dataclass
class Word:
    """One transcribed word and when it was said."""

    word: str
    start: float
    end: float


@dataclass
class Signal:
    """One signal's time series, plus how to explain a bin of it."""

    name: str
    values: np.ndarray
    raw: np.ndarray
    detail: Callable[[int], dict[str, Any]] = lambda index: {}

    def peak(self, low: int, high: int) -> tuple[float, int]:
        """The largest value in bins [low, high] and the bin it was in."""
        if high < low or self.values.size == 0:
            return 0.0, low
        window = self.values[low : high + 1]
        index = int(np.argmax(window)) + low
        return float(self.values[index]), index


@dataclass
class Moment:
    """One ranked moment."""

    time: float
    score: float
    signals: dict[str, dict[str, Any]] = field(default_factory=dict)
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "time": round(self.time, 2),
            "mmss": mmss(self.time),
            "score": round(self.score, 3),
            "signals": self.signals,
        }


def mmss(seconds: float) -> str:
    """`125.4` -> `"02:05.4"`."""
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


# --- input -------------------------------------------------------------------


def file_hash(path: Path) -> str:
    """The sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(*parts: Any) -> str:
    """A short, stable fingerprint of the settings a cached step depends on."""
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def resolve_source(path: Path, work_dir: Path) -> Path:
    """The video file to work on.

    A directory of .ts chunks (a recording) is joined into one file first.
    """
    if path.is_file():
        return path
    if path.is_dir():
        destination = work_dir / "source.mp4"
        if not destination.exists():
            audio.concat_chunks(path, destination)
        return destination
    raise FileNotFoundError(f"{path} is neither a file nor a directory")


# --- signals -----------------------------------------------------------------


def build_sound_signal(
    columns: dict[str, np.ndarray],
    weights: dict[str, float],
    n_bins: int,
    hop_seconds: float,
    min_class_score: float,
    penalty_floor: float = 0.5,
) -> Signal:
    """The weighted sound-event series.

    `columns` maps a class name to its framewise scores (one value per
    PANNs frame). A negative weight (Music) only bites above `penalty_floor`,
    so a constant music bed does not cancel out every other signal.
    """
    per_class = sed.bin_max_scores(
        columns, n_bins, hop_seconds, min_score=min_class_score
    )
    values = np.zeros(n_bins, dtype=np.float64)
    raw = np.zeros(n_bins, dtype=np.float64)
    for name, weight in weights.items():
        scores = per_class.get(name)
        if scores is None:
            continue
        if weight >= 0:
            contribution = scores * weight
        else:
            contribution = -np.maximum(0.0, scores - penalty_floor) * abs(weight)
        values += contribution
        raw += scores

    def detail(index: int) -> dict[str, Any]:
        classes = []
        for name, weight in weights.items():
            scores = per_class.get(name)
            if scores is None or scores[index] <= 0:
                continue
            score = float(scores[index])
            if weight >= 0:
                contribution = score * weight
            else:
                contribution = -max(0.0, score - penalty_floor) * abs(weight)
            classes.append(
                {
                    "label": name,
                    "score": round(score, 3),
                    "weight": weight,
                    "contribution": round(contribution, 3),
                }
            )
        classes.sort(key=lambda item: -abs(item["contribution"]))
        return {"classes": classes}

    return Signal(name="sound", values=values, raw=raw, detail=detail)


def build_loudness_signal(
    series: Sequence[tuple[float, float]],
    n_bins: int,
    hop_seconds: float,
    baseline_seconds: float,
    min_jump_lu: float,
    gate_lu: float,
    weight: float,
    full_jump_lu: float = 10.0,
    min_history_seconds: float = 10.0,
) -> Signal:
    """The loudness-jump series: the momentary loudness over a rolling baseline.

    The baseline is the mean of the file's own loudness over the past
    `baseline_seconds`. The first `min_history_seconds` have no baseline yet,
    so nothing is called a jump there — otherwise the first seconds of a file
    always look like a jump against an empty past.

    The jump is normalised: a jump of `full_jump_lu` or more scores the whole
    weight, so one very loud transient cannot out-score every other signal.
    """
    lufs = np.full(n_bins, np.nan, dtype=np.float64)
    for time, value in series:
        index = int(time / hop_seconds)
        if index < 0 or index >= n_bins:
            continue
        if math.isnan(lufs[index]) or value > lufs[index]:
            lufs[index] = value
    # Bins the 10 Hz series did not land in keep the nearest neighbour.
    last = None
    for index in range(n_bins):
        if not math.isnan(lufs[index]):
            last = lufs[index]
        elif last is not None:
            lufs[index] = last
    first = None
    for index in range(n_bins):
        if not math.isnan(lufs[index]):
            first = lufs[index]
            break
    if first is not None:
        for index in range(n_bins):
            if math.isnan(lufs[index]):
                lufs[index] = first

    audible = lufs > gate_lu
    if not audible.any():
        return Signal(name="loudness", values=np.zeros(n_bins), raw=np.zeros(n_bins))
    overall = float(lufs[audible].mean())
    window_bins = max(1, int(round(baseline_seconds / hop_seconds)))
    history_bins = max(1, int(round(min_history_seconds / hop_seconds)))
    values = np.zeros(n_bins, dtype=np.float64)
    baseline_series = np.zeros(n_bins, dtype=np.float64)
    for index in range(n_bins):
        if index < history_bins:
            # No baseline yet: a jump cannot be established.
            baseline_series[index] = overall
            continue
        start = max(0, index - window_bins)
        past = lufs[start:index]
        past = past[past > gate_lu]
        baseline = float(past.mean()) if past.size else overall
        baseline_series[index] = baseline
        if not audible[index]:
            continue
        jump = float(lufs[index]) - baseline
        if jump >= min_jump_lu:
            values[index] = weight * min(1.0, jump / max(full_jump_lu, 1e-6))

    def detail(index: int) -> dict[str, Any]:
        return {
            "lufs": round(float(lufs[index]), 1),
            "baseline_lufs": round(float(baseline_series[index]), 1),
            "jump_lu": round(float(lufs[index] - baseline_series[index]), 1),
        }

    return Signal(name="loudness", values=values, raw=lufs, detail=detail)


def find_keyword_hits(words: Sequence[Word], keywords: dict[str, float]) -> list[dict[str, Any]]:
    """Where the hype words were said."""
    hits: list[dict[str, Any]] = []
    lowered = [word.word.lower().strip() for word in words]
    for keyword, weight in keywords.items():
        parts = keyword.lower().split()
        if not parts:
            continue
        span = len(parts)
        for index in range(len(words) - span + 1):
            if lowered[index : index + span] != parts:
                continue
            hits.append(
                {
                    "keyword": keyword,
                    "weight": weight,
                    "at": round(words[index].start, 2),
                    "text": " ".join(lowered[index : index + span]),
                }
            )
    hits.sort(key=lambda hit: hit["at"])
    return hits


def build_transcript_signal(
    words: Sequence[Word],
    keywords: dict[str, float],
    n_bins: int,
    hop_seconds: float,
    max_hits_per_keyword: int,
    weight: float,
) -> Signal:
    """The hype-word series."""
    hits = find_keyword_hits(words, keywords)
    values = np.zeros(n_bins, dtype=np.float64)
    per_bin: dict[int, list[dict[str, Any]]] = {}
    counted: dict[tuple[int, str], int] = {}
    for hit in hits:
        index = int(hit["at"] / hop_seconds)
        if index < 0 or index >= n_bins:
            continue
        key = (index, hit["keyword"])
        if counted.get(key, 0) >= max_hits_per_keyword:
            continue
        counted[key] = counted.get(key, 0) + 1
        values[index] += hit["weight"] * weight
        per_bin.setdefault(index, []).append(hit)

    def detail(index: int) -> dict[str, Any]:
        return {"hits": per_bin.get(index, [])}

    return Signal(name="transcript", values=values, raw=values.copy(), detail=detail)


def load_chat(path: Path) -> list[dict[str, Any]]:
    """Read a chat jsonl file of {t, user, text}."""
    messages: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not JSON: {exc}") from exc
            try:
                messages.append(
                    {
                        "t": float(payload["t"]),
                        "user": str(payload.get("user", "")),
                        "text": str(payload.get("text", "")),
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return messages


_EMOTE = re.compile(r"\[emote:\d+:([^\]]+)\]")
_TOKEN = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)


def chat_tokens(text: str) -> set[str]:
    """Lower-cased whole tokens of a chat message, with Kick emotes reduced to their names.

    Whole tokens, so a reaction like "w" matches the message "W" but not "what".
    """
    names = [name.lower() for name in _EMOTE.findall(text)]
    rest = _EMOTE.sub(" ", text)
    return set(names) | {token.lower() for token in _TOKEN.findall(rest)}


def build_chat_signal(
    messages: Sequence[dict[str, Any]],
    n_bins: int,
    hop_seconds: float,
    window_seconds: float,
    min_rate: float,
    reaction_weight: float,
    weight: float,
    reactions: Sequence[str],
    full_rate: float = 10.0,
    full_reactions: float = 5.0,
) -> Signal:
    """The chat-rate series: messages per 10 s against the file's own baseline.

    The spike is normalised, like the loudness jump: a chat that is ten
    messages per window busier than its own baseline scores the whole weight.
    Without that, one busy minute would out-score every other signal.
    """
    values = np.zeros(n_bins, dtype=np.float64)
    if not messages:
        return Signal(name="chat", values=values, raw=np.zeros(n_bins))
    reaction_set = {reaction.lower() for reaction in reactions}
    times = np.array([message["t"] for message in messages], dtype=np.float64)
    is_reaction = np.array(
        [bool(chat_tokens(message["text"]) & reaction_set) for message in messages],
        dtype=bool,
    )
    order = np.argsort(times)
    times, is_reaction = times[order], is_reaction[order]
    reaction_cum = np.concatenate([[0], np.cumsum(is_reaction)])
    centers = (np.arange(n_bins) + 0.5) * hop_seconds
    half = window_seconds / 2.0
    lo = np.searchsorted(times, centers - half, side="left")
    hi = np.searchsorted(times, centers + half, side="right")
    rates = (hi - lo).astype(np.float64)
    reaction_counts = (reaction_cum[hi] - reaction_cum[lo]).astype(np.float64)

    # Busy chats (a big streamer runs dozens of messages per window) are
    # judged against their own baseline: the spike is relative. full_rate is
    # the fraction above baseline that scores the whole weight (1.0 = double).
    baseline = float(np.median(rates)) if rates.size else 0.0
    react_base = float(np.median(reaction_counts)) if reaction_counts.size else 0.0
    for index in range(n_bins):
        if rates[index] < min_rate:
            continue
        rel = (rates[index] - baseline) / max(baseline, min_rate)
        spike_part = weight * min(1.0, max(0.0, rel) / max(full_rate, 1e-6))
        react_rel = (reaction_counts[index] - react_base) / max(react_base, 1.0)
        reaction_part = reaction_weight * min(
            1.0, max(0.0, react_rel) / max(full_reactions, 1e-6)
        )
        values[index] = spike_part + reaction_part

    def detail(index: int) -> dict[str, Any]:
        return {
            "messages_in_window": int(rates[index]),
            "baseline_per_window": round(baseline, 1),
            "reactions": int(reaction_counts[index]),
        }

    return Signal(name="chat", values=values, raw=rates, detail=detail)


# --- scoring -----------------------------------------------------------------


def _window_overlap(a: float, b: float, build_up: float, tail: float) -> float:
    """The share of one clip window that the window of another peak covers."""
    length = build_up + tail
    if length <= 0:
        return 0.0
    return max(0.0, length - abs(a - b)) / length


def score_moments(
    signals: Sequence[Signal],
    n_bins: int,
    hop_seconds: float,
    window_seconds: float,
    threshold: float,
    merge_seconds: float,
    top_n: int,
    build_up_seconds: float = 35.0,
    tail_seconds: float = 12.0,
) -> list[Moment]:
    """Find the peaks, merge the ones that are close, and rank them.

    A candidate whose clip window (build-up before the peak, tail after it)
    overlaps a higher-scoring chosen one by more than 50 % is dropped; the
    list is walked until `top_n` moments are chosen or it runs out.
    """
    half_bins = max(1, int(round((window_seconds / 2.0) / hop_seconds)))
    candidates: list[tuple[float, float]] = []
    for index in range(n_bins):
        low = max(0, index - half_bins)
        high = min(n_bins - 1, index + half_bins)
        total = 0.0
        for signal in signals:
            peak, _ = signal.peak(low, high)
            total += peak
        if total < threshold:
            continue
        # The score is flat across the whole window, so the moment is placed
        # at the strongest combined instant inside it.
        combined = np.zeros(high - low + 1, dtype=np.float64)
        for signal in signals:
            combined += signal.values[low : high + 1]
        peak_index = low + int(np.argmax(combined))
        candidates.append(((peak_index + 0.5) * hop_seconds, total))

    # Merge peaks that are closer together than `merge_seconds`, keeping the
    # strongest one of each group.
    merged: list[tuple[float, float]] = []
    for time, score in sorted(candidates):
        if merged and time - merged[-1][0] <= merge_seconds:
            if score > merged[-1][1]:
                merged[-1] = (time, score)
        else:
            merged.append((time, score))

    merged.sort(key=lambda item: -item[1])
    chosen: list[tuple[float, float]] = []
    for time, score in merged:
        if len(chosen) >= top_n:
            break
        if all(
            _window_overlap(time, other, build_up_seconds, tail_seconds) <= 0.5
            for other, _ in chosen
        ):
            chosen.append((time, score))
    moments: list[Moment] = []
    for rank, (time, score) in enumerate(chosen, start=1):
        index = min(n_bins - 1, max(0, int(time / hop_seconds)))
        low = max(0, index - half_bins)
        high = min(n_bins - 1, index + half_bins)
        details: dict[str, dict[str, Any]] = {}
        for signal in signals:
            peak, peak_index = signal.peak(low, high)
            if peak == 0:
                continue
            info = dict(signal.detail(peak_index))
            info["value"] = round(peak, 3)
            info["at"] = round((peak_index + 0.5) * hop_seconds, 2)
            details[signal.name] = info
        moments.append(Moment(time=time, score=score, signals=details, rank=rank))
    return moments


# --- the cached steps ---------------------------------------------------------


def sound_analysis(
    source: Path,
    cache_dir: Path,
    cache_key: str,
    config: Config,
    n_bins: int,
    hop_seconds: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], float]:
    """The framewise sound scores of the classes we care about (cached).

    Only the columns of the configured classes are kept: the full 527-class
    matrix is hundreds of megabytes per recording.
    """
    columns_path = cache_dir / f"{cache_key}.sound.npz"
    events_path = cache_dir / f"{cache_key}.sound.events.json"
    wanted = list(config.sound_weights)
    if columns_path.exists() and events_path.exists():
        with np.load(columns_path, allow_pickle=False) as bundle:
            names = [str(name) for name in bundle["names"]]
            columns = {
                name: np.asarray(bundle[f"c{index}"], dtype=np.float64)
                for index, name in enumerate(names)
            }
        events = json.loads(events_path.read_text(encoding="utf-8"))
        return columns, events, 0.0

    started = time.monotonic()
    samples = audio.decode_mono(source, audio.PANNS_SAMPLE_RATE)
    detector = sed.SoundEventDetector(config.panns_weights, config.panns_labels)
    framewise = detector.framewise_scores(samples)
    events = sed.extract_events(
        framewise,
        detector.labels,
        min_score=config.min_class_score,
        classes=wanted,
    )
    columns = sed.class_columns(framewise, detector.labels, wanted)
    elapsed = time.monotonic() - started

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        columns_path,
        names=np.array(list(columns)),
        **{f"c{index}": array for index, array in enumerate(columns.values())},
    )
    events_path.write_text(json.dumps(events), encoding="utf-8")
    return columns, events, elapsed


def transcript_analysis(
    source: Path,
    cache_dir: Path,
    cache_key: str,
    config: Config,
) -> tuple[list[Word], float]:
    """The transcript with word times (cached by the hash of the input)."""
    cache_path = cache_dir / f"{cache_key}.transcript.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Word(**item) for item in payload["words"]], 0.0
    started = time.monotonic()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "the transcript needs faster-whisper (pip install faster-whisper in .venv)"
        ) from exc
    samples = audio.decode_mono(source, audio.WHISPER_SAMPLE_RATE)
    model = WhisperModel(
        config.whisper_model, device="cpu", compute_type=config.whisper_compute
    )
    segments, _info = model.transcribe(
        samples,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    for segment in segments:
        for word in segment.words or []:
            text = (word.word or "").strip()
            if not text:
                continue
            words.append(Word(word=text, start=float(word.start), end=float(word.end)))
    elapsed = time.monotonic() - started
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"words": [word.__dict__ for word in words], "source": str(source)}),
        encoding="utf-8",
    )
    return words, elapsed


# --- the run ------------------------------------------------------------------


def analyse(
    source: Path,
    out_dir: Path,
    config: Config,
    chat_path: Path | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Rank the moments in one recording. Writes `moments.json` in `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    timings: dict[str, float] = {}
    total_started = time.monotonic()

    duration = audio.probe_duration(source)
    hop = config.hop_seconds
    n_bins = max(1, int(math.ceil(duration / hop)))
    top_n = top_n if top_n is not None else config.top_n

    source_hash = file_hash(source)
    sound_key = fingerprint(source_hash, "sound", config.min_class_score, sorted(config.sound_weights))
    transcript_key = fingerprint(source_hash, "transcript", config.whisper_model, config.whisper_compute)

    framewise, events, sound_s = sound_analysis(
        source, cache_dir, sound_key, config, n_bins, hop
    )
    timings["sound_s"] = round(sound_s, 2)
    timings["sound_cached"] = sound_s == 0.0

    started = time.monotonic()
    words, transcript_s = transcript_analysis(
        source, cache_dir, transcript_key, config
    )
    timings["transcript_s"] = round(transcript_s, 2)
    timings["transcript_load_s"] = round(time.monotonic() - started - transcript_s, 2)

    started = time.monotonic()
    loudness_series = audio.loudness_series(source)
    timings["loudness_s"] = round(time.monotonic() - started, 2)

    signals: list[Signal] = []
    signals.append(
        build_sound_signal(
            framewise,
            config.sound_weights,
            n_bins,
            hop,
            config.min_class_score,
            penalty_floor=float(config.get("signals.sound_penalty_floor", 0.5) or 0.5),
        )
    )
    loudness_cfg = config.loudness_settings
    signals.append(
        build_loudness_signal(
            loudness_series,
            n_bins,
            hop,
            float(loudness_cfg.get("baseline_seconds", 60.0)),
            float(loudness_cfg.get("min_jump_lu", 3.0)),
            float(loudness_cfg.get("gate_lu", -60.0)),
            float(loudness_cfg.get("weight", 2.0)),
            float(loudness_cfg.get("full_jump_lu", 10.0)),
            float(loudness_cfg.get("min_history_seconds", 10.0)),
        )
    )
    transcript_cfg = config.transcript_settings
    keywords = {
        str(name): float(weight)
        for name, weight in (transcript_cfg.get("keywords") or {}).items()
    }
    signals.append(
        build_transcript_signal(
            words,
            keywords,
            n_bins,
            hop,
            int(transcript_cfg.get("max_hits_per_keyword", 3)),
            float(transcript_cfg.get("weight", 1.0)),
        )
    )
    chat_cfg = config.chat_settings
    if chat_path is not None:
        messages = load_chat(chat_path)
        signals.append(
            build_chat_signal(
                messages,
                n_bins,
                hop,
                float(chat_cfg.get("window_seconds", 10.0)),
                float(chat_cfg.get("min_rate", 1.0)),
                float(chat_cfg.get("reaction_weight", 0.5)),
                float(chat_cfg.get("weight", 2.0)),
                [str(reaction) for reaction in (chat_cfg.get("reactions") or [])],
                float(chat_cfg.get("full_rate", 10.0)),
                float(chat_cfg.get("full_reactions", 5.0)),
            )
        )

    moments = score_moments(
        signals,
        n_bins,
        hop,
        config.window_seconds,
        config.threshold,
        config.merge_seconds,
        top_n,
        float(config.kit_settings.get("build_up_seconds", 35)),
        float(config.kit_settings.get("tail_seconds", 12)),
    )
    timings["total_s"] = round(time.monotonic() - total_started, 2)

    payload = {
        "source": str(source),
        "duration": round(duration, 2),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "hop_seconds": hop,
        "threshold": config.threshold,
        "window_seconds": config.window_seconds,
        "merge_seconds": config.merge_seconds,
        "top_n": top_n,
        "source_sha256": source_hash,
        "moments": [moment.as_dict() for moment in moments],
        "timings": timings,
        "sound_events": events[:200],
        "transcript_words": len(words),
        "cache": {
            "sound": sound_key,
            "transcript": transcript_key,
        },
    }
    (out_dir / "moments.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


# --- CLI ----------------------------------------------------------------------


def add_listen_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", help="the video file, or a directory of .ts chunks")
    parser.add_argument("--out", required=True, help="the run folder, e.g. data/runs/blade")
    parser.add_argument("--chat", default=None, help="a chat jsonl file of {t, user, text}")
    parser.add_argument("--top", type=int, default=None, help="how many moments to keep")


def run_listen(args: argparse.Namespace) -> int:
    config = load_config()
    source_path = Path(args.video).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = resolve_source(source_path, out_dir)
    chat_path = Path(args.chat).expanduser() if args.chat else None
    payload = analyse(source, out_dir, config, chat_path=chat_path, top_n=args.top)

    print(f"source: {source}  ({payload['duration']:.1f} s)")
    print()
    print(f"{'#':>2}  {'time':>8}  {'score':>6}  signals")
    for moment in payload["moments"]:
        fired = ", ".join(sorted(moment["signals"]))
        print(f"{moment['rank']:>2}  {moment['mmss']:>8}  {moment['score']:>6.2f}  {fired}")
    print()
    print("timings: " + ", ".join(f"{key}={value}" for key, value in payload["timings"].items()))
    print(f"moments.json: {out_dir / 'moments.json'}")
    return 0
