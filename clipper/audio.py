"""ffmpeg/ffprobe helpers.

Everything here shells out to ffmpeg; nothing is decoded in Python. Audio is
read as raw float32 through a pipe, because the PyAV path failed in testing.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

#: The sample rate faster-whisper wants.
WHISPER_SAMPLE_RATE = 16000
#: The sample rate PANNs wants.
PANNS_SAMPLE_RATE = 32000


class MediaError(Exception):
    """ffmpeg or ffprobe failed, or the media is not what we expected."""


def ffmpeg_bin() -> str:
    return "ffmpeg"


def ffprobe_bin() -> str:
    return "ffprobe"


def run(argv: list[str], *, timeout: float = 3600.0, text: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        argv,
        capture_output=True,
        timeout=timeout,
        text=text,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", "replace") if not text else (proc.stderr or "")
        tail = stderr.strip()[-400:]
        raise MediaError(f"{argv[0]} failed (rc={proc.returncode}): {tail}")
    return proc


def probe_duration(path: Path) -> float:
    """The media duration in seconds."""
    proc = run(
        [
            ffprobe_bin(), "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ],
        timeout=60.0,
    )
    raw = (proc.stdout or "").strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise MediaError(f"{path}: duration is unknown") from None
    if value <= 0:
        raise MediaError(f"{path}: duration is unknown")
    return value


def probe_video(path: Path) -> dict:
    """Width, height and frame rate of the first video stream."""
    proc = run(
        [
            ffprobe_bin(), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
            "-of", "json", str(path),
        ],
        timeout=60.0,
    )
    try:
        streams = json.loads(proc.stdout or "{}").get("streams") or []
    except json.JSONDecodeError:
        streams = []
    if not streams:
        raise MediaError(f"{path}: no video stream")
    stream = streams[0]

    def _rate(value: str) -> float:
        try:
            num, _, den = str(value).partition("/")
            den_value = float(den) if den else 1.0
            return float(num) / den_value if den_value else 0.0
        except (TypeError, ValueError):
            return 0.0

    rate = _rate(stream.get("avg_frame_rate") or "") or _rate(stream.get("r_frame_rate") or "")
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": rate,
    }


def decode_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Decode the audio to a mono float32 array at `sample_rate`."""
    argv = [
        ffmpeg_bin(), "-v", "error", "-i", str(path),
        "-map", "a:0",
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sample_rate), "-",
    ]
    proc = subprocess.run(argv, capture_output=True, timeout=3600.0)
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip()[-400:]
        raise MediaError(f"ffmpeg failed (rc={proc.returncode}): {tail}")
    data = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if data.size == 0:
        raise MediaError(f"{path}: no audio stream")
    return data


def loudness_series(path: Path) -> list[tuple[float, float]]:
    """The ebur128 momentary loudness time series: [(seconds, LUFS), ...].

    The ebur128 filter reports one momentary value every 100 ms (a 400 ms
    window sliding by 100 ms), so the series runs at 10 Hz.
    """
    argv = [
        ffmpeg_bin(), "-hide_banner", "-nostats", "-v", "error",
        "-i", str(path),
        "-af", "ebur128=metadata=1,ametadata=print:key=lavfi.r128.M:file=-",
        "-f", "null", "-",
    ]
    proc = subprocess.run(argv, capture_output=True, timeout=3600.0, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-400:]
        raise MediaError(f"ffmpeg ebur128 failed (rc={proc.returncode}): {tail}")
    series: list[tuple[float, float]] = []
    pending_time: float | None = None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("frame:") and "pts_time:" in line:
            _, _, tail = line.partition("pts_time:")
            try:
                pending_time = float(tail.split()[0])
            except (IndexError, ValueError):
                pending_time = None
        elif line.startswith("lavfi.r128.M=") and pending_time is not None:
            try:
                value = float(line.partition("=")[2])
            except ValueError:
                value = float("-inf")
            series.append((pending_time, value))
            pending_time = None
    return series


def concat_chunks(directory: Path, destination: Path) -> Path:
    """Join the .ts chunks of a recording, in order, into one file.

    The chunks are the recorder's 10-minute segments; the concat demuxer
    joins them without re-encoding.
    """
    chunks = sorted(p for p in directory.glob("*.ts") if p.is_file())
    if not chunks:
        raise MediaError(f"{directory}: no .ts chunks to join")
    list_path = destination.with_suffix(".concat.txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(f"file '{chunk.resolve()}'\n")
    argv = [
        ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(destination),
    ]
    run(argv)
    return destination
