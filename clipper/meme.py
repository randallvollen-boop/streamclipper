"""The meme outro: a short meme card appended after the clip, chosen by emotion.

The meme library is a folder of our own files, one sub-folder per emotion:

    data/memes/<emotion>/<name>.{png,jpg,jpeg,webp,gif,mp4,mov}

Nothing is downloaded. The person (or Claude, through the MCP) picks a file
from this folder; `append_meme` puts it after the finished Short.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import audio

EMOTIONS = ("funny", "shock", "hype", "cringe", "awkward", "rage", "sad", "win")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXT = {".gif", ".mp4", ".mov"}
OUTRO_SECONDS = 2.5
MAX_OUTRO_SECONDS = 6.0

#: Signal names (see `listen`) that point at an emotion, strongest first.
SIGNAL_EMOTION = (
    ("Laughter", "funny"), ("Giggle", "funny"), ("Belly laugh", "funny"), ("Chuckle, chortle", "funny"),
    ("Gasp", "shock"), ("Screaming", "shock"), ("Explosion", "shock"),
    ("Cheering", "hype"), ("Applause", "hype"), ("Whoop", "hype"),
    ("Yell", "rage"), ("Shout", "rage"),
)


class MemeError(Exception):
    pass


def library_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "memes"


def list_memes(data_dir: Path, emotion: str | None = None) -> dict[str, list[str]]:
    """Meme files by emotion (relative names). One emotion, or all of them."""
    root = library_dir(data_dir)
    emotions = [emotion] if emotion else sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    out: dict[str, list[str]] = {}
    for name in emotions:
        folder = root / name
        files = sorted(
            f"{name}/{p.name}" for p in folder.glob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXT | VIDEO_EXT
        ) if folder.is_dir() else []
        out[name] = files
    return out


def resolve(data_dir: Path, meme: str) -> Path:
    """A meme name from the library, or a refusal. Never a path outside it."""
    root = library_dir(data_dir).resolve()
    path = (root / meme).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise MemeError(f"{meme!r} is not inside the meme library {root}") from None
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXT | VIDEO_EXT:
        raise MemeError(f"no meme {meme!r} in {root} (try list_memes)")
    return path


def suggest_emotion(signals: list[str]) -> str:
    """The emotion a kit's signals point at; `funny` when none does."""
    text = " ".join(signals).lower()
    for signal, emotion in SIGNAL_EMOTION:
        if signal.lower() in text:
            return emotion
    return "hype" if "louder" in text or "chat" in text else "funny"


def _has_audio(path: Path) -> bool:
    out = audio.run([audio.ffprobe_bin(), "-v", "error", "-select_streams", "a:0",
                     "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)]).stdout
    return "audio" in out


def append_memes(short: Path, memes: list[Path], seconds: float = OUTRO_SECONDS) -> Path:
    """Re-encode `short` with the memes, in order, as closing cards. Replaces `short`.

    A reaction video keeps its own sound (up to its own length, at most
    MAX_OUTRO_SECONDS); a still image shows for `seconds` over silence.
    """
    short = Path(short)
    memes = [Path(m) for m in memes]
    if not memes:
        return short
    info = audio.probe_video(short)
    w, h = int(info.get("width", 1080)), int(info.get("height", 1920))
    fps = info.get("fps") or 30
    inputs = ["-i", str(short)]
    parts = [f"[0:v]setsar=1,fps={fps},format=yuv420p[v0];[0:a]aresample=44100,asetpts=PTS-STARTPTS[a0]"]
    for n, meme in enumerate(memes, start=1):
        still = meme.suffix.lower() in IMAGE_EXT
        if still:
            length = max(0.5, min(float(seconds), MAX_OUTRO_SECONDS))
            inputs += ["-loop", "1", "-t", f"{length}", "-i", str(meme)]
        else:
            length = min(audio.probe_duration(meme), MAX_OUTRO_SECONDS)
            inputs += ["-t", f"{length}", "-i", str(meme)]
        parts.append(
            f"[{n}:v]split[a{n}][b{n}];[a{n}]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=30[bg{n}];"
            f"[b{n}]scale={w}:{h}:force_original_aspect_ratio=decrease[fg{n}];"
            f"[bg{n}][fg{n}]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps},format=yuv420p,setpts=PTS-STARTPTS[v{n}]"
        )
        if not still and _has_audio(meme):
            parts.append(f"[{n}:a]aresample=44100,apad,atrim=0:{length},asetpts=PTS-STARTPTS[a{n}]")
        else:
            parts.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{length},asetpts=PTS-STARTPTS[a{n}]")
    chain = "".join(f"[v{n}][a{n}]" for n in range(len(memes) + 1))
    parts.append(f"{chain}concat=n={len(memes) + 1}:v=1:a=1[v][a]")
    tmp = short.with_suffix(".meme.mp4")
    audio.run([
        audio.ffmpeg_bin(), "-hide_banner", "-v", "error", "-y", *inputs,
        "-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "19", "-preset", "medium",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp),
    ])
    tmp.replace(short)
    return short
