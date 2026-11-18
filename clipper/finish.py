"""The finished Short. Burned ASS captions, a hook and a credit.

`finish(kit_dir, hook, credit)` writes `final/captions.ass` and
`final/short.mp4` inside the kit. `vertical.mp4` is left alone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from . import audio

WIDTH = 1080
HEIGHT = 1920
GROUP = 3
#: ASS colours are &HAABBGGRR.
WHITE = "&H00FFFFFF"
YELLOW = "&H0000E1FF"
BLACK = "&H00000000"

HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,78,{WHITE},{WHITE},{BLACK},{BLACK},-1,0,0,0,100,100,0,0,1,6,0,2,60,60,520,1
Style: Hook,DejaVu Sans,60,{BLACK},{BLACK},{WHITE},{WHITE},-1,0,0,0,100,100,0,0,3,14,0,8,60,60,230,1
Style: Credit,DejaVu Sans,40,{WHITE},{WHITE},{BLACK},{BLACK},0,0,0,0,100,100,0,0,1,4,0,2,60,60,300,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_escape(text: str) -> str:
    """Text safe inside an ASS dialogue line: no override blocks, no newlines."""
    out = str(text).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return " ".join(out.split())


def ass_time(seconds: float) -> str:
    cs = int(round(max(0.0, seconds) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def dialogue(start: float, end: float, style: str, text: str) -> str:
    return f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{text}"


def caption_lines(words: Sequence[dict[str, Any]]) -> list[str]:
    """One event per word: its group of 3, upper case, that word in yellow."""
    clean = [
        (ass_escape(str(w.get("word", ""))).upper(), float(w.get("start", 0.0)), float(w.get("end", 0.0)))
        for w in words
    ]
    clean = [w for w in clean if w[0]]
    lines: list[str] = []
    for first in range(0, len(clean), GROUP):
        group = clean[first : first + GROUP]
        for i, (_text, start, end) in enumerate(group):
            stop = group[i + 1][1] if i + 1 < len(group) else end
            stop = max(stop, start + 0.05)
            parts = [
                (f"{{\\c{YELLOW}&}}{t}{{\\c{WHITE}&}}" if j == i else t)
                for j, (t, _s, _e) in enumerate(group)
            ]
            lines.append(dialogue(start, stop, "Cap", " ".join(parts)))
    return lines


def build_ass(words: Sequence[dict[str, Any]], hook: str, credit: str, duration: float) -> str:
    lines = []
    if str(hook).strip():
        lines.append(dialogue(0.0, duration, "Hook", ass_escape(hook)))
    if str(credit).strip():
        lines.append(dialogue(0.0, duration, "Credit", ass_escape(credit)))
    lines.extend(caption_lines(words))
    return HEADER + "\n".join(lines) + "\n"


def _filter_path(path: Path) -> str:
    text = str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return text.replace(",", "\\,").replace("[", "\\[").replace("]", "\\]").replace(";", "\\;")


def finish(kit_dir: Path, hook: str, credit: str, memes: list[Path] | None = None) -> Path:
    """Burn captions, hook and credit into the kit's vertical. Returns short.mp4.

    `memes` (files from the meme library) are appended in order as closing cards.
    """
    kit_dir = Path(kit_dir)
    vertical = kit_dir / "vertical.mp4"
    if not vertical.exists():
        raise audio.MediaError(f"{kit_dir} has no vertical.mp4 to finish")
    words: list[dict[str, Any]] = []
    transcript = kit_dir / "transcript.json"
    if transcript.exists():
        try:
            payload = json.loads(transcript.read_text(encoding="utf-8"))
            words = payload.get("words", []) if isinstance(payload, dict) else []
        except json.JSONDecodeError:
            words = []
    final = kit_dir / "final"
    final.mkdir(exist_ok=True)
    ass = final / "captions.ass"
    ass.write_text(build_ass(words, hook, credit, audio.probe_duration(vertical)), encoding="utf-8")
    short = final / "short.mp4"
    audio.run(
        [
            audio.ffmpeg_bin(), "-hide_banner", "-v", "error", "-y",
            "-i", str(vertical),
            "-vf", f"ass={_filter_path(ass)}",
            "-c:v", "libx264", "-crf", "19", "-preset", "medium",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            str(short),
        ]
    )
    if memes:
        from . import meme as meme_outro

        meme_outro.append_memes(short, memes)
    return short
