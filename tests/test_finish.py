"""The finished Short."""
from __future__ import annotations

import json
import re
from pathlib import Path

from clipper import audio, finish, library
from test_review import KIT, RUN, world  # noqa: F401  (fixture)

WORDS = [
    {"word": "what", "start": 0.2, "end": 0.6},
    {"word": "is", "start": 0.7, "end": 0.9},
    {"word": "{that}\\", "start": 1.0, "end": 1.5},
]


def test_ass_text():
    text = finish.build_ass(WORDS + [{"word": "ok", "start": 2.0, "end": 2.3}], "HOOK line", "@Streamer", 3.0)
    assert "Style: Cap,DejaVu Sans,78" in text and ",6,0,2,60,60,520," in text
    assert "Style: Hook,DejaVu Sans,60" in text and ",3,14,0,8,60,60,230," in text
    assert "Style: Credit,DejaVu Sans,40" in text and ",60,60,300," in text
    events = [l for l in text.splitlines() if l.startswith("Dialogue")]
    cap = [l for l in events if ",Cap," in l]
    assert len(cap) == 4  # one event per word; two groups (3 + 1)
    assert all(l.count("&H0000E1FF&") == 1 for l in cap)
    assert "WHAT" in cap[0] and "{\\c&H0000E1FF&}WHAT{" in cap[0]
    assert "{\\c&H0000E1FF&}IS{" in cap[1]
    assert "OK" in cap[3] and "WHAT" not in cap[3]
    assert "\\{THAT\\}\\\\" in cap[2]
    hook = [l for l in events if ",Hook," in l]
    credit = [l for l in events if ",Credit," in l]
    assert len(hook) == 1 and hook[0].endswith("HOOK line")
    assert hook[0].startswith("Dialogue: 0,0:00:00.00,0:00:03.00")
    assert len(credit) == 1 and credit[0].endswith("@Streamer")


def test_finish_encodes_a_real_short(tmp_path: Path):
    kit = tmp_path / "kit"
    kit.mkdir()
    audio.run([
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=10:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(kit / "vertical.mp4"),
    ])
    (kit / "transcript.json").write_text(json.dumps({"words": WORDS}), encoding="utf-8")
    short = finish.finish(kit, "Wait for this", "@Streamer on kick")
    assert short == kit / "final" / "short.mp4" and short.exists()
    assert (kit / "final" / "captions.ass").exists()
    assert (kit / "vertical.mp4").exists()
    out = audio.run([
        "ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
        str(short),
    ]).stdout
    assert "1080,1920" in out
    assert abs(audio.probe_duration(short) - 3.0) <= 0.1


def test_approve_hands_off_the_short(world):  # noqa: F811
    result = library.approve_kit(world.data_dir, RUN, KIT, 0, world.config)
    folder = Path(result["handoff"])
    assert (folder / "short.mp4").exists()
    assert not (folder / "vertical.mp4").exists()
    assert (folder / "caption.txt").exists()
    assert (world.kit_folder / "vertical.mp4").exists()
    assert library.get_kit(world.data_dir, RUN, KIT)["finished"] is True
