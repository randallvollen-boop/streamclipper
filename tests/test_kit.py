"""The remix kits.

Rules under test (decision a-length): 30-45 s of build-up before the peak,
10-15 s after it, 45-60 s in total, clamped to the file's bounds, with the
peak always inside the clip. The vertical is a letterbox-free 9:16 centre
crop. The credit line comes from the streamer. The hooks are templated, not
written by a language model.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from clipper.allowlist import Conditions, Streamer
from clipper.kit import (
    ClipWindow,
    build_hooks,
    centre_crop,
    clip_relative_words,
    clip_window,
    credit_line,
    safe_name,
    strongest_phrase,
)


def make_streamer(credit_format: str | None = None, name="ExampleStreamer", platform="kick") -> Streamer:
    return Streamer(
        id="x",
        name=name,
        platform=platform,
        channel=name.lower(),
        status="verified",
        conditions=Conditions(credit_format=credit_format),
    )


# --- the clip window ----------------------------------------------------------


def test_the_window_takes_the_build_up_and_the_tail():
    window = clip_window(peak=100.0, duration=300.0, build_up=35, tail=12, min_total=45, max_total=60)
    assert window.start == pytest.approx(65.0)
    assert window.end == pytest.approx(112.0)
    assert window.duration == pytest.approx(47.0)


def test_the_window_is_inside_the_45_to_60_second_band():
    for peak in (0.0, 5.0, 100.0, 295.0, 300.0):
        window = clip_window(peak, 300.0, 35, 12, 45, 60)
        assert 45.0 <= window.duration <= 60.0


def test_the_window_keeps_the_peak_inside():
    for peak in (0.0, 1.0, 5.0, 150.0, 295.0, 299.9):
        window = clip_window(peak, 300.0, 35, 12, 45, 60)
        assert window.start <= window.peak <= window.end


def test_a_peak_near_the_start_slides_the_window_instead_of_shrinking_it():
    window = clip_window(peak=3.0, duration=300.0, build_up=35, tail=12, min_total=45, max_total=60)
    assert window.start == pytest.approx(0.0)
    assert window.duration == pytest.approx(47.0)
    assert window.peak == pytest.approx(3.0)


def test_a_peak_near_the_end_slides_the_window():
    window = clip_window(peak=298.0, duration=300.0, build_up=35, tail=12, min_total=45, max_total=60)
    assert window.end == pytest.approx(300.0)
    assert window.duration == pytest.approx(47.0)
    assert window.peak == pytest.approx(298.0)


def test_a_file_shorter_than_a_clip_is_used_whole():
    window = clip_window(peak=5.0, duration=20.0, build_up=35, tail=12, min_total=45, max_total=60)
    assert window.start == 0.0
    assert window.end == 20.0


def test_the_window_dict():
    window = ClipWindow(start=1.0, end=48.0, peak=20.0)
    payload = window.as_dict()
    assert payload["start"] == 1.0
    assert payload["duration"] == 47.0


# --- the vertical crop ---------------------------------------------------------


def test_the_crop_of_a_landscape_frame_is_the_centre_9_16():
    # 1920x1080 -> keep the full height, cut the sides to 9:16.
    assert centre_crop(1920, 1080) == "crop=608:1080:656:0"


def test_the_crop_of_the_blade_frame():
    assert centre_crop(480, 832) == "crop=468:832:6:0"


def test_the_crop_of_a_frame_that_is_already_vertical():
    assert centre_crop(1080, 1920) == "crop=1080:1920:0:0"


def test_the_crop_of_a_frame_that_is_too_tall():
    # 1080x2400 -> keep the full width, cut the top and the bottom.
    assert centre_crop(1080, 2400) == "crop=1080:1920:0:240"


def test_the_crop_is_never_larger_than_the_frame():
    for width, height in [(320, 240), (1920, 1080), (480, 832), (1080, 2400), (768, 1344)]:
        crop = centre_crop(width, height)
        parts = crop.split(":")
        crop_width = int(parts[0].split("=")[1])
        crop_height = int(parts[1])
        assert crop_width <= width
        assert crop_height <= height
        # The crop is 9:16 (within a pixel of rounding).
        assert abs(crop_width / crop_height - 9 / 16) < 0.01


# --- the credit line ------------------------------------------------------------


def test_the_default_credit_line():
    assert credit_line(make_streamer(), "x") == "@ExampleStreamer on kick"


def test_the_credit_line_uses_the_streamers_own_format():
    streamer = make_streamer(credit_format="@ExampleStreamer on Kick (clips welcome)")
    assert credit_line(streamer, "x") == "@ExampleStreamer on Kick (clips welcome)"


def test_the_credit_line_for_our_own_footage():
    assert credit_line(None, "self") == "me"


# --- the hooks -------------------------------------------------------------------


def test_the_hooks_use_the_strongest_phrase_and_the_signal_names():
    moment = {
        "mmss": "02:25.2",
        "signals": {
            "loudness": {"jump_lu": 3.9, "value": 0.8},
            "transcript": {"hits": [{"keyword": "no way", "weight": 2.0, "text": "no way"}]},
            "sound": {"classes": [{"label": "Laughter", "score": 0.5}]},
        },
    }
    hooks = build_hooks(moment, "@ExampleStreamer on kick")
    assert len(hooks) == 3
    assert "no way" in hooks[0]
    assert "loudness" in hooks[2]
    assert "Laughter" in hooks[2]


def test_the_hooks_fall_back_when_there_is_no_phrase():
    moment = {"mmss": "00:10.0", "signals": {"loudness": {"jump_lu": 4.0}}}
    hooks = build_hooks(moment, "Example")
    assert len(hooks) == 3
    assert hooks[0].startswith("Wait for this")
    assert "loudness" in hooks[2]


def test_the_strongest_phrase_picks_the_heaviest_hit():
    moment = {
        "signals": {
            "transcript": {
                "hits": [
                    {"keyword": "what", "weight": 0.6, "text": "what"},
                    {"keyword": "oh my god", "weight": 2.5, "text": "oh my god"},
                ]
            }
        }
    }
    assert strongest_phrase(moment) == "oh my god"


# --- the transcript ---------------------------------------------------------------


def test_the_words_are_made_clip_relative():
    words = [
        {"word": "before", "start": 10.0, "end": 10.4},
        {"word": "inside", "start": 12.0, "end": 12.5},
        {"word": "after", "start": 90.0, "end": 90.4},
    ]
    window = ClipWindow(start=10.0, end=60.0, peak=20.0)
    out = clip_relative_words(words, window)
    assert [word["word"] for word in out] == ["before", "inside"]
    assert out[0]["start"] == pytest.approx(0.0)
    assert out[1]["start"] == pytest.approx(2.0)
    assert out[1]["source_start"] == 12.0


# --- names ----------------------------------------------------------------------


def test_the_folder_name_has_no_colon():
    assert safe_name("02:25.2") == "02-25.2"


# --- hooks from the transcript ----------------------------------------------------


def _words(text: str, start: float = 100.0, step: float = 0.4) -> list[dict]:
    return [
        {"word": w, "start": start + i * step, "end": start + i * step + 0.3}
        for i, w in enumerate(text.split())
    ]


def test_the_hooks_come_from_the_words_around_the_peak():
    words = _words("so I was walking. uh no no no way he just did that to me. okay anyway then")
    moment = {"time": 100.0 + 9 * 0.4, "signals": {"loudness": {"jump_lu": 5}}}
    hooks = build_hooks(moment, "Example", words)
    assert hooks[0] == "no way he just did that to me"
    assert hooks[1].startswith('Chat lost it when he said "no way he just')
    assert "loudness" in hooks[2]
    assert all(len(h) <= 60 for h in hooks[:2])


def test_filler_and_repeats_are_cleaned_and_long_quotes_trimmed():
    words = _words("um the the thing is that this whole entire situation has gotten completely out of hand now")
    hooks = build_hooks({"time": 102.0, "signals": {}}, "x", words)
    assert not hooks[0].startswith("um")
    assert "the the" not in hooks[0]
    assert len(hooks[0]) <= 60
    assert len(hooks[1]) <= 60
    assert hooks[1].startswith("Chat lost it when he said")
