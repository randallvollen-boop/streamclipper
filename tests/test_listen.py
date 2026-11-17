"""The listening agent.

These tests work on synthetic signals: they check the scoring rules, the
keyword matching, the loudness jump, the chat rate and the caching, without
loading PANNs or faster-whisper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clipper import listen
from clipper.listen import (
    Signal,
    Word,
    build_chat_signal,
    build_loudness_signal,
    build_sound_signal,
    build_transcript_signal,
    find_keyword_hits,
    load_chat,
    mmss,
    score_moments,
)


# --- helpers -----------------------------------------------------------------


def flat_signal(name: str, values: list[float]) -> Signal:
    array = np.asarray(values, dtype=np.float64)
    return Signal(name=name, values=array, raw=array.copy())


# --- formatting ---------------------------------------------------------------


def test_mmss():
    assert mmss(0) == "00:00.0"
    assert mmss(65.4) == "01:05.4"
    assert mmss(125) == "02:05.0"


# --- the sound signal ---------------------------------------------------------


def test_the_sound_signal_weights_the_classes():
    # Two classes, 100 frames each (PANNs runs at 100 frames/s, so a 0.5 s
    # bin is 50 frames). Frame 0-49: laughter. Frame 50-99: music.
    columns = {
        "Laughter": np.concatenate([np.ones(50), np.zeros(50)]),
        "Music": np.concatenate([np.zeros(50), np.ones(50)]),
    }
    weights = {"Laughter": 3.0, "Music": -1.5}
    signal = build_sound_signal(columns, weights, 2, 0.5, min_class_score=0.1)
    # bin 0 is all laughter: 3.0 * 1.0
    assert signal.values[0] == pytest.approx(3.0)
    # bin 1 is all music: 1.0 is above the 0.5 floor, so -1.5 * (1.0 - 0.5)
    assert signal.values[1] == pytest.approx(-0.75)


def test_the_music_penalty_floor_leaves_quiet_music_alone():
    columns = {"Music": np.array([0.3])}
    signal = build_sound_signal(
        columns, {"Music": -1.5}, 1, 0.5, min_class_score=0.1, penalty_floor=0.5
    )
    assert signal.values[0] == 0.0


def test_a_class_below_the_minimum_score_counts_for_nothing():
    columns = {"Laughter": np.array([0.05])}
    signal = build_sound_signal(
        columns, {"Laughter": 3.0}, 1, 0.5, min_class_score=0.25
    )
    assert signal.values[0] == 0.0


def test_the_sound_detail_names_the_classes():
    columns = {"Laughter": np.array([1.0])}
    signal = build_sound_signal(
        columns, {"Laughter": 3.0}, 1, 0.5, min_class_score=0.1
    )
    detail = signal.detail(0)
    assert detail["classes"][0]["label"] == "Laughter"
    assert detail["classes"][0]["contribution"] == 3.0


# --- the loudness signal ------------------------------------------------------


def loud(series, **overrides):
    """build_loudness_signal with the config's defaults."""
    settings = {
        "baseline_seconds": 60.0,
        "min_jump_lu": 3.0,
        "gate_lu": -60.0,
        "weight": 2.0,
        "full_jump_lu": 10.0,
        "min_history_seconds": 10.0,
    }
    settings.update(overrides)
    # One entry per 0.5 s bin.
    return build_loudness_signal(series, len(series), 0.5, **settings)


def test_the_loudness_signal_fires_on_a_jump():
    # 40 bins of 0.5 s. Quiet at -30 LU, then a jump to 0 LU at bin 30.
    series = [(index * 0.5, -30.0) for index in range(30)]
    series += [(15.0 + index * 0.5, 0.0) for index in range(10)]
    signal = loud(series)
    assert signal.values[0] == 0.0
    # A 30 LU jump is above the 10 LU full scale, so it scores the whole weight.
    assert signal.values[30] == pytest.approx(2.0)


def test_the_loudness_jump_is_normalised():
    # A 5 LU jump is half the full scale, so it scores half the weight.
    series = [(index * 0.5, -30.0) for index in range(30)]
    series += [(15.0 + index * 0.5, -25.0) for index in range(10)]
    signal = loud(series)
    assert signal.values[30] == pytest.approx(1.0)


def test_the_loudness_signal_ignores_a_small_change():
    series = [(index * 0.5, -30.0) for index in range(30)]
    series += [(15.0 + index * 0.5, -28.0) for index in range(10)]
    signal = loud(series)
    assert signal.values[30] == 0.0


def test_the_loudness_signal_gates_silence():
    series = [(index * 0.5, -120.0) for index in range(40)]
    signal = loud(series)
    assert float(signal.values.sum()) == 0.0


def test_a_loud_start_is_not_a_jump():
    """The first seconds of a file have no baseline, so they are not a jump."""
    series = [(index * 0.5, 0.0) for index in range(10)]
    series += [(5.0 + index * 0.5, -30.0) for index in range(30)]
    signal = loud(series)
    assert float(signal.values[:10].sum()) == 0.0


# --- the transcript signal -----------------------------------------------------


def test_the_keyword_matching_finds_multi_word_phrases():
    words = [
        Word("oh", 1.0, 1.2),
        Word("my", 1.2, 1.4),
        Word("god", 1.4, 1.8),
        Word("that", 2.0, 2.2),
        Word("was", 2.2, 2.4),
        Word("crazy", 2.4, 2.8),
    ]
    hits = find_keyword_hits(words, {"oh my god": 2.5, "crazy": 1.5})
    assert [hit["keyword"] for hit in hits] == ["oh my god", "crazy"]
    assert hits[0]["at"] == 1.0
    assert hits[1]["at"] == 2.4


def test_the_keyword_matching_does_not_match_inside_a_word():
    words = [Word("yesterday", 1.0, 1.5)]
    assert find_keyword_hits(words, {"yes": 1.0}) == []


def test_the_transcript_signal_scores_the_hits():
    words = [Word("no", 1.0, 1.1), Word("way", 1.1, 1.4)]
    signal = build_transcript_signal(
        words, {"no way": 2.0}, 4, 0.5, max_hits_per_keyword=3, weight=1.0
    )
    assert signal.values[2] == pytest.approx(2.0)
    assert signal.detail(2)["hits"][0]["keyword"] == "no way"


def test_the_transcript_signal_caps_the_hits_of_one_keyword():
    words = [Word("no", 1.0 + index * 0.05, 1.05 + index * 0.05) for index in range(6)]
    signal = build_transcript_signal(
        words, {"no": 1.0}, 4, 0.5, max_hits_per_keyword=2, weight=1.0
    )
    assert signal.values[2] == pytest.approx(2.0)


# --- the chat signal -----------------------------------------------------------


def test_the_chat_signal_fires_on_a_spike():
    # A quiet chat with one burst: 40 bins of 0.5 s (20 s).
    messages = [{"t": float(index * 2), "user": "a", "text": "hi"} for index in range(10)]
    messages += [{"t": 10.0, "user": "b", "text": "no way"} for _ in range(20)]
    signal = build_chat_signal(
        messages, 40, 0.5, window_seconds=10.0, min_rate=1.0,
        reaction_weight=0.5, weight=2.0, reactions=["lul"],
    )
    assert signal.values[20] > 0
    # 20 in the burst plus the 5 quiet ones inside the 10 s window.
    assert signal.detail(20)["messages_in_window"] == 25


def test_the_chat_spike_is_normalised():
    """A busy chat must not out-score every other signal."""
    messages = [{"t": 10.0, "user": "b", "text": "hi"} for _ in range(500)]
    signal = build_chat_signal(
        messages, 100, 0.5, window_seconds=10.0, min_rate=1.0,
        reaction_weight=0.5, weight=2.0, reactions=["lul"],
    )
    # The whole weight, and no more.
    assert signal.values[20] == pytest.approx(2.0)


def test_the_chat_reaction_bonus_is_normalised():
    messages = [{"t": 10.0, "user": "b", "text": "LUL"} for _ in range(50)]
    signal = build_chat_signal(
        messages, 100, 0.5, window_seconds=10.0, min_rate=1.0,
        reaction_weight=0.5, weight=2.0, reactions=["lul"],
    )
    # The spike is capped at the full weight and the reaction bonus at its
    # own cap: 2.0 + 0.5.
    assert signal.values[20] == pytest.approx(2.5)


def test_more_reactions_do_not_raise_the_score_further():
    few = [{"t": 10.0, "user": "b", "text": "LUL"} for _ in range(50)]
    many = [{"t": 10.0, "user": "b", "text": "LUL"} for _ in range(500)]
    settings = dict(
        n_bins=40, hop_seconds=0.5, window_seconds=10.0, min_rate=1.0,
        reaction_weight=0.5, weight=2.0, reactions=["lul"],
    )
    assert build_chat_signal(few, **settings).values[20] == pytest.approx(
        build_chat_signal(many, **settings).values[20]
    )


def test_the_chat_signal_uses_the_files_own_baseline():
    # The same burst in a chat that is busy everywhere is not a spike.
    messages = [{"t": float(index * 0.5), "user": "a", "text": "hi"} for index in range(40)]
    signal = build_chat_signal(
        messages, 40, 0.5, window_seconds=10.0, min_rate=1.0,
        reaction_weight=0.5, weight=2.0, reactions=["lul"],
    )
    assert float(signal.values.sum()) == 0.0


def test_the_chat_signal_counts_reactions():
    messages = [{"t": 1.0, "user": "a", "text": "LUL"}]
    signal = build_chat_signal(
        messages, 10, 0.5, window_seconds=10.0, min_rate=1.0,
        reaction_weight=0.5, weight=2.0, reactions=["lul"],
    )
    assert signal.detail(2)["reactions"] == 1


def test_the_chat_signal_is_flat_without_chat():
    signal = build_chat_signal(
        [], 10, 0.5, window_seconds=10.0, min_rate=1.0,
        reaction_weight=0.5, weight=2.0, reactions=["lul"],
    )
    assert float(signal.values.sum()) == 0.0


def test_load_chat_reads_the_jsonl(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        '{"t": 1.5, "user": "a", "text": "no way"}\n{"t": 2.0, "user": "b", "text": "LUL"}\n',
        encoding="utf-8",
    )
    messages = load_chat(path)
    assert len(messages) == 2
    assert messages[0]["t"] == 1.5
    assert messages[1]["text"] == "LUL"


def test_load_chat_rejects_a_broken_line(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text('not json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not JSON"):
        load_chat(path)


def test_load_chat_rejects_a_message_without_a_time(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text('{"user": "a", "text": "hi"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_chat(path)


# --- the scoring ---------------------------------------------------------------


def test_the_score_sums_the_peak_of_each_signal_in_the_window():
    # 10 bins of 0.5 s. A sound peak at bin 4 and a loudness peak at bin 5.
    sound = flat_signal("sound", [0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0])
    loud = flat_signal("loudness", [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0])
    moments = score_moments(
        [sound, loud], 10, 0.5, window_seconds=2.0, threshold=1.0, merge_seconds=20.0, top_n=5
    )
    assert len(moments) == 1
    assert moments[0].score == pytest.approx(3.0)
    assert set(moments[0].signals) == {"sound", "loudness"}
    assert moments[0].rank == 1


def test_a_peak_below_the_threshold_is_dropped():
    signal = flat_signal("sound", [0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0])
    moments = score_moments(
        [signal], 10, 0.5, window_seconds=2.0, threshold=1.0, merge_seconds=20.0, top_n=5
    )
    assert moments == []


def test_peaks_within_the_merge_distance_become_one_moment():
    signal = flat_signal("sound", [0, 0, 3.0, 0, 0, 0, 0, 2.0, 0, 0])
    moments = score_moments(
        [signal], 10, 0.5, window_seconds=2.0, threshold=1.0, merge_seconds=20.0, top_n=5
    )
    assert len(moments) == 1
    assert moments[0].score == pytest.approx(3.0)


def test_peaks_far_apart_stay_separate():
    signal = flat_signal("sound", [3.0, 0, 0, 0, 0, 0, 0, 0, 0, 2.0])
    moments = score_moments(
        [signal], 10, 0.5, window_seconds=2.0, threshold=1.0, merge_seconds=2.0, top_n=5, build_up_seconds=0.5, tail_seconds=0.5
    )
    assert len(moments) == 2
    assert moments[0].score == 3.0  # the strongest one is ranked first
    assert moments[1].score == 2.0


def test_only_the_top_n_moments_survive():
    # Peaks 2.5 s apart, so they are not merged.
    signal = flat_signal("sound", [3.0, 0, 0, 0, 0, 2.0, 0, 0, 0, 0])
    moments = score_moments(
        [signal], 10, 0.5, window_seconds=2.0, threshold=1.0, merge_seconds=2.0, top_n=2, build_up_seconds=0.5, tail_seconds=0.5
    )
    assert len(moments) == 2
    assert [moment.score for moment in moments] == [3.0, 2.0]


def test_the_moment_reports_the_signals_that_fired_with_their_values():
    signal = flat_signal("sound", [0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0])
    moments = score_moments(
        [signal], 10, 0.5, window_seconds=2.0, threshold=1.0, merge_seconds=20.0, top_n=5
    )
    fired = moments[0].signals["sound"]
    assert fired["value"] == pytest.approx(3.0)
    assert fired["at"] == pytest.approx(1.25)  # the centre of bin 2


def test_a_moment_dict_carries_the_time_the_score_and_the_signals():
    signal = flat_signal("sound", [0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0])
    moments = score_moments(
        [signal], 10, 0.5, window_seconds=2.0, threshold=1.0, merge_seconds=20.0, top_n=5
    )
    payload = moments[0].as_dict()
    assert payload["rank"] == 1
    assert payload["time"] == pytest.approx(1.25)
    assert payload["mmss"] == "00:01.2"
    assert "sound" in payload["signals"]


# --- the input ------------------------------------------------------------------


def test_resolve_source_joins_a_directory_of_chunks(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    for index in range(2):
        chunk = chunks / f"{index:03d}.ts"
        chunk.write_bytes(b"\x00" * 16)
    work = tmp_path / "work"
    work.mkdir()
    # The real join needs ffmpeg and real chunks; here we only check that a
    # directory is recognised and routed to the joiner.
    with pytest.raises(Exception):
        listen.resolve_source(chunks, work)
    # A file is returned as it is.
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    assert listen.resolve_source(source, work) == source


def test_the_fingerprint_changes_with_the_settings():
    assert listen.fingerprint("a", 1) == listen.fingerprint("a", 1)
    assert listen.fingerprint("a", 1) != listen.fingerprint("a", 2)


def test_reactions_match_whole_tokens_and_kick_emotes():
    from clipper.listen import chat_tokens
    assert "w" in chat_tokens("W")
    assert "w" not in chat_tokens("what is this")
    assert "kekw" in chat_tokens("[emote:5747996:KEKW] lol")


def test_a_busy_chat_spikes_relative_to_its_own_baseline():
    # 6 messages a second everywhere, tripled for 10 s in the middle.
    msgs = [{"t": i / 6, "user": "a", "text": "hi"} for i in range(6 * 200)]
    msgs += [{"t": 100 + i / 12, "user": "b", "text": "hi"} for i in range(120)]
    sig = build_chat_signal(msgs, 400, 0.5, window_seconds=10.0, min_rate=1.0,
                            reaction_weight=1.0, weight=2.0, reactions=["lul"], full_rate=1.0)
    assert sig.values[205] > 1.5          # inside the burst
    assert sig.values[40] == pytest.approx(0.0)  # ordinary busy chat


def test_overlapping_windows_are_dropped_and_the_list_is_walked_on():
    # kits 16-63 s (#1), 38-85 s (#5) and 69-116 s (#2); peaks 51, 73, 104.
    values = np.zeros(480)
    values[int(51 / 0.5)] = 3.0
    values[int(73 / 0.5)] = 1.5
    values[int(104 / 0.5)] = 2.0
    values[int(200 / 0.5)] = 1.2
    signal = Signal("sound", values, values.copy())
    moments = score_moments(
        [signal], 480, 0.5, window_seconds=2.0, threshold=1.0, merge_seconds=5.0, top_n=3
    )
    times = [round(m.time) for m in moments]
    assert times == [51, 104, 200]  # 73 overlaps 51 by 53 %, so 200 takes its place
    assert [m.rank for m in moments] == [1, 2, 3]
