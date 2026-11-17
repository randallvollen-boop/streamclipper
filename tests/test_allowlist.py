"""The allow-list is the hard rule.

The rule under test: a streamer may be watched, recorded or clipped ONLY when
the status is `verified` AND at least one proof entry carries both a url and
the date it was checked.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from clipper import allowlist
from clipper.allowlist import AllowlistError, Streamer


def write_allowlist(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "streamers.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def streamer(**overrides) -> Streamer:
    base = {
        "id": "x",
        "name": "X",
        "platform": "kick",
        "channel": "x",
        "status": "verified",
        "proof": (
            allowlist.Proof(url="https://example.test/brief", kind="program-brief", checked="2026-11-16"),
        ),
    }
    base.update(overrides)
    return Streamer(**base)


# --- the rule ---------------------------------------------------------------


def test_verified_with_proof_is_allowed():
    assert allowlist.allowed(streamer()) is True
    assert "verified with proof" in allowlist.reason(streamer())


def test_verified_without_proof_is_refused():
    s = streamer(proof=())
    assert allowlist.allowed(s) is False
    assert "no proof recorded" in allowlist.reason(s)


def test_verified_with_proof_missing_url_is_refused():
    s = streamer(proof=(allowlist.Proof(url="", kind="program-brief", checked="2026-11-16"),))
    assert allowlist.allowed(s) is False


def test_verified_with_proof_missing_date_is_refused():
    s = streamer(proof=(allowlist.Proof(url="https://example.test/brief", kind="program-brief", checked="not-a-date"),))
    assert allowlist.allowed(s) is False
    assert "no proof with both a url and a checked date" in allowlist.reason(s)


@pytest.mark.parametrize("status", ["pending", "unverified", "rejected"])
def test_non_verified_status_is_refused(status):
    s = streamer(status=status)
    assert allowlist.allowed(s) is False
    assert status in allowlist.reason(s)


def test_pending_is_refused_even_with_proof():
    s = streamer(status="pending")
    assert allowlist.allowed(s) is False


# --- schema errors are loud -------------------------------------------------


def test_bad_platform_is_refused_from_a_real_file(tmp_path):
    path = write_allowlist(
        tmp_path,
        """
        streamers:
          - id: x
            name: X
            platform: myspace
            channel: x
            status: verified
            proof: []
        """,
    )
    with pytest.raises(AllowlistError, match="platform 'myspace' is not one of"):
        allowlist.load(path)


def test_bad_status_is_loud(tmp_path):
    path = write_allowlist(
        tmp_path,
        """
        streamers:
          - id: x
            name: X
            platform: kick
            channel: x
            status: maybe
            proof: []
        """,
    )
    with pytest.raises(AllowlistError, match="status 'maybe' is not one of"):
        allowlist.load(path)


def test_missing_key_is_loud(tmp_path):
    path = write_allowlist(
        tmp_path,
        """
        streamers:
          - id: x
            name: X
            platform: kick
            status: verified
            proof: []
        """,
    )
    with pytest.raises(AllowlistError, match="missing required key 'channel'"):
        allowlist.load(path)


def test_bad_proof_kind_is_loud(tmp_path):
    path = write_allowlist(
        tmp_path,
        """
        streamers:
          - id: x
            name: X
            platform: kick
            channel: x
            status: verified
            proof:
              - url: https://example.test/brief
                kind: rumour
                checked: 2026-11-16
        """,
    )
    with pytest.raises(AllowlistError, match="kind 'rumour' is not one of"):
        allowlist.load(path)


def test_bad_checked_date_is_loud(tmp_path):
    path = write_allowlist(
        tmp_path,
        """
        streamers:
          - id: x
            name: X
            platform: kick
            channel: x
            status: verified
            proof:
              - url: https://example.test/brief
                kind: program-brief
                checked: last Tuesday
        """,
    )
    with pytest.raises(AllowlistError, match="is not an ISO date"):
        allowlist.load(path)


def test_duplicate_id_is_loud(tmp_path):
    path = write_allowlist(
        tmp_path,
        """
        streamers:
          - id: x
            name: X
            platform: kick
            channel: x
            status: verified
            proof: []
          - id: x
            name: X again
            platform: kick
            channel: x2
            status: verified
            proof: []
        """,
    )
    with pytest.raises(AllowlistError, match="duplicate streamer id 'x'"):
        allowlist.load(path)


def test_missing_file_is_loud(tmp_path):
    with pytest.raises(AllowlistError, match="cannot read allow-list"):
        allowlist.load(tmp_path / "nope.yaml")


# --- the shipped example allow-list -------------------------------------------

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "config" / "streamers.example.yaml"


def test_shipped_example_allowlist_loads():
    ids = [s.id for s in allowlist.load(EXAMPLE_PATH)]
    assert ids == ["examplestreamer"]


def test_shipped_example_entry_is_unverified_and_not_allowed():
    """The example is fictional: it must never be watched, recorded or clipped."""
    streamers = {s.id: s for s in allowlist.load(EXAMPLE_PATH)}
    example = streamers["examplestreamer"]
    assert example.status == "unverified"
    assert example.platform == "kick"
    assert example.proof == ()
    assert not allowlist.allowed(example)
    assert allowlist.only_allowed(streamers.values()) == []


def test_shipped_example_notes_are_kept():
    streamers = {s.id: s for s in allowlist.load(EXAMPLE_PATH)}
    assert "Fictional" in streamers["examplestreamer"].conditions.notes
