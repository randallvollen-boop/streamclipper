"""The review page.

The page is tested against a temporary data folder with tiny ffmpeg-made
videos: the kits are listed, the video is served with a Range so a phone can
seek, a path that tries to escape the runs folder is refused, and the three
decisions (approve, trim, reject) do what they say.

Nothing here touches the real `data/` folder.
"""
from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipper import audio
from clipper import kit as kit_builder
from clipper import library
from clipper.config import Config
from clipper.review.server import ReviewError, make_server

RUN = "run1"
KIT = "1-00-03.0"
SOURCE_SECONDS = 10.0


def make_video(path: Path, seconds: float = SOURCE_SECONDS) -> Path:
    """A tiny test video: the test pattern plus a tone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.run(
        [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=10:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ]
    )
    return path


@pytest.fixture
def world(tmp_path: Path) -> SimpleNamespace:
    """A data folder with one run and one kit in it."""
    data_dir = tmp_path / "data"
    kit_folder = data_dir / "runs" / RUN / "kits" / KIT
    kit_folder.mkdir(parents=True)
    source = make_video(tmp_path / "source.mp4")

    config = Config(
        raw={
            "data_dir": str(data_dir),
            "handoff_dir": str(tmp_path / "handoff"),
            "account": "default",
            "review_port": 8297,
            "kit": {"build_up_seconds": 35, "tail_seconds": 12},
        },
        path=tmp_path / "clipper.yaml",
    )

    window = kit_builder.ClipWindow(start=1.0, end=6.0, peak=3.0)
    kit_builder.cut_clip(source, window, kit_folder / "clip.mp4", config)
    kit_builder.make_vertical(kit_folder / "clip.mp4", kit_folder / "vertical.mp4", config)
    (kit_folder / "kit.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "run": str(data_dir / "runs" / RUN),
                "streamer_id": "self",
                "streamer_name": "Example",
                "credit": "@Example on our channel",
                "rank": 1,
                "mmss": "00:03.0",
                "times": window.as_dict(),
                "score": 1.25,
                "clip": str(kit_folder / "clip.mp4"),
                "vertical": str(kit_folder / "vertical.mp4"),
                "vertical_crop": {"mode": "centre", "crop_x": 0.0},
                "hooks": [
                    'Wait for this — and then it hits.',
                    "the loudness jumps 4.0 LU.",
                    "@Example on our channel — 00:03.0.",
                ],
                "words": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (kit_folder / "signals.json").write_text(
        json.dumps(
            {
                "loudness": {"lufs": -12.0, "baseline_lufs": -16.0, "jump_lu": 4.0, "value": 0.8},
                "transcript": {"hits": [{"keyword": "what", "weight": 0.6, "at": 3.0, "text": "what"}]},
            }
        ),
        encoding="utf-8",
    )
    (kit_folder / "hooks.txt").write_text(
        "Wait for this — and then it hits.\n", encoding="utf-8"
    )
    (kit_folder / "credit.txt").write_text("@Example on our channel\n", encoding="utf-8")
    return SimpleNamespace(
        data_dir=data_dir,
        config=config,
        kit_folder=kit_folder,
        source=source,
        handoff=tmp_path / "handoff",
    )


@pytest.fixture
def server(world: SimpleNamespace):
    """The review server on an unused port."""
    instance = make_server(world.config, data_dir=world.data_dir, port=0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()


def request(
    server,
    path: str,
    method: str = "GET",
    body: dict | None = None,
    headers: dict | None = None,
) -> tuple[int, dict, bytes]:
    """One request to the server."""
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_address[1], timeout=30
    )
    try:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request_headers = dict(headers or {})
        if payload is not None:
            request_headers.setdefault("Content-Type", "application/json")
        connection.request(method, path, body=payload, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, data
    finally:
        connection.close()


def decisions(world: SimpleNamespace) -> list[dict]:
    path = world.data_dir / "decisions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- the page ----------------------------------------------------------------


def test_the_page_is_served(server):
    status, headers, body = request(server, "/")
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert b"Clip review" in body


def test_the_server_is_bound_to_the_loopback_only(world: SimpleNamespace):
    """The page is for this machine (or a reverse proxy in front of it)."""
    instance = make_server(world.config, data_dir=world.data_dir, port=0)
    try:
        assert instance.server_address[0] == "127.0.0.1"
    finally:
        instance.server_close()
    with pytest.raises(ReviewError):
        make_server(world.config, data_dir=world.data_dir, host="0.0.0.0", port=0)


# --- listing the kits --------------------------------------------------------


def test_the_api_lists_the_kits_with_their_signals_and_hooks(server):
    status, _headers, body = request(server, "/api/kits")
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert len(payload["kits"]) == 1
    kit = payload["kits"][0]
    assert kit["run"] == RUN
    assert kit["kit"] == KIT
    assert kit["status"] == "new"
    assert kit["rank"] == 1
    assert kit["mmss"] == "00:03.0"
    assert len(kit["hooks"]) == 3
    assert kit["credit"] == "@Example on our channel"
    assert "louder by 4.0 LU" in kit["chips"]
    assert kit["media"]["vertical"] == f"/media/{RUN}/{KIT}/vertical.mp4"


# --- the media ---------------------------------------------------------------


def test_the_vertical_is_served(server):
    status, headers, body = request(server, f"/media/{RUN}/{KIT}/vertical.mp4")
    assert status == 200
    assert headers["content-type"] == "video/mp4"
    assert headers["accept-ranges"] == "bytes"
    assert len(body) == int(headers["content-length"])


def test_a_range_request_answers_206(server):
    status, headers, body = request(
        server, f"/media/{RUN}/{KIT}/vertical.mp4", headers={"Range": "bytes=0-99"}
    )
    assert status == 206
    assert len(body) == 100
    assert headers["content-range"].startswith("bytes 0-99/")
    assert headers["content-length"] == "100"


def test_a_range_request_at_the_end_of_the_file(server):
    status, headers, body = request(
        server, f"/media/{RUN}/{KIT}/clip.mp4", headers={"Range": "bytes=-50"}
    )
    assert status == 206
    assert len(body) == 50
    total = int(headers["content-range"].split("/")[1])
    first = int(headers["content-range"].split()[1].split("-")[0])
    assert first == total - 50


def test_a_path_that_tries_to_escape_is_refused(server):
    for path in (
        "/media/../../etc/passwd",
        "/media/..%2f..%2fetc%2fpasswd",
        "/media/%2e%2e/%2e%2e/etc/passwd",
    ):
        status, _headers, _body = request(server, path)
        assert status == 404, path


def test_only_the_two_videos_are_served(server):
    status, _headers, _body = request(server, f"/media/{RUN}/{KIT}/kit.json")
    assert status == 404
    status, _headers, _body = request(server, f"/media/{RUN}/{KIT}/../{KIT}/clip.mp4")
    assert status == 404


# --- the decisions ------------------------------------------------------------


def test_approving_copies_the_clip_and_the_caption_to_the_hand_off(server, world):
    status, _headers, body = request(
        server,
        f"/api/kits/{RUN}/{KIT}/decision",
        method="POST",
        body={"action": "approve", "hook_index": 1},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert payload["kit"]["status"] == "approved"

    folder = world.handoff / "default" / f"{RUN}-{KIT}"
    assert (folder / "short.mp4").exists()
    assert (world.kit_folder / "final" / "short.mp4").exists()
    assert (world.kit_folder / "vertical.mp4").exists()
    caption = (folder / "caption.txt").read_text(encoding="utf-8").splitlines()
    assert caption[0] == "the loudness jumps 4.0 LU."
    assert caption[1] == "@Example on our channel"

    log = decisions(world)
    assert len(log) == 1
    assert log[0]["action"] == "approve"
    assert log[0]["run"] == RUN
    assert log[0]["kit"] == KIT
    assert log[0]["who"] == "local"
    assert log[0]["values"]["hook_index"] == 1
    assert log[0]["time"]


def test_the_decision_log_says_who_approved(server, world):
    request(
        server,
        f"/api/kits/{RUN}/{KIT}/decision",
        method="POST",
        body={"action": "approve"},
        headers={"Tailscale-User-Login": "someone@example.com"},
    )
    log = decisions(world)
    assert log[0]["who"] == "someone@example.com"


def test_rejecting_marks_the_kit(server, world):
    status, _headers, body = request(
        server,
        f"/api/kits/{RUN}/{KIT}/decision",
        method="POST",
        body={"action": "reject", "reason": "not the moment"},
    )
    assert status == 200
    assert json.loads(body)["kit"]["status"] == "rejected"
    log = decisions(world)
    assert log[0]["action"] == "reject"
    assert log[0]["values"]["reason"] == "not the moment"
    # Nothing is deleted.
    assert (world.kit_folder / "clip.mp4").exists()
    assert (world.kit_folder / "vertical.mp4").exists()


def test_trimming_recuts_the_clip_and_keeps_the_old_one(server, world):
    before = audio.probe_duration(world.kit_folder / "clip.mp4")
    status, _headers, body = request(
        server,
        f"/api/kits/{RUN}/{KIT}/decision",
        method="POST",
        body={"action": "trim", "in_s": 2.0, "out_s": 4.5},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True

    after = audio.probe_duration(world.kit_folder / "clip.mp4")
    assert after == pytest.approx(2.5, abs=0.3)
    assert after < before
    vertical = audio.probe_duration(world.kit_folder / "vertical.mp4")
    assert vertical == pytest.approx(2.5, abs=0.3)

    # The old files are kept, not deleted.
    assert (world.kit_folder / "prev" / "clip.mp4").exists()
    assert (world.kit_folder / "prev" / "vertical.mp4").exists()

    # The kit knows its new window.
    kit = json.loads((world.kit_folder / "kit.json").read_text(encoding="utf-8"))
    assert kit["times"]["start"] == pytest.approx(2.0, abs=0.05)
    assert kit["times"]["end"] == pytest.approx(4.5, abs=0.05)

    log = decisions(world)
    assert log[0]["action"] == "trim"
    assert log[0]["values"]["in_s"] == pytest.approx(2.0)
    assert log[0]["values"]["out_s"] == pytest.approx(4.5)


def test_a_trim_outside_the_source_is_refused(server, world):
    status, _headers, body = request(
        server,
        f"/api/kits/{RUN}/{KIT}/decision",
        method="POST",
        body={"action": "trim", "in_s": 2.0, "out_s": 900.0},
    )
    assert status == 400
    assert "past the end" in json.loads(body)["error"]
    # The clip is untouched.
    assert audio.probe_duration(world.kit_folder / "clip.mp4") == pytest.approx(5.0, abs=0.3)


def test_the_crop_slider_moves_the_picture(server, world):
    status, _headers, body = request(
        server,
        f"/api/kits/{RUN}/{KIT}/decision",
        method="POST",
        body={"action": "crop", "crop_x": 1.0},
    )
    assert status == 200
    kit = json.loads((world.kit_folder / "kit.json").read_text(encoding="utf-8"))
    assert kit["vertical_crop"]["crop_x"] == 1.0
    assert kit["vertical_crop"]["mode"] == "offset"
    # The old vertical is kept.
    assert (world.kit_folder / "prev" / "vertical.mp4").exists()
    log = decisions(world)
    assert log[0]["action"] == "crop"


def test_an_unknown_action_is_refused(server):
    status, _headers, body = request(
        server,
        f"/api/kits/{RUN}/{KIT}/decision",
        method="POST",
        body={"action": "publish"},
    )
    assert status == 400
    assert "action must be one of" in json.loads(body)["error"]


def test_a_decision_about_a_kit_that_is_not_there_is_refused(server):
    status, _headers, _body = request(
        server,
        f"/api/kits/{RUN}/nope/decision",
        method="POST",
        body={"action": "approve"},
    )
    assert status == 400


def test_the_latest_decision_wins(world):
    library.approve_kit(world.data_dir, RUN, KIT, 0, world.config, who="local")
    library.reject_kit(world.data_dir, RUN, KIT, "changed my mind", world.config, who="local")
    kits = library.list_kits(world.data_dir)
    assert kits[0]["status"] == "rejected"
    library.approve_kit(world.data_dir, RUN, KIT, 0, world.config, who="local")
    kits = library.list_kits(world.data_dir)
    assert kits[0]["status"] == "approved"


def test_the_handoff_list_shows_the_approved_clip(world):
    assert library.handoff_list(world.config) == []
    library.approve_kit(world.data_dir, RUN, KIT, 0, world.config, who="local")
    handoff = library.handoff_list(world.config)
    assert len(handoff) == 1
    assert handoff[0]["account"] == "default"
    assert handoff[0]["folder"] == f"{RUN}-{KIT}"
    assert [entry["name"] for entry in handoff[0]["files"]] == ["caption.txt", "short.mp4"]
