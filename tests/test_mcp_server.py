"""The MCP server.

Every tool is called directly (they are the same functions the CLI and the
review page use), and one handshake over stdio proves that the server
answers and lists its tools.

Nothing here touches the real `data/` folder: the config is pointed at a
temporary one.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

from clipper import allowlist
from clipper import audio
from clipper import library
from clipper import mcp_server
from clipper.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = "mcp-run"
KIT = "1-00-10.2"


def make_video(path: Path, seconds: float = 12.0) -> Path:
    """A quiet test video that gets loud after ten seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.run(
        [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=10:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-af", "volume='if(lt(t,10),0.05,1.0)':eval=frame",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        ]
    )
    return path


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A temporary data folder, with the config pointed at it."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # The source lives under `data/`, so it is our own material.
    source = make_video(data_dir / "source.mp4")
    # And one that does not, so the refusal can be tested.
    elsewhere = make_video(tmp_path / "elsewhere.mp4")
    config = Config(
        raw={
            "data_dir": str(data_dir),
            "handoff_dir": str(tmp_path / "handoff"),
            "account": "default",
            "review_port": 8297,
            "listen": {"threshold": 1.0, "merge_seconds": 20.0},
            "kit": {"build_up_seconds": 35, "tail_seconds": 12},
        },
        path=tmp_path / "clipper.yaml",
    )
    monkeypatch.setattr(mcp_server, "load_config", lambda *a, **k: config)
    return SimpleNamespace(
        data_dir=data_dir,
        config=config,
        source=source,
        elsewhere=elsewhere,
        handoff=tmp_path / "handoff",
    )


def build_one_kit(world: SimpleNamespace) -> None:
    """A run with one kit in it, made the way the CLI makes it."""
    result = mcp_server.listen(
        str(world.source), run_name=RUN, top=5, streamer=None
    )
    assert result["moments"], "the test video should have a moment in it"
    kits = mcp_server.make_kits(run_name=RUN, streamer="self")
    assert kits["kits"]


# --- the tools ---------------------------------------------------------------


ALLOWLIST_YAML = """\
streamers:
  - id: examplestreamer
    name: ExampleStreamer
    platform: kick
    channel: examplestreamer
    status: verified
    proof:
      - url: https://example.test/clipping-policy
        kind: program-brief
        checked: 2026-01-01
    conditions:
      credit_format: "@ExampleStreamer on Kick"
  - id: exampletube
    name: ExampleTube
    platform: youtube
    channel: "@ExampleTube"
    status: unverified
    proof: []
    conditions:
      notes: "No public clipping policy found"
"""


@pytest.fixture(autouse=True)
def allowlist_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fictional allow-list: one verified streamer with proof, one unverified."""
    path = tmp_path / "streamers.yaml"
    path.write_text(ALLOWLIST_YAML, encoding="utf-8")
    monkeypatch.setattr(allowlist, "STREAMERS_PATH", path)
    return path


def test_list_streamers_says_who_is_allowed(allowlist_file):
    result = mcp_server.list_streamers()
    ids = [streamer["id"] for streamer in result["streamers"]]
    assert "examplestreamer" in ids
    assert "exampletube" in ids
    examplestreamer = next(s for s in result["streamers"] if s["id"] == "examplestreamer")
    assert examplestreamer["allowed"] is True
    assert examplestreamer["credit"]
    exampletube = next(s for s in result["streamers"] if s["id"] == "exampletube")
    assert exampletube["allowed"] is False
    assert exampletube["why"]


def test_listen_refuses_a_path_that_is_not_our_own_material(world: SimpleNamespace):
    with pytest.raises(mcp_server.McpError) as excinfo:
        mcp_server.listen("/etc/hostname", run_name="nope")
    assert "refused" in str(excinfo.value)


def test_listen_refuses_a_streamer_who_is_not_allowed(world: SimpleNamespace):
    with pytest.raises(mcp_server.McpError) as excinfo:
        mcp_server.listen(str(world.elsewhere), run_name="nope", streamer="exampletube")
    assert "not allowed" in str(excinfo.value)


def test_listen_refuses_an_unknown_streamer(world: SimpleNamespace):
    with pytest.raises(mcp_server.McpError) as excinfo:
        mcp_server.listen(str(world.elsewhere), run_name="nope", streamer="nobody")
    assert "no streamer with id" in str(excinfo.value)


def test_listen_ranks_the_moments_of_our_own_material(world: SimpleNamespace):
    result = mcp_server.listen(str(world.source), run_name=RUN, top=5)
    assert result["run"] == RUN
    assert result["moments"]
    moment = result["moments"][0]
    assert moment["rank"] == 1
    assert moment["score"] > 0
    assert isinstance(moment["signals"], list)
    # The result is small: no media bytes, just numbers and paths.
    assert len(json.dumps(result)) < 4000


def test_make_kits_cuts_the_kits(world: SimpleNamespace):
    mcp_server.listen(str(world.source), run_name=RUN, top=5)
    result = mcp_server.make_kits(run_name=RUN, streamer="self")
    assert result["streamer"] == "self"
    assert result["kits"]
    kit = result["kits"][0]
    assert kit["rank"] == 1
    assert len(kit["hooks"]) == 3
    assert kit["credit"]
    folder = Path(kit["folder"])
    assert (folder / "clip.mp4").exists()
    assert (folder / "vertical.mp4").exists()


def test_make_kits_refuses_a_streamer_who_is_not_allowed(world: SimpleNamespace):
    mcp_server.listen(str(world.source), run_name=RUN, top=5)
    with pytest.raises(mcp_server.McpError) as excinfo:
        mcp_server.make_kits(run_name=RUN, streamer="exampletube")
    assert "may not be clipped" in str(excinfo.value)


def test_make_kits_refuses_an_unknown_streamer(world: SimpleNamespace):
    mcp_server.listen(str(world.source), run_name=RUN, top=5)
    with pytest.raises(mcp_server.McpError):
        mcp_server.make_kits(run_name=RUN, streamer="nobody")


def test_make_kits_needs_a_run_with_moments(world: SimpleNamespace):
    with pytest.raises(mcp_server.McpError) as excinfo:
        mcp_server.make_kits(run_name="nothing-here", streamer="self")
    assert "no moments" in str(excinfo.value)


def test_list_kits_and_get_kit(world: SimpleNamespace):
    build_one_kit(world)
    kits = mcp_server.list_kits()
    assert kits["kits"]
    kit = kits["kits"][0]
    assert kit["status"] == "new"
    assert len(kit["hooks"]) == 3
    assert kit["chips"]
    assert "vertical.mp4" in kit["vertical"]

    one = mcp_server.get_kit(run=kit["run"], kit=kit["kit"])
    assert one["kit"]["run"] == kit["run"]
    assert one["kit"]["kit"] == kit["kit"]


def test_list_kits_can_filter_by_status(world: SimpleNamespace):
    build_one_kit(world)
    assert len(mcp_server.list_kits(status="new")["kits"]) == 1
    assert mcp_server.list_kits(status="approved")["kits"] == []


def test_approve_puts_the_clip_in_the_handoff_folder(world: SimpleNamespace):
    build_one_kit(world)
    kit = mcp_server.list_kits()["kits"][0]
    result = mcp_server.approve(run=kit["run"], kit=kit["kit"], hook_index=1)
    assert result["status"] == "approved"
    folder = Path(result["handoff"])
    assert (folder / "short.mp4").exists()
    caption = (folder / "caption.txt").read_text(encoding="utf-8").splitlines()
    assert caption[0] == kit["hooks"][1]
    assert caption[1] == kit["credit"]

    handoff = mcp_server.handoff_list()["handoff"]
    assert len(handoff) == 1
    assert handoff[0]["account"] == "default"


def test_reject_marks_the_kit(world: SimpleNamespace):
    build_one_kit(world)
    kit = mcp_server.list_kits()["kits"][0]
    result = mcp_server.reject(run=kit["run"], kit=kit["kit"], reason="not the bit")
    assert result["status"] == "rejected"
    assert result["reason"] == "not the bit"
    assert mcp_server.list_kits(status="rejected")["kits"]


def test_trim_recuts_the_clip(world: SimpleNamespace):
    build_one_kit(world)
    kit = mcp_server.list_kits()["kits"][0]
    folder = Path(kit["clip"]).parent
    before = audio.probe_duration(folder / "clip.mp4")
    result = mcp_server.trim(run=kit["run"], kit=kit["kit"], in_s=1.0, out_s=4.0)
    assert result["times"]["start"] == pytest.approx(1.0, abs=0.05)
    assert result["times"]["end"] == pytest.approx(4.0, abs=0.05)
    after = audio.probe_duration(folder / "clip.mp4")
    assert after == pytest.approx(3.0, abs=0.3)
    assert after < before
    assert (folder / "prev" / "clip.mp4").exists()


def test_the_decisions_are_logged_with_the_mcp_as_the_who(world: SimpleNamespace):
    build_one_kit(world)
    kit = mcp_server.list_kits()["kits"][0]
    mcp_server.approve(run=kit["run"], kit=kit["kit"])
    log = library.read_decisions(world.data_dir)
    assert log[-1]["action"] == "approve"
    assert log[-1]["who"] == "mcp"


# --- the handshake ------------------------------------------------------------


def test_the_stdio_server_lists_its_tools(allowlist_file):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def handshake() -> tuple[list[str], str]:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "clipper.mcp_server"],
            cwd=str(REPO_ROOT),
            env={**os.environ, "STREAMCLIPPER_ALLOWLIST": str(allowlist_file)},
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = [tool.name for tool in listed.tools]
                # And one tool over the wire, not just the list.
                called = await session.call_tool("list_streamers", {})
                return names, called.content[0].text

    names, payload = anyio.run(handshake)
    assert sorted(names) == sorted(
        [
            "approve",
            "get_kit",
            "kit_emotion",
            "list_memes",
            "handoff_list",
            "list_kits",
            "list_streamers",
            "listen",
            "make_kits",
            "reject",
            "trim",
        ]
    )
    streamers = json.loads(payload)["streamers"]
    assert "examplestreamer" in [streamer["id"] for streamer in streamers]
