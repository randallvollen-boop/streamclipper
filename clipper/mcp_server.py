"""The MCP server: the same clipper, over the Model Context Protocol.

Eleven tools, all thin wrappers over the functions the CLI and the review page
already use (`clipper.library`, `clipper.listen`, `clipper.kit`,
`clipper.allowlist`). No rule lives here that does not live somewhere else.

    python -m clipper.mcp_server      # the stdio transport

Every result is small JSON — paths, times, scores, status. No media bytes
come back through the protocol.

Two gates are stricter than the CLI on purpose, because an agent drives this
surface and a person does not:

  * `listen` only takes our own material (`data/`, `~/own-uploads`)
    unless it names a streamer who is on the allow-list and allowed.
  * `make_kits` only takes an allow-listed streamer whose status is
    `verified` (or `self`, our own footage).
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import allowlist
from . import kit as kit_builder
from . import library
from . import meme as meme_outro
from . import listen as listen_agent
from .config import Config, load as load_config

#: Our own long videos, where the listening agent is developed and tested.
OWN_UPLOADS = Path.home() / "own-uploads"

#: The kit builder's own id for our own footage.
SELF_ID = kit_builder.SELF_ID


class McpError(Exception):
    """The tool cannot do what it was asked. The message says why."""


@contextlib.contextmanager
def _stdout_to_stderr():
    """Keep stdout for the protocol.

    The detector prints a line or two when it loads its weights; over stdio
    that would land in the middle of the protocol stream.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _config() -> Config:
    return load_config()


def _run_dir(config: Config, run_name: str) -> Path:
    try:
        name = library.safe_name(run_name)
    except library.LibraryError as exc:
        raise McpError(str(exc)) from exc
    return config.data_dir / "runs" / name


def _under(path: Path, root: Path) -> bool:
    """True when `path` is inside `root`."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def _streamer_by_id(streamer_id: str, file: str | None = None) -> allowlist.Streamer:
    """The allow-list entry, or a refusal."""
    try:
        streamers = allowlist.load(file)
    except allowlist.AllowlistError as exc:
        raise McpError(f"allow-list error: {exc}") from exc
    match = next((s for s in streamers if s.id == streamer_id), None)
    if match is None:
        raise McpError(
            f"no streamer with id {streamer_id!r} on the allow-list — "
            "an unknown streamer is never used"
        )
    return match


def _kit_summary(kit: dict[str, Any]) -> dict[str, Any]:
    """A kit without the bulky parts: the MCP answers stay small."""
    return {
        "run": kit["run"],
        "kit": kit["kit"],
        "rank": kit["rank"],
        "peak": kit["peak"],
        "mmss": kit["mmss"],
        "score": kit["score"],
        "times": kit["times"],
        "status": kit["status"],
        "chips": kit["chips"],
        "hooks": kit["hooks"],
        "credit": kit["credit"],
        "streamer": kit["streamer"],
        "clip": str(Path(kit["folder"]) / "clip.mp4"),
        "vertical": str(Path(kit["folder"]) / "vertical.mp4"),
    }


server = MCPServer(
    name="streamclipper",
    version="0.1.0",
    instructions=(
        "StreamClipper: find the moments people react to in a streamer's video "
        "and cut them into remix kits. A person approves every clip before it "
        "goes anywhere. Only allow-listed streamers may be used."
    ),
)


@server.tool()
def list_streamers() -> dict[str, Any]:
    """The allow-list: who may be watched, recorded or clipped, and why."""
    try:
        streamers = allowlist.load()
    except allowlist.AllowlistError as exc:
        raise McpError(f"allow-list error: {exc}") from exc
    return {
        "streamers": [
            {
                "id": streamer.id,
                "name": streamer.name,
                "platform": streamer.platform,
                "channel": streamer.channel,
                "status": streamer.status,
                "allowed": allowlist.allowed(streamer),
                "why": allowlist.reason(streamer),
                "credit": streamer.conditions.credit_format,
                "notes": streamer.conditions.notes,
            }
            for streamer in streamers
        ]
    }


@server.tool()
def listen(video_path: str, run_name: str, top: int = 5, streamer: str | None = None) -> dict[str, Any]:
    """Rank the moments in a video and write them to a run folder.

    `video_path` is our own material: a file under `data/` or under
    `~/own-uploads`. Any other path needs `streamer` to name a streamer
    who is on the allow-list and allowed.
    """
    config = _config()
    source = Path(video_path).expanduser()
    if not source.exists():
        candidate = config.data_dir.parent / source
        if candidate.exists():
            source = candidate
        else:
            raise McpError(f"there is no video at {video_path}")
    if not (source.is_file() or source.is_dir()):
        raise McpError(f"{source} is not a video")

    if not (_under(source, config.data_dir) or _under(source, OWN_UPLOADS)):
        if not streamer:
            raise McpError(
                f"refused: {source} is neither under {config.data_dir} nor under "
                f"{OWN_UPLOADS}; name an allowed streamer to use it"
            )
        entry = _streamer_by_id(streamer)
        if not allowlist.allowed(entry):
            raise McpError(
                f"refused: {entry.id} ({entry.name}) is not allowed: "
                f"{allowlist.reason(entry)}"
            )

    run_dir = _run_dir(config, run_name)
    with _stdout_to_stderr():
        payload = listen_agent.analyse(
            source,
            run_dir,
            config,
            top_n=int(top),
        )
    return {
        "run": run_name,
        "source": payload["source"],
        "duration": payload["duration"],
        "moments": [
            {
                "rank": moment["rank"],
                "mmss": moment["mmss"],
                "time": moment["time"],
                "score": moment["score"],
                "signals": sorted(moment["signals"]),
            }
            for moment in payload["moments"]
        ],
        "run_dir": str(run_dir),
        "timings": payload["timings"],
    }


@server.tool()
def make_kits(run_name: str, streamer: str) -> dict[str, Any]:
    """Cut the remix kits of a finished run, one kit per moment.

    `streamer` is an allow-list id (which must be `verified`) or `self` for
    our own footage.
    """
    config = _config()
    run_dir = _run_dir(config, run_name)
    if not (run_dir / "moments.json").exists():
        raise McpError(
            f"there are no moments in {run_name} — run `clipper listen` first"
        )
    entry = None
    if streamer != SELF_ID:
        entry = _streamer_by_id(streamer)
        if not allowlist.allowed(entry):
            raise McpError(
                f"refused: {entry.id} ({entry.name}) may not be clipped: "
                f"{allowlist.reason(entry)}"
            )
    kits = kit_builder.build_kits(run_dir, streamer, config, entry)
    return {
        "run": run_name,
        "streamer": streamer,
        "kits": [
            {
                "rank": kit["rank"],
                "mmss": kit["mmss"],
                "score": kit["score"],
                "times": kit["times"],
                "folder": str(Path(kit["clip"]).parent),
                "hooks": kit["hooks"],
                "credit": kit["credit"],
            }
            for kit in kits
        ],
    }


@server.tool()
def list_kits(status: str | None = None) -> dict[str, Any]:
    """Every kit, newest run first. `status` may be new, approved or rejected."""
    config = _config()
    kits = library.list_kits(config.data_dir, status=status)
    return {"kits": [_kit_summary(kit) for kit in kits]}


@server.tool()
def get_kit(run: str, kit: str) -> dict[str, Any]:
    """One kit: its times, its signals, its hooks and its status."""
    config = _config()
    try:
        found = library.get_kit(config.data_dir, run, kit)
    except library.LibraryError as exc:
        raise McpError(str(exc)) from exc
    return {"kit": _kit_summary(found)}


@server.tool()
def approve(run: str, kit: str, hook_index: int = 0, memes: list[str] | None = None) -> dict[str, Any]:
    """Approve a kit: the vertical and the caption go to the hand-off folder.

    `hook_index` picks the hook line (0, 1 or 2). A person still uploads it.
    `memes` are names from `list_memes` (e.g. ["shock/speed-gasp.mp4"]), played in
    order after the clip like the reaction cuts at the end of a TikTok. OPTIONAL:
    leave it out for no meme (the default). Add one only when the user asks for a
    meme or the clip clearly calls for it; see `kit_emotion` to pick by emotion.
    """
    config = _config()
    try:
        result = library.approve_kit(
            config.data_dir, run, kit, hook_index=int(hook_index), config=config, who="mcp", memes=memes
        )
    except library.LibraryError as exc:
        raise McpError(str(exc)) from exc
    return {
        "run": result["run"],
        "kit": result["kit"],
        "status": result["status"],
        "handoff": result["handoff"],
        "files": result["files"],
        "caption": result["caption"],
    }


@server.tool()
def kit_emotion(run: str, kit: str) -> dict[str, Any]:
    """What a clip feels like, so a fitting meme can be picked for its ending.

    Returns the emotion its signals suggest, the hook lines, the words around
    the peak, and the memes available for that emotion. Read the words and
    override the suggestion when it is wrong, then pass a meme to `approve`.
    """
    config = _config()
    try:
        found = library.get_kit(config.data_dir, run, kit)
    except library.LibraryError as exc:
        raise McpError(str(exc)) from exc
    chips = [str(c) for c in (found.get("chips") or [])]
    emotion = meme_outro.suggest_emotion(chips)
    words: list[str] = []
    transcript = Path(found["folder"]) / "transcript.json"
    if transcript.exists():
        try:
            data = json.loads(transcript.read_text(encoding="utf-8"))
            peak, start = found["times"]["peak"], found["times"]["start"]
            words = [
                str(w.get("word", "")) for w in data.get("words", [])
                if abs(float(w.get("start", 0)) - (peak - start)) <= 8
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            words = []
    return {
        "suggested_emotion": emotion,
        "signals": chips,
        "hooks": found["hooks"],
        "words_near_peak": " ".join(words),
        "emotions": list(meme_outro.EMOTIONS),
        "memes": meme_outro.list_memes(config.data_dir, emotion),
    }


@server.tool()
def list_memes(emotion: str | None = None) -> dict[str, Any]:
    """The meme library (data/memes/<emotion>/), one emotion or all of them."""
    return {"memes": meme_outro.list_memes(_config().data_dir, emotion)}


@server.tool()
def reject(run: str, kit: str, reason: str | None = None) -> dict[str, Any]:
    """Reject a kit. Nothing is deleted; the files stay where they are."""
    config = _config()
    try:
        result = library.reject_kit(
            config.data_dir, run, kit, reason=reason or "", config=config, who="mcp"
        )
    except library.LibraryError as exc:
        raise McpError(str(exc)) from exc
    return {
        "run": result["run"],
        "kit": result["kit"],
        "status": result["status"],
        "reason": result["reason"],
    }


@server.tool()
def trim(run: str, kit: str, in_s: float, out_s: float) -> dict[str, Any]:
    """Re-cut a kit between two points of the source.

    The in and out points are seconds in the source video. The old files are
    moved to `prev/` inside the kit; nothing is deleted.
    """
    config = _config()
    try:
        result = library.apply_trim(
            config.data_dir, run, kit, in_s=in_s, out_s=out_s, config=config, who="mcp"
        )
    except library.LibraryError as exc:
        raise McpError(str(exc)) from exc
    return {
        "run": result["run"],
        "kit": result["kit"],
        "times": result["times"],
        "kept_previous": result["kept_previous"],
    }


@server.tool()
def handoff_list() -> dict[str, Any]:
    """What is waiting in the hand-off folder for the person who uploads."""
    config = _config()
    return {"handoff": library.handoff_list(config)}


def main() -> int:
    """Serve the tools over stdio."""
    # stdout belongs to the protocol, so anything else goes to stderr.
    print("streamclipper mcp server ready on stdio", file=sys.stderr)
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
