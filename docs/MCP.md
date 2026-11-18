# The StreamClipper MCP server

`clipper/mcp_server.py` serves the eleven tools of the clipper over the Model
Context Protocol, on stdio. It is the same code the CLI and the review page
use — `clipper.library`, `clipper.listen`, `clipper.kit`,
`clipper.allowlist` — so there is one implementation of every rule.

Nothing is posted and nothing leaves your machine: the server runs locally,
with this repository's virtualenv.

## Registering it in Claude Code

To add it to your own Claude Code (from the repository root, after setting up the virtualenv; see the README):

```
claude mcp add streamclipper -- <repo>/.venv/bin/python -m clipper.mcp_server
```

The same thing as a `.mcp.json` file (put it in the project you want it in):

```json
{
  "mcpServers": {
    "streamclipper": {
      "command": "<repo>/.venv/bin/python",
      "args": ["-m", "clipper.mcp_server"],
      "cwd": "<repo>"
    }
  }
}
```

The virtualenv matters: the tools need `numpy`, `torch` (CPU), `faster-whisper`
and `mcp`, and `ffmpeg` on the PATH.

## The tools

| Tool | What it does | Returns |
|---|---|---|
| `list_streamers()` | the allow-list, and why each entry is or is not allowed | id, platform, status, allowed, why, credit |
| `listen(video_path, run_name, top=5, streamer=None)` | ranks the moments in a video and writes them to a run folder | the moments: rank, time, score, which signals fired |
| `make_kits(run_name, streamer)` | cuts the remix kits of a finished run | one entry per kit: rank, times, folder, hooks, credit |
| `list_kits(status=None)` | every kit, newest run first | kit, times, score, status, chips, hooks, paths |
| `get_kit(run, kit)` | one kit | the same, for one kit |
| `approve(run, kit, hook_index=0, memes=None)` | approves a kit and puts it in the hand-off folder; optional `memes` (names from `list_memes`) are appended as closing reaction cards | the hand-off path, the files, the caption |
| `kit_emotion(run, kit)` | the emotion a clip suggests, its words near the peak and the memes for that emotion | emotion, signals, hooks, memes |
| `list_memes(emotion=None)` | the meme library, `data/memes/<emotion>/` (your own files) | names by emotion |
| `reject(run, kit, reason=None)` | rejects a kit | the status and the reason |
| `trim(run, kit, in_s, out_s)` | re-cuts a kit between two points of the source | the new window |
| `handoff_list()` | what is waiting to be uploaded | one entry per approved kit |

Every result is small JSON — paths, times, scores, status. No media bytes
come back through the protocol; the tools never read the video into the
answer.

## Two gates

The MCP surface is stricter than the CLI in two places, because an agent
drives it and a person does not:

* **`listen`** only takes our own material: a file under `data/` or under
  `~/own-uploads`. Any other path needs `streamer` to name a streamer
  who is on the allow-list *and* allowed (`status: verified` with a proof
  entry that has a url and a checked date).
* **`make_kits`** only takes an allow-listed streamer whose status is
  `verified`, or `self` for our own footage.

Everything else is the allow-list rule the rest of the project follows: an
unknown streamer is refused, not guessed.

## What a decision writes

Approving, rejecting, trimming or cropping appends one line to
`data/decisions.jsonl`:

```json
{"time": "2026-11-17T06:12:44+00:00", "run": "blade", "kit": "1-02-25.2",
 "action": "approve", "who": "mcp", "values": {"hook_index": 0, "handoff": "..."}}
```

`who` is `mcp` for these tools, `local` for the review page on your machine,
and the `Tailscale-User-Login` header when an identity-aware proxy is in front of
the page. The last approve or reject decides the kit's status: new,
approved or rejected.

## The tests

`.venv/bin/python -m pytest tests/test_mcp_server.py -q` calls every tool
directly against a temporary data folder, and then does one handshake over
stdio: it starts the server as a subprocess, initialises a session and lists
the tools.
