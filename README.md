# StreamClipper

A listening agent for streamers who allow clipping. It watches a live stream
(or a recording), ranks the moments worth clipping, and builds a **remix kit**
for each one: the cut clip, a vertical crop, the transcript with word times,
a credit line and three hook lines. A small review page lets a person approve,
reject or trim each kit. Nothing is posted anywhere; approved kits land in a
hand-off folder for a person to upload.

An MCP server exposes the same functions to an AI agent.

## The allow-list rule

**Only streamers who publicly allow clipping, with proof, may be watched,
recorded or clipped.** This is the hard rule of the project and the code
enforces it.

A streamer is allowed only when their entry in `config/streamers.yaml` has

* `status: verified`, **and**
* at least one `proof` entry with both a `url` and the date it was `checked`
  (a clipping-program brief, the streamer's own public statement, or a
  sponsor campaign that invites clips).

Anything else (unlisted, `unverified`, `pending`, `rejected`, or no proof) is
refused by the watcher, the recorder, the kit builder and the MCP server. An
unknown streamer is refused, not guessed. Do not mark an entry `verified`
unless you have read the policy yourself and saved where it says so.
`config/streamers.example.yaml` ships a single fictional, unverified entry.

## What it does

* **watch** polls Kick, Twitch and YouTube for allow-listed streamers going live.
* **record** records an allow-listed stream in chunks with a disk budget
  (needs `streamlink`).
* **listen** ranks moments from sound events (PANNs), loudness jumps, hype
  words in the transcript (faster-whisper) and, optionally, a chat replay.
  It runs on the CPU only.
* **kits** cuts each ranked moment into a remix kit (45-60 s, with build-up
  before the peak).
* **review** serves a phone-friendly review page, bound to `127.0.0.1`.
  Approved kits, with a caption and credit line, go to `data/handoff/<account>/`.
* An **MCP server** drives the same code (see [docs/MCP.md](docs/MCP.md)).

## Install

Needs Python 3.12, `ffmpeg` on the PATH, and `streamlink` for recording
(`pipx install streamlink`).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium   # only for the browser test
cp config/streamers.example.yaml config/streamers.yaml   # then edit it
```

Models are not downloaded at runtime. Put the PANNs `Cnn14_DecisionLevelMax.pth`
weights and `class_labels_indices.csv` in `~/panns_data/` (paths are set in
`config/clipper.yaml`); faster-whisper `small.en` comes from your HuggingFace cache.

Credentials, only for the platforms you watch: `TWITCH_CLIENT_ID` and
`TWITCH_TOKEN` for Twitch, `YOUTUBE_API_KEY` for YouTube. Kick needs none.

Set `STREAMCLIPPER_ALLOWLIST` to read the allow-list from another path.
Settings (weights, thresholds, kit lengths, review port, hand-off account) are
in `config/clipper.yaml`. Runtime files go to `data/`, which is git-ignored.

## CLI

```sh
.venv/bin/python -m clipper streamers                 # the allow-list, and who is allowed
.venv/bin/python -m clipper watch [--once]            # poll allow-listed streamers
.venv/bin/python -m clipper record <id> [--dry-run]   # record an allowed stream
.venv/bin/python -m clipper listen <video> --out data/runs/<name> [--chat chat.jsonl] [--top 5]
.venv/bin/python -m clipper kits data/runs/<name> --streamer <id|self>
.venv/bin/python -m clipper run <id> [--once]         # watch, record, listen, build kits
.venv/bin/python -m clipper run --from-file <video> --streamer <id>
.venv/bin/python -m clipper review [--port 8297]      # the review page
```

`--streamer self` is for your own footage.

## MCP

```sh
claude mcp add streamclipper -- <repo>/.venv/bin/python -m clipper.mcp_server
```

Replace `<repo>` with the path of your checkout. Eleven tools: `list_streamers`,
`listen`, `make_kits`, `list_kits`, `get_kit`, `approve`, `kit_emotion`, `list_memes`, `reject`, `trim`,
`handoff_list`. The MCP surface is stricter than the CLI: it refuses anyone
who is not allowed. Details in [docs/MCP.md](docs/MCP.md).

## Tests

```sh
.venv/bin/python -m pytest -q
```

## Licence

All rights reserved. The code is public to read; using, copying or redistributing it needs written permission. See [LICENSE](LICENSE).
