"""StreamClipper — a listening agent attached to allow-listed streamers.

The package is deliberately small and stdlib-first:

* `allowlist` — the hard rule: who may be watched, recorded or clipped.
* `platforms` — one `is_live()` per platform, with the HTTP call injected.
* `watch` — the live watcher loop.
* `record` — the streamlink recorder and its disk budget.
* `listen` — the listening agent: ranked moments from a recording.
* `kit` — the remix kits: a cut clip plus everything needed to remix it.
"""

__version__ = "0.1.0"
