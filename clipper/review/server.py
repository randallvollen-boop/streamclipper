"""The review page: the small HTTP server the phone opens.

It is bound to 127.0.0.1 and nothing else (never 0.0.0.0), so it is only
reachable from this machine, or through a reverse proxy in front of it. The
decisions it takes are written by `clipper.library`, which is the same code
the CLI and the MCP server use.

What it serves:

    GET  /                                  the page
    GET  /api/kits                          every kit, newest run first
    GET  /media/<run>/<kit>/<file>          clip.mp4 or vertical.mp4 (Range)
    POST /api/kits/<run>/<kit>/decision     approve, reject, trim or crop
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .. import library
from ..config import Config, load as load_config

#: The page itself.
INDEX_PATH = Path(__file__).resolve().parent / "index.html"

#: The actions a decision may take.
ACTIONS = ("approve", "reject", "trim", "crop")

#: `bytes=start-end`
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

#: The biggest decision body we read.
MAX_BODY_BYTES = 64 * 1024


class ReviewError(Exception):
    """A request cannot be served. The message goes to the operator."""


class ReviewHandler(BaseHTTPRequestHandler):
    """One request."""

    server_version = "StreamClipper/0.1"
    protocol_version = "HTTP/1.1"

    # -- helpers ------------------------------------------------------------

    @property
    def config(self) -> Config:
        return self.server.config  # type: ignore[attr-defined]

    @property
    def data_dir(self) -> Path:
        return self.server.data_dir  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """One line per request on stderr, so the operator sees who did what."""
        sys.stderr.write(
            f"{self.log_date_time_string()} {self.address_string()} {format % args}\n"
        )

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def handle_one_request(self) -> None:
        """A phone that walks away mid-download is not an error."""
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"ok": False, "error": message})

    def _who(self) -> str:
        """Who made the request: a reverse proxy says who, otherwise local."""
        for name in ("Tailscale-User-Login", "X-Forwarded-User"):
            value = self.headers.get(name)
            if value and value.strip():
                return value.strip()
        return "local"

    # -- GET ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send(
                    HTTPStatus.OK,
                    INDEX_PATH.read_bytes(),
                    "text/html; charset=utf-8",
                )
                return
            if path == "/api/kits":
                kits = library.list_kits(self.data_dir)
                self._json(HTTPStatus.OK, {"ok": True, "kits": kits})
                return
            if path.startswith("/media/"):
                self._serve_media(path)
                return
            self._error(HTTPStatus.NOT_FOUND, "There is nothing at this address.")
        except ReviewError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:  # pragma: no cover - the operator sees it
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{exc}")

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def _serve_media(self, path: str) -> None:
        """Serve clip.mp4 or vertical.mp4, with Range so a phone can seek."""
        parts = [unquote(part) for part in path.split("/") if part]
        # ['media', run, kit, file]
        if len(parts) != 4:
            raise ReviewError("The media address is not a kit file.")
        _, run, kit, name = parts
        if name not in library.MEDIA_FILES:
            raise ReviewError("Only clip.mp4 and vertical.mp4 are served here.")
        try:
            folder = library.kit_folder(self.data_dir, run, kit)
        except library.LibraryError as exc:
            raise ReviewError(str(exc)) from exc
        # Belt and braces: the file has to be inside the runs folder.
        root = library.runs_root(self.data_dir).resolve()
        try:
            folder.resolve().relative_to(root)
        except ValueError:
            raise ReviewError("That address is outside the runs folder.") from None
        media = folder / name
        if not media.is_file():
            raise ReviewError(f"There is no {name} in this kit.")
        self._send_file(media)

    def _send_file(self, path: Path) -> None:
        size = path.stat().st_size
        start = 0
        end = size - 1
        partial = False
        header = self.headers.get("Range")
        if header:
            match = RANGE_RE.match(header.strip())
            if not match:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            first, last = match.group(1), match.group(2)
            if first == "" and last == "":
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if first == "":
                # The last N bytes.
                length = min(int(last), size)
                start = max(0, size - length)
                end = size - 1
            else:
                start = int(first)
                end = int(last) if last else size - 1
            if start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            end = min(end, size - 1)
            partial = True
        length = max(0, end - start + 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        remaining = length
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # -- POST ---------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            parts = [unquote(part) for part in path.split("/") if part]
            # ['api', 'kits', run, kit, 'decision']
            if len(parts) != 5 or parts[0] != "api" or parts[1] != "kits":
                self._error(HTTPStatus.NOT_FOUND, "There is nothing at this address.")
                return
            _, _, run, kit, action_route = parts
            if action_route != "decision":
                self._error(HTTPStatus.NOT_FOUND, "There is nothing at this address.")
                return
            body = self._read_json()
            action = str(body.get("action") or "").strip().lower()
            if action not in ACTIONS:
                self._error(
                    HTTPStatus.BAD_REQUEST,
                    f"The action must be one of {', '.join(ACTIONS)}.",
                )
                return
            result = self._decide(action, run, kit, body)
            self._json(HTTPStatus.OK, {"ok": True, **result})
        except ReviewError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except library.LibraryError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - the operator sees it
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{exc}")

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ReviewError("The request body has no length.") from None
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ReviewError("The request body is too big.")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewError(f"The request body is not JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ReviewError("The request body must be a JSON object.")
        return payload

    def _decide(
        self, action: str, run: str, kit: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Do what the decision says. The library does the work."""
        who = self._who()
        config = self.config
        data_dir = self.data_dir
        if action == "approve":
            hook_index = body.get("hook_index", 0)
            result = library.approve_kit(
                data_dir, run, kit, hook_index=int(hook_index), config=config, who=who
            )
        elif action == "reject":
            result = library.reject_kit(
                data_dir,
                run,
                kit,
                reason=str(body.get("reason") or ""),
                config=config,
                who=who,
            )
        elif action == "trim":
            if "in_s" not in body or "out_s" not in body:
                raise ReviewError("A trim needs an in point and an out point.")
            result = library.apply_trim(
                data_dir,
                run,
                kit,
                in_s=body["in_s"],
                out_s=body["out_s"],
                config=config,
                who=who,
            )
        else:  # crop
            if "crop_x" not in body:
                raise ReviewError("A crop needs a horizontal offset.")
            result = library.apply_crop(
                data_dir,
                run,
                kit,
                crop_x=body["crop_x"],
                config=config,
                who=who,
            )
        summary = library.get_kit(data_dir, run, kit)
        return {"result": result, "kit": summary, "who": who}


class ReviewServer(ThreadingHTTPServer):
    """The server, with the config and the data folder on it."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        config: Config,
        data_dir: Path,
    ) -> None:
        super().__init__(address, handler)
        self.config = config
        self.data_dir = Path(data_dir)


def make_server(
    config: Config,
    data_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> ReviewServer:
    """The server, ready to serve. It is always bound to the loopback."""
    if host not in ("127.0.0.1", "localhost"):
        raise ReviewError(
            f"the review page is only served on 127.0.0.1, not on {host}"
        )
    return ReviewServer(
        (host, config.review_port if port is None else int(port)),
        ReviewHandler,
        config,
        Path(data_dir) if data_dir is not None else config.data_dir,
    )


# --- CLI ---------------------------------------------------------------------


def add_review_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="the port to serve on (default from config: review_port)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="where the runs live (default from config: data_dir)",
    )


def run_review(args: argparse.Namespace) -> int:
    config = load_config()
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else config.data_dir
    try:
        server = make_server(config, data_dir=data_dir, port=args.port)
    except ReviewError as exc:
        print(f"review page error: {exc}", file=sys.stderr)
        return 2
    port = server.server_address[1]
    print(f"review page on http://127.0.0.1:{port}/  (Ctrl-C to stop)")
    print(f"kits from {data_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
