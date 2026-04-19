"""Serve the capture-enrichment viewer locally.

Scans the current directory for ``*_result/`` folders, writes a fresh
``sessions.json`` index, then starts a static HTTP server and opens the
viewer in a browser.

Usage:
    uv run python tools/serve_viewer.py [--port 8000] [--no-open]
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import threading
import time
import webbrowser
from pathlib import Path


def build_sessions_index(root: Path) -> list[dict]:
    sessions: list[dict] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.endswith("_result"):
            continue
        result_json = entry / "result.json"
        if not result_json.is_file():
            continue
        try:
            data = json.loads(result_json.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sessions.append(
            {
                "session_id": data.get("session_id", entry.name.removesuffix("_result")),
                "folder": entry.name,
                "duration": data.get("duration_seconds"),
                "processed_at": data.get("processed_at"),
                "chapter_count": len(data.get("chapters", [])),
                "event_count": sum(len(c.get("events", [])) for c in data.get("chapters", [])),
            }
        )
    sessions.sort(key=lambda s: s.get("processed_at") or "", reverse=True)
    return sessions


def write_index(root: Path, sessions: list[dict]) -> Path:
    path = root / "sessions.json"
    path.write_text(json.dumps({"sessions": sessions}, indent=2))
    return path


def serve(root: Path, port: int) -> None:
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *a, directory=str(root), **kw
    )

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving {root} at http://127.0.0.1:{port}/viewer.html")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    sessions = build_sessions_index(root)
    write_index(root, sessions)
    print(f"Indexed {len(sessions)} session(s) → {root / 'sessions.json'}")
    for s in sessions:
        print(f"  · {s['session_id']}  ({s['chapter_count']} chapters, {s['event_count']} events)")

    if not args.no_open and sessions:
        url = f"http://127.0.0.1:{args.port}/viewer.html"
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    serve(root, args.port)


if __name__ == "__main__":
    main()
