"""Tests for tools/serve_viewer.py indexing helpers."""

import json
from pathlib import Path

import tools.serve_viewer as serve_viewer


def _write_result(folder: Path, payload: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(json.dumps(payload))


class TestBuildSessionsIndex:
    def test_skips_invalid_and_sorts_by_processed_at(self, tmp_path):
        _write_result(
            tmp_path / "old_result",
            {
                "session_id": "old",
                "duration_seconds": 10,
                "processed_at": "2026-04-01T00:00:00Z",
                "chapters": [{"events": [{}, {}]}],
            },
        )
        _write_result(
            tmp_path / "new_result",
            {
                "session_id": "new",
                "duration_seconds": 20,
                "processed_at": "2026-04-02T00:00:00Z",
                "chapters": [{"events": [{}]}, {"events": [{}, {}]}],
            },
        )
        _write_result(
            tmp_path / "fallback_id_result",
            {
                "duration_seconds": 30,
                "processed_at": "2026-04-03T00:00:00Z",
                "chapters": [],
            },
        )

        # Should be ignored: malformed JSON.
        invalid = tmp_path / "bad_result"
        invalid.mkdir()
        (invalid / "result.json").write_text("{not-json")

        # Should be ignored: folder doesn't end with _result.
        _write_result(tmp_path / "not_a_result_folder", {"session_id": "x", "chapters": []})

        # Should be ignored: _result folder without result.json.
        (tmp_path / "empty_result").mkdir()

        sessions = serve_viewer.build_sessions_index(tmp_path)

        assert [s["session_id"] for s in sessions] == ["fallback_id", "new", "old"]
        assert sessions[0]["folder"] == "fallback_id_result"
        assert sessions[1]["chapter_count"] == 2
        assert sessions[1]["event_count"] == 3
        assert sessions[2]["chapter_count"] == 1
        assert sessions[2]["event_count"] == 2


class TestWriteIndex:
    def test_writes_sessions_json(self, tmp_path):
        sessions = [{"session_id": "abc", "folder": "abc_result", "chapter_count": 1, "event_count": 2}]

        output_path = serve_viewer.write_index(tmp_path, sessions)

        assert output_path == tmp_path / "sessions.json"
        written = json.loads(output_path.read_text())
        assert written == {"sessions": sessions}


class TestMain:
    def test_main_indexes_and_opens_browser_without_starting_server(self, tmp_path, monkeypatch):
        _write_result(
            tmp_path / "demo_result",
            {
                "session_id": "demo",
                "duration_seconds": 5,
                "processed_at": "2026-04-04T00:00:00Z",
                "chapters": [{"events": [{}]}],
            },
        )

        monkeypatch.setattr(
            "sys.argv",
            ["serve_viewer.py", "--root", str(tmp_path), "--port", "8123"],
        )

        opened_urls: list[str] = []
        monkeypatch.setattr(serve_viewer.webbrowser, "open", lambda url: opened_urls.append(url))

        server_calls: list[tuple[Path, int]] = []
        monkeypatch.setattr(serve_viewer, "serve", lambda root, port: server_calls.append((root, port)))

        timer_calls: list[float] = []

        class _FakeTimer:
            def __init__(self, delay, func):
                timer_calls.append(delay)
                self._func = func

            def start(self):
                self._func()

        monkeypatch.setattr(serve_viewer.threading, "Timer", _FakeTimer)

        serve_viewer.main()

        written_index = json.loads((tmp_path / "sessions.json").read_text())
        assert written_index["sessions"][0]["session_id"] == "demo"
        assert server_calls == [(tmp_path.resolve(), 8123)]
        assert timer_calls == [0.6]
        assert opened_urls == ["http://127.0.0.1:8123/viewer.html"]
