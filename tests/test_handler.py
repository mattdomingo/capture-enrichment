"""Tests for handler.py orchestration and output writing."""

import json
from pathlib import Path
from types import SimpleNamespace

from capture_enrichment import handler
from capture_enrichment.config import Config
from capture_enrichment.models import Chapter, Event


class TestUniqueResultDir:
    def test_adds_suffix_when_needed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "session-1_result").mkdir()
        (tmp_path / "session-1_result (1)").mkdir()

        result = handler._unique_result_dir("session-1")

        assert result.name == "session-1_result (2)"
        assert not result.exists()


class TestProcessCapture:
    def test_writes_result_json_and_thumbnails(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        pkg = SimpleNamespace(tracked_objects=[])
        events = [
            Event(timestamp="00:00:04", action="Open menu", confidence=0.9),
            Event(timestamp="00:00:15", action="Select item", confidence=0.8),
        ]
        chapters = [
            Chapter(
                id=1,
                title="Task",
                start_ts="00:00:00",
                end_ts="00:00:20",
                events=events,
            )
        ]

        monkeypatch.setattr(handler, "load_capture_package", lambda _: pkg)
        monkeypatch.setattr(handler, "load_metadata", lambda _: {"id": "session-abc", "duration": 123.4})
        monkeypatch.setattr(handler, "load_transcript_tokens", lambda _: [])
        monkeypatch.setattr(handler, "load_device_pose", lambda _: object())
        monkeypatch.setattr(handler, "load_hand_pose", lambda _: object())
        monkeypatch.setattr(handler, "load_object_pose", lambda _: object())
        monkeypatch.setattr(handler, "create_client", lambda _: object())
        monkeypatch.setattr(handler, "select_video", lambda _: tmp_path / "input.mov")
        monkeypatch.setattr(handler, "downsample_video", lambda *_args, **_kwargs: Path(_args[1]))
        monkeypatch.setattr(handler, "get_video_duration", lambda _p: 20.0)
        monkeypatch.setattr(handler, "plan_chunks", lambda *_args, **_kwargs: [(0.0, 20.0)])
        monkeypatch.setattr(handler, "extract_chunk", lambda *_args, **_kwargs: Path(_args[1]))
        monkeypatch.setattr(
            handler,
            "build_telemetry",
            lambda *_args, **_kwargs: {"head_movement": [], "gestures": [], "tracked_objects": [], "transcript": ""},
        )
        monkeypatch.setattr(handler, "annotate_chunk", lambda *_args, **_kwargs: events)
        monkeypatch.setattr(handler, "deduplicate_events", lambda _chunk_results: events)
        monkeypatch.setattr(handler, "segment_into_chapters", lambda *_args, **_kwargs: chapters)

        def _fake_extract_thumbnail(_video_path, out_path, _offset):
            Path(out_path).write_bytes(b"jpg")

        monkeypatch.setattr(handler, "extract_thumbnail", _fake_extract_thumbnail)

        cfg = Config(gemini_api_key="test-key")
        result_dir = handler.process_capture(Path("/fake/session.capture"), cfg)

        assert result_dir.name == "session-abc_result"
        result_json = result_dir / "result.json"
        thumbnails_dir = result_dir / "thumbnails"
        assert result_json.exists()
        assert thumbnails_dir.exists()

        output = json.loads(result_json.read_text())
        assert output["session_id"] == "session-abc"
        assert output["duration_seconds"] == 123.4
        assert len(output["chapters"]) == 1
        assert len(output["chapters"][0]["events"]) == 2
        assert output["chapters"][0]["events"][0]["thumbnail_path"] == "thumbnails/ch1_pt1.jpg"
        assert output["chapters"][0]["events"][1]["thumbnail_path"] == "thumbnails/ch1_pt2.jpg"
        assert (thumbnails_dir / "ch1_pt1.jpg").exists()
        assert (thumbnails_dir / "ch1_pt2.jpg").exists()
