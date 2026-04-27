"""Tests for ingest.py — capture package loading and parsing."""

import json
import pytest

from capture_enrichment.ingest import load_capture_package, load_metadata, load_transcript_tokens


class TestLoadCapturePackage:
    def test_capture1_uses_sbs_video(self, capture_1):
        pkg = load_capture_package(capture_1)
        assert pkg.video_path.name == "camera_sbs.mov"

    def test_capture2_falls_back_to_left(self, capture_2):
        pkg = load_capture_package(capture_2)
        assert pkg.video_path.name == "camera_left.mov"

    def test_capture1_no_tracked_objects(self, capture_1):
        pkg = load_capture_package(capture_1)
        assert pkg.tracked_objects == []

    def test_capture2_has_tracked_object(self, capture_2):
        pkg = load_capture_package(capture_2)
        assert len(pkg.tracked_objects) == 1
        assert pkg.tracked_objects[0].name == "Nakamichi 610 Scan"

    def test_all_paths_exist(self, capture_1, capture_2):
        for capture in (capture_1, capture_2):
            pkg = load_capture_package(capture)
            for attr in ("video_path", "metadata_json", "device_pose_csv",
                         "hand_pose_world_csv", "object_pose_csv", "transcript_json"):
                assert getattr(pkg, attr).exists(), f"{attr} missing in {capture.name}"

    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises((ValueError, FileNotFoundError)):
            load_capture_package(tmp_path / "nonexistent.capture")

    def test_world_anchor_normalised(self, capture_1, capture_2):
        """Both captures should return world_anchor as a plain string or empty string."""
        for capture in (capture_1, capture_2):
            pkg = load_capture_package(capture)
            meta = load_metadata(pkg)
            wa = meta.get("world_anchor")
            if wa is not None:
                assert isinstance(wa, str), f"world_anchor should be str, got {type(wa)}"


class TestLoadTranscriptTokens:
    def test_capture1_has_tokens(self, capture_1):
        pkg = load_capture_package(capture_1)
        tokens = load_transcript_tokens(pkg)
        assert len(tokens) > 0
        assert all("text" in t and "startSec" in t and "endSec" in t for t in tokens)

    def test_capture2_empty_transcript_returns_empty_list(self, capture_2):
        pkg = load_capture_package(capture_2)
        tokens = load_transcript_tokens(pkg)
        assert tokens == []

    def test_token_types(self, capture_1):
        pkg = load_capture_package(capture_1)
        tokens = load_transcript_tokens(pkg)
        for t in tokens:
            assert isinstance(t["text"], str)
            assert isinstance(t["startSec"], float)
            assert isinstance(t["endSec"], float)

    def test_missing_transcript_file_returns_empty(self, tmp_path):
        """Gracefully handle a transcript file that is missing or has bad JSON."""
        # Create a minimal fake package structure
        (tmp_path / "metadata").mkdir()
        (tmp_path / "video").mkdir()
        (tmp_path / "tracking").mkdir()
        (tmp_path / "transcripts").mkdir()
        (tmp_path / "vlm").mkdir()

        # Write required stubs
        (tmp_path / "metadata" / "metadata.json").write_text(
            json.dumps({"id": "x", "duration": 10, "start_uptime": 0,
                        "start_wall": 0, "tracked_objects": []})
        )
        for name in ("device_pose.csv", "hand_pose_world.csv", "object_pose.csv"):
            (tmp_path / "tracking" / name).write_text("t_mono,t_wall\n")
        (tmp_path / "video" / "camera_left.mov").write_bytes(b"")
        transcript = tmp_path / "transcripts" / "timecoded_transcript.json"
        transcript.write_text("not valid json {{{{")

        pkg = load_capture_package(tmp_path)
        assert load_transcript_tokens(pkg) == []
