"""Tests for annotate.py — prompt building and response parsing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from capture_enrichment.annotate import _build_prompt, _parse_events, _sec_to_ts
from capture_enrichment.models import VideoChunk


def _make_chunk(start: float = 0.0, end: float = 30.0) -> VideoChunk:
    return VideoChunk(
        start_sec=start,
        end_sec=end,
        video_path=Path("/tmp/chunk.mov"),
        telemetry={
            "head_movement": [{"t": 0.0, "displacement_m": 0.01, "label": "static"}],
            "gestures": [{"t": 5.0, "hand": "right", "type": "pinch"}],
            "tracked_objects": [],
            "transcript": "Test speech",
        },
    )


class TestSecToTs:
    def test_zero(self):
        assert _sec_to_ts(0) == "00:00:00"

    def test_minutes(self):
        assert _sec_to_ts(90) == "00:01:30"

    def test_hours(self):
        assert _sec_to_ts(3661) == "01:01:01"


class TestParseEvents:
    def test_valid_json_array(self):
        text = '[{"timestamp": "00:00:04", "action": "User pinches", "confidence": 0.9}]'
        events = _parse_events(text)
        assert len(events) == 1
        assert events[0].timestamp == "00:00:04"
        assert events[0].confidence == 0.9

    def test_markdown_fenced(self):
        text = '```json\n[{"timestamp": "00:00:10", "action": "Tap", "confidence": 0.8}]\n```'
        events = _parse_events(text)
        assert len(events) == 1
        assert events[0].action == "Tap"

    def test_empty_array(self):
        assert _parse_events("[]") == []

    def test_malformed_json_returns_empty(self):
        assert _parse_events("Sorry, I cannot analyze this.") == []

    def test_non_list_json_returns_empty(self):
        assert _parse_events('{"timestamp": "00:00:01", "action": "x"}') == []

    def test_empty_string_returns_empty(self):
        assert _parse_events("") == []

    def test_missing_confidence_defaults_to_half(self):
        text = '[{"timestamp": "00:00:05", "action": "No confidence"}]'
        events = _parse_events(text)
        assert events[0].confidence == 0.5

    def test_multiple_events(self):
        text = '[{"timestamp": "00:00:01", "action": "A", "confidence": 0.7}, {"timestamp": "00:00:02", "action": "B", "confidence": 0.8}]'
        events = _parse_events(text)
        assert len(events) == 2


class TestBuildPrompt:
    def test_prompt_contains_segment_times(self):
        chunk = _make_chunk(start=90.0, end=120.0)
        prompt = _build_prompt(chunk)
        assert "00:01:30" in prompt
        assert "00:02:00" in prompt

    def test_prompt_contains_telemetry_keys(self):
        chunk = _make_chunk()
        prompt = _build_prompt(chunk)
        assert "head_movement" in prompt
        assert "gestures" in prompt
        assert "transcript" in prompt

    def test_prompt_contains_speech(self):
        chunk = _make_chunk()
        prompt = _build_prompt(chunk)
        assert "Test speech" in prompt
