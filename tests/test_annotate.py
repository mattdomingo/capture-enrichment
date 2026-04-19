"""Tests for annotate.py — prompt building and response parsing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from capture_enrichment.annotate import _apply_source_weights, _build_prompt, _parse_events, _sec_to_ts
from capture_enrichment.models import Event, VideoChunk


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


class TestApplySourceWeights:
    """Tests for the deterministic pinch-only confidence penalty."""

    def _pinch_telemetry(self, transcript: str = "", objects: list | None = None) -> dict:
        return {
            "gestures": [{"t": 5.0, "hand": "right", "type": "pinch"}],
            "tracked_objects": objects if objects is not None else [],
            "transcript": transcript,
        }

    def _make_event(self, timestamp: str = "00:00:05", action: str = "User pinches to select", confidence: float = 0.95) -> Event:
        return Event(timestamp=timestamp, action=action, confidence=confidence)

    def test_pinch_only_event_gets_penalized(self):
        """Pinch keyword + near pinch + no transcript + no objects → confidence reduced."""
        event = self._make_event()
        result = _apply_source_weights([event], self._pinch_telemetry())
        assert result[0].confidence < 0.95
        assert result[0].confidence <= 0.85

    def test_penalty_is_multiplier(self):
        """Penalized confidence equals original × 0.85."""
        event = self._make_event(confidence=0.95)
        result = _apply_source_weights([event], self._pinch_telemetry())
        assert abs(result[0].confidence - round(0.95 * 0.85, 4)) < 1e-6

    def test_speech_present_suppresses_penalty(self):
        """Transcript speech present → no penalty even if pinch keyword matches."""
        event = self._make_event()
        result = _apply_source_weights([event], self._pinch_telemetry(transcript="open that"))
        assert result[0].confidence == 0.95

    def test_tracked_objects_suppress_penalty(self):
        """Tracked objects present → no penalty."""
        event = self._make_event()
        telemetry = self._pinch_telemetry(objects=[{"name": "bowl", "positions": []}])
        result = _apply_source_weights([event], telemetry)
        assert result[0].confidence == 0.95

    def test_no_pinch_keyword_no_penalty(self):
        """Action without pinch-related keywords is not penalized."""
        event = self._make_event(action="User looks around the room")
        result = _apply_source_weights([event], self._pinch_telemetry())
        assert result[0].confidence == 0.95

    def test_event_far_from_pinch_no_penalty(self):
        """Event timestamp more than 1s away from any pinch → no penalty."""
        event = self._make_event(timestamp="00:00:20")  # 20s; pinch at 5s
        result = _apply_source_weights([event], self._pinch_telemetry())
        assert result[0].confidence == 0.95

    def test_no_pinch_gestures_no_penalty(self):
        """No pinch gestures in telemetry → no penalty."""
        telemetry = {"gestures": [], "tracked_objects": [], "transcript": ""}
        event = self._make_event()
        result = _apply_source_weights([event], telemetry)
        assert result[0].confidence == 0.95

    def test_already_high_confidence_capped(self):
        """Confidence multiplied by 0.85 never exceeds 0.85."""
        event = self._make_event(confidence=1.0)
        result = _apply_source_weights([event], self._pinch_telemetry())
        assert result[0].confidence <= 0.85

    def test_multiple_events_independent(self):
        """Penalty applies per-event; non-matching events are unchanged."""
        pinch_event = self._make_event(timestamp="00:00:05", action="User taps button", confidence=0.95)
        visual_event = self._make_event(timestamp="00:00:15", action="User opens application", confidence=0.88)
        result = _apply_source_weights([pinch_event, visual_event], self._pinch_telemetry())
        assert result[0].confidence < 0.95   # penalized
        assert result[1].confidence == 0.88  # unchanged

    def test_thumbnail_path_preserved(self):
        """Penalized events retain thumbnail_path."""
        event = Event(timestamp="00:00:05", action="User pinches", confidence=0.95, thumbnail_path="/tmp/thumb.jpg")
        result = _apply_source_weights([event], self._pinch_telemetry())
        assert result[0].thumbnail_path == "/tmp/thumb.jpg"


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

    def test_prompt_contains_source_priority(self):
        chunk = _make_chunk()
        prompt = _build_prompt(chunk)
        assert "VIDEO" in prompt
        assert "PINCH GESTURES" in prompt
        assert "corroborating signal" in prompt.lower() or "corroborating" in prompt.lower()

    def test_prompt_contains_confidence_rubric(self):
        chunk = _make_chunk()
        prompt = _build_prompt(chunk)
        assert "0.9" in prompt
        assert "0.7" in prompt
        assert "0.5" in prompt
