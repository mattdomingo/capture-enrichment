"""Tests for segment.py — deduplication and chapter parsing."""

import pytest

from capture_enrichment.models import Event
from capture_enrichment.segment import (
    _fallback_chapter,
    _parse_chapters,
    _ts_to_sec,
    deduplicate_events,
)


def _ev(ts: str, action: str = "action", confidence: float = 0.8) -> Event:
    return Event(timestamp=ts, action=action, confidence=confidence)


class TestTsToSec:
    def test_zero(self):
        assert _ts_to_sec("00:00:00") == 0

    def test_minutes(self):
        assert _ts_to_sec("00:01:30") == 90

    def test_hours(self):
        assert _ts_to_sec("01:00:00") == 3600


class TestDeduplicateEvents:
    def test_keeps_higher_confidence_within_window(self):
        chunk_events = [
            (0.0, [_ev("00:00:10", "Low confidence", 0.5)]),
            (5.0, [_ev("00:00:11", "High confidence", 0.9)]),  # 1s apart → dedup
        ]
        result = deduplicate_events(chunk_events, window_sec=3.0)
        assert len(result) == 1
        assert result[0].action == "High confidence"

    def test_keeps_both_events_outside_window(self):
        chunk_events = [
            (0.0, [_ev("00:00:05", "Event A", 0.7)]),
            (0.0, [_ev("00:00:09", "Event B", 0.8)]),  # 4s apart > 3s window
        ]
        result = deduplicate_events(chunk_events, window_sec=3.0)
        assert len(result) == 2

    def test_result_is_sorted_by_timestamp(self):
        chunk_events = [
            (10.0, [_ev("00:00:20")]),
            (0.0, [_ev("00:00:05")]),
        ]
        result = deduplicate_events(chunk_events)
        assert _ts_to_sec(result[0].timestamp) < _ts_to_sec(result[1].timestamp)

    def test_empty_returns_empty(self):
        assert deduplicate_events([]) == []

    def test_exactly_at_window_boundary(self):
        # Events exactly 3.0s apart — NOT within window → both kept
        chunk_events = [
            (0.0, [_ev("00:00:00")]),
            (0.0, [_ev("00:00:03")]),
        ]
        result = deduplicate_events(chunk_events, window_sec=3.0)
        assert len(result) == 2

    def test_single_event_per_chunk(self):
        chunk_events = [(i * 30.0, [_ev(f"00:0{i}:00")]) for i in range(3)]
        result = deduplicate_events(chunk_events)
        assert len(result) == 3


class TestParseChapters:
    def test_valid_json(self):
        events = [_ev("00:00:05")]
        text = '[{"id":1,"title":"Intro","start_ts":"00:00:00","end_ts":"00:00:30","events":[{"timestamp":"00:00:05","action":"action","confidence":0.8}]}]'
        chapters = _parse_chapters(text, events, 30.0)
        assert len(chapters) == 1
        assert chapters[0].title == "Intro"
        assert len(chapters[0].events) == 1

    def test_malformed_falls_back(self):
        events = [_ev("00:00:05")]
        chapters = _parse_chapters("not valid json", events, 30.0)
        assert len(chapters) == 1
        assert chapters[0].title == "Session"
        assert chapters[0].events == events

    def test_empty_response_falls_back(self):
        events = [_ev("00:00:05")]
        chapters = _parse_chapters("", events, 30.0)
        assert chapters[0].title == "Session"

    def test_markdown_fenced_response(self):
        events = [_ev("00:00:05")]
        text = '```json\n[{"id":1,"title":"Test","start_ts":"00:00:00","end_ts":"00:00:30","events":[]}]\n```'
        chapters = _parse_chapters(text, events, 30.0)
        assert chapters[0].title == "Test"

    def test_fallback_chapter_spans_full_duration(self):
        events = [_ev("00:00:10")]
        chapters = _fallback_chapter(events, 90.0)
        assert chapters[0].start_ts == "00:00:00"
        assert chapters[0].end_ts == "00:01:30"
        assert chapters[0].events == events
