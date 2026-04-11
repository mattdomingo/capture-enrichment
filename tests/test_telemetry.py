"""Tests for telemetry.py — CSV summarisation and gesture detection."""

import numpy as np
import pandas as pd
import pytest

from capture_enrichment.models import TrackedObject
from capture_enrichment.telemetry import (
    _detect_gestures,
    _head_movement,
    _object_positions,
    _transcript_window,
    build_telemetry,
)

from .conftest import _small_device_df, _small_hand_df


class TestHeadMovement:
    def test_static_label(self):
        df = _small_device_df()
        # No movement → all displacements near 0 → static
        buckets = _head_movement(df, 0.0, 5.0, 1.0)
        assert len(buckets) == 5
        assert all(b["label"] == "static" for b in buckets)

    def test_active_label(self):
        df = _small_device_df()
        # Add large movement in second 2
        mask = (df["session_t"] >= 2.0) & (df["session_t"] < 3.0)
        df.loc[mask, "x"] = np.linspace(0, 5, mask.sum())  # 5m displacement in 1s
        buckets = _head_movement(df, 0.0, 5.0, 1.0)
        labels = {b["t"]: b["label"] for b in buckets}
        assert labels[2.0] == "active"

    def test_empty_window_returns_empty(self):
        df = _small_device_df(duration=10.0)
        result = _head_movement(df, 20.0, 30.0, 1.0)
        assert result == []

    def test_bucket_t_is_relative_to_start(self):
        df = _small_device_df()
        buckets = _head_movement(df, 5.0, 8.0, 1.0)
        assert buckets[0]["t"] == 0.0
        assert buckets[1]["t"] == 1.0


class TestDetectGestures:
    def test_pinch_detected_in_window(self):
        df = _small_hand_df(duration=10.0, pinch_start=3.0, pinch_end=4.0)
        gestures = _detect_gestures(df, 0.0, 10.0)
        assert len(gestures) == 1
        assert gestures[0]["type"] == "pinch"
        assert gestures[0]["hand"] == "right"
        # Midpoint of pinch should be around t=3.5s
        assert 3.0 <= gestures[0]["t"] <= 4.0

    def test_no_pinch_outside_window(self):
        df = _small_hand_df(duration=10.0, pinch_start=8.0, pinch_end=9.0)
        # Query only first 5s — pinch is outside
        gestures = _detect_gestures(df, 0.0, 5.0)
        assert gestures == []

    def test_no_tip_columns_returns_empty(self):
        df = pd.DataFrame({"t_mono": [0, 1], "session_t": [0, 1], "chirality": ["right", "right"]})
        assert _detect_gestures(df, 0.0, 2.0) == []

    def test_empty_df_returns_empty(self):
        assert _detect_gestures(pd.DataFrame(), 0.0, 5.0) == []

    def test_gesture_t_is_relative_to_start(self):
        df = _small_hand_df(duration=20.0, pinch_start=10.0, pinch_end=11.0)
        gestures = _detect_gestures(df, 8.0, 15.0)
        assert len(gestures) == 1
        # t should be relative to start_sec=8.0
        assert 0.0 < gestures[0]["t"] < 7.0


class TestTranscriptWindow:
    tokens = [
        {"text": "Hello", "startSec": 1.0, "endSec": 1.5},
        {"text": " world", "startSec": 1.5, "endSec": 2.0},
        {"text": " there", "startSec": 5.0, "endSec": 5.5},
    ]

    def test_filters_by_window(self):
        result = _transcript_window(self.tokens, 0.0, 3.0)
        assert result == "Hello world"

    def test_excludes_tokens_outside_window(self):
        result = _transcript_window(self.tokens, 0.0, 1.5)
        assert result == "Hello"

    def test_empty_tokens_returns_empty(self):
        assert _transcript_window([], 0.0, 10.0) == ""

    def test_no_tokens_in_window_returns_empty(self):
        assert _transcript_window(self.tokens, 10.0, 20.0) == ""


class TestBuildTelemetry:
    def test_returns_all_keys(self):
        device_df = _small_device_df()
        hand_df = _small_hand_df()
        object_df = pd.DataFrame()
        result = build_telemetry(device_df, hand_df, object_df, [], [], 0.0, 5.0, 1.0)
        assert set(result.keys()) == {"head_movement", "gestures", "tracked_objects", "transcript"}

    def test_json_serialisable(self):
        import json
        device_df = _small_device_df()
        hand_df = _small_hand_df(pinch_start=2.0, pinch_end=3.0)
        result = build_telemetry(device_df, hand_df, pd.DataFrame(), [], [], 0.0, 5.0, 1.0)
        json.dumps(result)  # should not raise
