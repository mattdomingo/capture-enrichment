"""Tests for video.py — chunk planning and video selection logic."""

import pytest

from capture_enrichment.video import plan_chunks


class TestPlanChunks:
    def test_single_chunk_short_video(self):
        chunks = plan_chunks(24.0, chunk_sec=30, overlap_sec=5)
        assert chunks == [(0.0, 24.0)]

    def test_single_chunk_exact_fit(self):
        chunks = plan_chunks(30.0, chunk_sec=30, overlap_sec=5)
        assert chunks == [(0.0, 30.0)]

    def test_multiple_chunks(self):
        chunks = plan_chunks(90.0, chunk_sec=30, overlap_sec=5)
        assert chunks == [(0.0, 30.0), (25.0, 55.0), (50.0, 80.0), (75.0, 90.0)]

    def test_last_chunk_ends_at_duration(self):
        chunks = plan_chunks(65.0, chunk_sec=30, overlap_sec=5)
        assert chunks[-1][1] == 65.0

    def test_all_chunks_start_before_end(self):
        for chunk in plan_chunks(120.0, 30, 5):
            assert chunk[0] < chunk[1]

    def test_step_equals_chunk_minus_overlap(self):
        chunks = plan_chunks(100.0, chunk_sec=20, overlap_sec=4)
        starts = [c[0] for c in chunks]
        steps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
        assert all(s == 16.0 for s in steps)

    def test_zero_duration_returns_empty_or_single(self):
        # Edge case: duration <= 0 should return nothing or one degenerate chunk
        chunks = plan_chunks(0.0, 30, 5)
        assert chunks == []
