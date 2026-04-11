"""Shared fixtures for the test suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CAPTURE_1 = Path("/Users/matthewdomingo/Dev/holos/2026-03-08T17-35-19-39A90878-5D69-437A-946C-F8B3C65F6E40.capture")
CAPTURE_2 = Path("/Users/matthewdomingo/Dev/holos/2026-03-13T17-38-41-D834FA49-D935-4FAA-A4BA-C816E3C162AC.capture")


def _small_device_df(duration: float = 10.0, hz: float = 50.0) -> pd.DataFrame:
    """Minimal device_pose DataFrame spanning [0, duration] at ~hz."""
    n = int(duration * hz)
    t = np.linspace(0, duration, n)
    return pd.DataFrame({
        "t_mono": t,
        "t_wall": t + 1_700_000_000,
        "x": np.zeros(n),
        "y": np.ones(n) * 1.6,
        "z": np.zeros(n),
        "qx": np.zeros(n), "qy": np.zeros(n), "qz": np.zeros(n), "qw": np.ones(n),
        "session_t": t,
    })


def _small_hand_df(
    duration: float = 10.0,
    hz: float = 100.0,
    chirality: str = "right",
    pinch_start: float = 3.0,
    pinch_end: float = 4.0,
) -> pd.DataFrame:
    """
    Minimal hand_pose DataFrame with realistic thumbTip / indexFingerTip columns.
    Fingers are ~0.10 m apart normally; within pinch window they close to 0.005 m.
    """
    n = int(duration * hz)
    t = np.linspace(0, duration, n)
    pinch_mask = (t >= pinch_start) & (t < pinch_end)

    # Base positions — thumb at (0, 0, 0), index at (0.10, 0, 0)
    thumb_x = np.zeros(n)
    thumb_y = np.zeros(n)
    thumb_z = np.zeros(n)
    index_x = np.where(pinch_mask, 0.005, 0.10)
    index_y = np.zeros(n)
    index_z = np.zeros(n)

    return pd.DataFrame({
        "t_mono": t,
        "t_wall": t + 1_700_000_000,
        "chirality": chirality,
        "thumbTip_px": thumb_x, "thumbTip_py": thumb_y, "thumbTip_pz": thumb_z,
        "thumbTip_qx": np.zeros(n), "thumbTip_qy": np.zeros(n),
        "thumbTip_qz": np.zeros(n), "thumbTip_qw": np.ones(n),
        "indexFingerTip_px": index_x, "indexFingerTip_py": index_y, "indexFingerTip_pz": index_z,
        "indexFingerTip_qx": np.zeros(n), "indexFingerTip_qy": np.zeros(n),
        "indexFingerTip_qz": np.zeros(n), "indexFingerTip_qw": np.ones(n),
        "session_t": t,
    })
