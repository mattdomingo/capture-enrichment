"""
Converts raw tracking CSVs into structured JSON telemetry dicts for Gemini prompts.

Time alignment:
    t_mono in the tracking CSVs is session-relative (starts near 0 at recording start),
    verified by t_mono ≈ t_wall - start_wall. It is NOT device absolute uptime.
    session_t = t_mono   (used directly)

Head movement labels (per time bucket):
    static  — displacement < 0.01 m
    gentle  — 0.01 – 0.10 m
    active  — > 0.10 m

Pinch detection:
    Finds thumbTip and indexFingerTip position columns at first load (regex).
    Pinch = thumb-to-index distance < 0.02 m for ≥ 3 consecutive frames.
    Emits one event at the midpoint of each pinch run.
"""

import re

import numpy as np
import pandas as pd

from .models import CapturePackage, TrackedObject

# ── Column detection ─────────────────────────────────────────────────────────

_THUMB_PAT = re.compile(r"^thumbTip_p[xyz]$", re.IGNORECASE)
_INDEX_PAT = re.compile(r"^indexFingerTip_p[xyz]$", re.IGNORECASE)

_PINCH_THRESHOLD_M = 0.02
_PINCH_MIN_FRAMES = 3


def _find_tip_cols(columns: list[str]) -> tuple[list[str], list[str]]:
    """Return (thumb_xyz_cols, index_xyz_cols) sorted by axis (px, py, pz)."""
    thumb = sorted([c for c in columns if _THUMB_PAT.match(c)], key=lambda c: c[-1])
    index = sorted([c for c in columns if _INDEX_PAT.match(c)], key=lambda c: c[-1])
    return thumb, index


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_device_pose(pkg: CapturePackage) -> pd.DataFrame:
    df = pd.read_csv(pkg.device_pose_csv)
    df["session_t"] = df["t_mono"]   # t_mono is already session-relative
    return df


def load_hand_pose(pkg: CapturePackage) -> pd.DataFrame:
    df = pd.read_csv(pkg.hand_pose_world_csv)
    df = df.copy()  # de-fragment the wide DataFrame (222+ cols) before adding a column
    df["session_t"] = df["t_mono"]
    return df


def load_object_pose(pkg: CapturePackage) -> pd.DataFrame:
    df = pd.read_csv(pkg.object_pose_csv)
    if not df.empty:
        df["session_t"] = df["t_mono"]
    return df


# ── Per-window builders ──────────────────────────────────────────────────────


def _head_movement(df: pd.DataFrame, start_sec: float, end_sec: float, resolution_sec: float) -> list[dict]:
    """Per-bucket head displacement from device_pose."""
    window = df[(df["session_t"] >= start_sec) & (df["session_t"] < end_sec)].copy()
    if window.empty:
        return []

    buckets = np.arange(start_sec, end_sec, resolution_sec)
    result = []
    for t in buckets:
        bucket = window[(window["session_t"] >= t) & (window["session_t"] < t + resolution_sec)]
        if len(bucket) < 2:
            continue
        dx = bucket["x"].diff().dropna()
        dy = bucket["y"].diff().dropna()
        dz = bucket["z"].diff().dropna()
        displacement = float(np.sqrt(dx**2 + dy**2 + dz**2).sum())
        if displacement < 0.01:
            label = "static"
        elif displacement < 0.10:
            label = "gentle"
        else:
            label = "active"
        result.append({"t": round(t - start_sec, 3), "displacement_m": round(displacement, 4), "label": label})
    return result


def _detect_gestures(df: pd.DataFrame, start_sec: float, end_sec: float) -> list[dict]:
    """Detect pinch events from hand_pose_world within the given window."""
    if df.empty:
        return []

    cols = list(df.columns)
    thumb_cols, index_cols = _find_tip_cols(cols)
    if len(thumb_cols) < 3 or len(index_cols) < 3:
        return []

    window = df[(df["session_t"] >= start_sec) & (df["session_t"] < end_sec)].copy()
    if window.empty:
        return []

    events = []
    for chirality, hand_df in window.groupby("chirality", sort=False):
        hand_df = hand_df.sort_values("session_t").reset_index(drop=True)
        tx, ty, tz = hand_df[thumb_cols[0]], hand_df[thumb_cols[1]], hand_df[thumb_cols[2]]
        ix, iy, iz = hand_df[index_cols[0]], hand_df[index_cols[1]], hand_df[index_cols[2]]
        dist = np.sqrt((tx - ix)**2 + (ty - iy)**2 + (tz - iz)**2)

        in_pinch = False
        run_start = 0
        for i, d in enumerate(dist):
            if d < _PINCH_THRESHOLD_M:
                if not in_pinch:
                    in_pinch = True
                    run_start = i
            else:
                if in_pinch and (i - run_start) >= _PINCH_MIN_FRAMES:
                    mid_idx = (run_start + i) // 2
                    t_abs = float(hand_df["session_t"].iloc[mid_idx])
                    events.append({
                        "t": round(t_abs - start_sec, 3),
                        "hand": str(chirality).lower(),
                        "type": "pinch",
                    })
                in_pinch = False
        # Close an open run at end of window
        if in_pinch and (len(dist) - run_start) >= _PINCH_MIN_FRAMES:
            mid_idx = (run_start + len(dist)) // 2
            t_abs = float(hand_df["session_t"].iloc[mid_idx])
            events.append({
                "t": round(t_abs - start_sec, 3),
                "hand": str(chirality).lower(),
                "type": "pinch",
            })

    return sorted(events, key=lambda e: e["t"])


def _object_positions(
    df: pd.DataFrame,
    tracked_objects: list[TrackedObject],
    start_sec: float,
    end_sec: float,
    resolution_sec: float,
) -> list[dict]:
    """Per-tracked-object, per-bucket averaged world position."""
    if df.empty or not tracked_objects:
        return []

    obj_map = {obj.id: obj.name for obj in tracked_objects}
    result = []
    window = df[(df["session_t"] >= start_sec) & (df["session_t"] < end_sec)]

    for anchor_id, name in obj_map.items():
        obj_window = window[window["anchorID"] == anchor_id]
        if obj_window.empty:
            continue
        buckets = np.arange(start_sec, end_sec, resolution_sec)
        positions = []
        for t in buckets:
            bucket = obj_window[
                (obj_window["session_t"] >= t) & (obj_window["session_t"] < t + resolution_sec)
            ]
            if bucket.empty:
                continue
            positions.append({
                "t": round(t - start_sec, 3),
                "x": round(float(bucket["x"].mean()), 4),
                "y": round(float(bucket["y"].mean()), 4),
                "z": round(float(bucket["z"].mean()), 4),
            })
        if positions:
            result.append({"name": name, "positions": positions})
    return result


def _transcript_window(tokens: list[dict], start_sec: float, end_sec: float) -> str:
    """Filter transcript tokens by time window; join into a single string."""
    words = [t["text"] for t in tokens if start_sec <= t["startSec"] < end_sec]
    return "".join(words).strip()


# ── Public API ───────────────────────────────────────────────────────────────


def build_telemetry(
    device_df: pd.DataFrame,
    hand_df: pd.DataFrame,
    object_df: pd.DataFrame,
    tokens: list[dict],
    tracked_objects: list[TrackedObject],
    start_sec: float,
    end_sec: float,
    resolution_sec: float,
) -> dict:
    """
    Assemble the structured telemetry JSON dict for a chunk window.

    All timestamps inside the dict are relative to start_sec (i.e. t=0 means start of chunk).
    """
    return {
        "head_movement": _head_movement(device_df, start_sec, end_sec, resolution_sec),
        "gestures": _detect_gestures(hand_df, start_sec, end_sec),
        "tracked_objects": _object_positions(object_df, tracked_objects, start_sec, end_sec, resolution_sec),
        "transcript": _transcript_window(tokens, start_sec, end_sec),
    }
