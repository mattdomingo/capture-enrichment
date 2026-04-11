import json
from pathlib import Path

from .models import CapturePackage, TrackedObject


def load_capture_package(path: str | Path) -> CapturePackage:
    """
    Resolve and validate all expected paths inside a .capture directory.

    Video selection: uses camera_sbs.mov when present, falls back to camera_left.mov.
    metadata.json world_anchor field handled as either a bare UUID string or {"id": UUID}.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"Capture package path is not a directory: {root}")

    video_dir = root / "video"
    tracking_dir = root / "tracking"
    metadata_dir = root / "metadata"
    transcripts_dir = root / "transcripts"

    sbs = video_dir / "camera_sbs.mov"
    left = video_dir / "camera_left.mov"
    video_path = sbs if sbs.exists() else left

    if not video_path.exists():
        raise FileNotFoundError(f"No usable video file found in {video_dir}")

    metadata_json = metadata_dir / "metadata.json"
    device_pose_csv = tracking_dir / "device_pose.csv"
    hand_pose_world_csv = tracking_dir / "hand_pose_world.csv"
    object_pose_csv = tracking_dir / "object_pose.csv"
    transcript_json = transcripts_dir / "timecoded_transcript.json"

    for p in (metadata_json, device_pose_csv, hand_pose_world_csv, object_pose_csv, transcript_json):
        if not p.exists():
            raise FileNotFoundError(f"Expected file missing: {p}")

    raw = json.loads(metadata_json.read_text())
    tracked_objects = [
        TrackedObject(id=obj["id"], name=obj["name"])
        for obj in raw.get("tracked_objects", [])
    ]

    return CapturePackage(
        root=root,
        video_path=video_path,
        metadata_json=metadata_json,
        device_pose_csv=device_pose_csv,
        hand_pose_world_csv=hand_pose_world_csv,
        object_pose_csv=object_pose_csv,
        transcript_json=transcript_json,
        tracked_objects=tracked_objects,
    )


def load_metadata(pkg: CapturePackage) -> dict:
    """Return parsed metadata.json. world_anchor normalised to a plain UUID string."""
    raw = json.loads(pkg.metadata_json.read_text())

    # Normalise world_anchor: some captures use {"id": UUID}, others a bare string
    wa = raw.get("world_anchor")
    if isinstance(wa, dict):
        raw["world_anchor"] = wa.get("id", "")

    return raw


def load_transcript_tokens(pkg: CapturePackage) -> list[dict]:
    """
    Parse timecoded_transcript.json into a flat list of {text, startSec, endSec} token dicts.
    Returns [] if the transcript is empty or missing.
    """
    try:
        segments = json.loads(pkg.transcript_json.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    tokens: list[dict] = []
    for segment in segments:
        for token in segment.get("tokens", []):
            tokens.append({
                "text": token.get("text", ""),
                "startSec": float(token.get("startSec", 0)),
                "endSec": float(token.get("endSec", 0)),
            })
    return tokens
