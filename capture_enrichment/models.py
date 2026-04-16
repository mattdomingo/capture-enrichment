from pathlib import Path

from pydantic import BaseModel


# ── Output schema (matches readme) ──────────────────────────────────────────


class Event(BaseModel):
    timestamp: str   # "HH:MM:SS" absolute session time
    action: str
    confidence: float
    thumbnail_path: str | None = None


class Chapter(BaseModel):
    id: int
    title: str
    start_ts: str
    end_ts: str
    events: list[Event]


class EnrichmentResult(BaseModel):
    session_id: str
    duration_seconds: float
    processed_at: str   # ISO 8601
    chapters: list[Chapter]


# ── Internal types ───────────────────────────────────────────────────────────


class TrackedObject(BaseModel):
    id: str
    name: str


class CapturePackage(BaseModel):
    root: Path
    video_path: Path            # camera_sbs.mov if present, else camera_left.mov
    metadata_json: Path
    device_pose_csv: Path
    hand_pose_world_csv: Path
    object_pose_csv: Path
    transcript_json: Path
    tracked_objects: list[TrackedObject]


class VideoChunk(BaseModel):
    start_sec: float
    end_sec: float
    video_path: Path
    telemetry: dict             # structured JSON ready for Gemini prompt
