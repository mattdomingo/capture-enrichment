"""
Gemini Pass 1 — per-chunk event annotation.

Flow per chunk:
  1. Upload video chunk to Gemini Files API
  2. Poll until state == ACTIVE (timeout 60s)
  3. Call generate_content with video part + structured telemetry prompt
  4. Parse JSON array from response
  5. Delete uploaded file (always, in finally)

Malformed or empty Gemini responses return [] rather than raising.
"""

import json
import time
from pathlib import Path

import google.genai as genai
import google.genai.types as gtypes

from .models import Event, VideoChunk

_MODEL = "gemini-2.0-flash"
_UPLOAD_POLL_INTERVAL_SEC = 2
_UPLOAD_TIMEOUT_SEC = 60


def create_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def upload_video_chunk(client: genai.Client, chunk_path: Path) -> gtypes.File:
    """Upload a video file to the Gemini Files API and wait until it is ACTIVE."""
    uploaded = client.files.upload(path=str(chunk_path))
    deadline = time.monotonic() + _UPLOAD_TIMEOUT_SEC
    while uploaded.state.name != "ACTIVE":
        if time.monotonic() > deadline:
            raise TimeoutError(f"Gemini file upload did not become ACTIVE within {_UPLOAD_TIMEOUT_SEC}s")
        time.sleep(_UPLOAD_POLL_INTERVAL_SEC)
        uploaded = client.files.get(name=uploaded.name)
    return uploaded


def _build_prompt(chunk: VideoChunk) -> str:
    start_ts = _sec_to_ts(chunk.start_sec)
    end_ts = _sec_to_ts(chunk.end_sec)
    telemetry_json = json.dumps(chunk.telemetry, indent=2)

    return f"""You are analyzing a segment of an Apple Vision Pro recording.
Segment: {start_ts}–{end_ts} within the full session.

Telemetry for this segment (JSON):
{telemetry_json}

Identify all discrete user actions visible in this video segment. Focus on:
- App interactions (launch, navigate, close)
- Gesture inputs (pinch, tap, swipe, grab)
- Gaze-driven selections
- Spatial awareness (looking around, spatial anchors, tracked objects)
- Speech-driven actions

For each event provide:
- timestamp: absolute session time (HH:MM:SS). If the segment starts at {start_ts} \
and an event occurs 10s in, the timestamp is {_sec_to_ts(chunk.start_sec + 10)}.
- action: concise description, max 15 words
- confidence: 0.0–1.0

Respond ONLY with a valid JSON array (no markdown fences):
[{{"timestamp": "HH:MM:SS", "action": "...", "confidence": 0.92}}]
Return [] if no notable events occur in this segment."""


def annotate_chunk(client: genai.Client, chunk: VideoChunk) -> list[Event]:
    """
    Upload the chunk video, run Gemini annotation, return parsed Events.
    Deletes the uploaded file from Gemini's Files API before returning.
    """
    uploaded = upload_video_chunk(client, chunk.video_path)
    try:
        video_part = gtypes.Part.from_uri(
            file_uri=uploaded.uri,
            mime_type="video/quicktime",
        )
        response = client.models.generate_content(
            model=_MODEL,
            contents=[video_part, _build_prompt(chunk)],
        )
        return _parse_events(response.text)
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass  # best-effort cleanup


def _parse_events(text: str) -> list[Event]:
    """
    Parse a JSON array of event dicts from Gemini's response text.
    Strips markdown code fences if present. Returns [] on any parse failure.
    """
    if not text:
        return []
    # Strip ```json ... ``` fences that Gemini sometimes adds despite instructions
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(stripped)
        if not isinstance(data, list):
            return []
        events = []
        for item in data:
            if isinstance(item, dict) and "timestamp" in item and "action" in item:
                events.append(Event(
                    timestamp=str(item["timestamp"]),
                    action=str(item["action"]),
                    confidence=float(item.get("confidence", 0.5)),
                ))
        return events
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def _sec_to_ts(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    s = int(seconds)
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"
