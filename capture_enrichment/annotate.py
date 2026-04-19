"""
Gemini Pass 1 — per-chunk event annotation.

Flow per chunk:
  1. Upload video chunk to Gemini Files API
  2. Poll until state == ACTIVE (timeout 60s)
  3. Call generate_content with video part + structured telemetry prompt
  4. Parse JSON array from response
  5. Delete uploaded file (always, in finally)

Malformed or empty Gemini responses return [] rather than raising.

Rate limit handling:
  429 responses include a retryDelay hint (e.g. "36s"). We honour that hint
  with a small buffer and retry up to _MAX_RETRIES times before re-raising.
  Note: if the quota ceiling itself is 0 (free-tier limit reached or billing
  not enabled), retrying will not help — the caller will see the error re-raised
  after all retries are exhausted.
"""

import json
import re
import time
from pathlib import Path

import google.genai as genai
import google.genai.types as gtypes
from google.genai import errors as genai_errors

from .models import Event, VideoChunk

_MODEL = "gemini-2.5-flash"
_UPLOAD_POLL_INTERVAL_SEC = 2
_UPLOAD_TIMEOUT_SEC = 60
_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY_SEC = 60  # fallback if no retryDelay hint in error


def create_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def upload_video_chunk(client: genai.Client, chunk_path: Path) -> gtypes.File:
    """Upload a video file to the Gemini Files API and wait until it is ACTIVE."""
    uploaded = client.files.upload(file=chunk_path)
    deadline = time.monotonic() + _UPLOAD_TIMEOUT_SEC
    while uploaded.state.value != "ACTIVE":
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

Data source priority — use this hierarchy when deciding whether to annotate an event and how to score confidence:
1. VIDEO (highest weight): Primary ground truth. An event must be visible in video to be annotated.
2. TRANSCRIPT: Speech unambiguously confirms intentional actions.
3. TRACKED OBJECTS: Spatial context for what the user interacted with, not action evidence alone.
4. HEAD MOVEMENT: Indicates attention shifts, not discrete actions.
5. PINCH GESTURES (lowest weight): Corroborating signal only. Do NOT annotate an event solely because \
a pinch was detected in telemetry — the action must be clearly visible in video. A pinch with strong \
visual confirmation warrants normal confidence; a pinch without visible interaction context should be \
omitted entirely or scored ≤ 0.5.

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
- confidence: scored using this rubric:
  - 0.9–1.0: Action is unambiguously visible in video AND corroborated by ≥1 sensor source
  - 0.7–0.89: Action is clearly visible in video with no contradicting signal
  - 0.5–0.69: Action inferred from partial visual evidence, or sensor signal with ambiguous video
  - below 0.5: Do not annotate — omit the event rather than reporting it at low confidence

Respond ONLY with a valid JSON array (no markdown fences):
[{{"timestamp": "HH:MM:SS", "action": "...", "confidence": 0.92}}]
Return [] if no notable events occur in this segment."""


def annotate_chunk(client: genai.Client, chunk: VideoChunk) -> list[Event]:
    """
    Upload the chunk video, run Gemini annotation, return parsed Events.
    Retries up to _MAX_RETRIES times on 429 rate-limit errors, honouring the
    retryDelay hint from the API response. Deletes the uploaded file before returning.
    """
    uploaded = upload_video_chunk(client, chunk.video_path)
    try:
        video_part = gtypes.Part.from_uri(
            file_uri=uploaded.uri,
            mime_type="video/quicktime",
        )
        prompt = _build_prompt(chunk)
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=_MODEL,
                    contents=[video_part, prompt],
                )
                events = _parse_events(response.text)
                return _apply_source_weights(events, chunk.telemetry)
            except genai_errors.ClientError as exc:
                if exc.code != 429 or attempt == _MAX_RETRIES:
                    raise
                last_exc = exc
                delay = _parse_retry_delay(str(exc)) or _DEFAULT_RETRY_DELAY_SEC
                import typer
                typer.echo(
                    f"    429 rate limit — retrying in {delay}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})",
                    err=True,
                )
                time.sleep(delay)
        raise last_exc  # unreachable but satisfies type checker
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass  # best-effort cleanup


_PINCH_ONLY_MULTIPLIER = 0.85
_PINCH_PENALTY_KEYWORDS = frozenset({"pinch", "tap", "select", "press", "click", "choose"})
_PINCH_WINDOW_SEC = 1.0


def _apply_source_weights(events: list[Event], telemetry: dict) -> list[Event]:
    """
    Apply a deterministic confidence penalty to events that appear to be driven
    solely by pinch telemetry with no corroborating transcript or tracked object.

    Penalty conditions (all must be true):
      1. Event timestamp coincides (±_PINCH_WINDOW_SEC) with a pinch gesture
      2. Event action text contains a pinch-related keyword
      3. No transcript speech is present in the chunk
      4. No tracked objects are present in the chunk

    When all conditions are met, confidence is multiplied by _PINCH_ONLY_MULTIPLIER
    and capped at _PINCH_ONLY_MULTIPLIER (0.85).
    """
    gesture_times = [g["t"] for g in telemetry.get("gestures", []) if g.get("type") == "pinch"]
    has_transcript = bool(telemetry.get("transcript", "").strip())
    has_objects = bool(telemetry.get("tracked_objects"))

    # If transcript or objects are present, no penalty applies to any event
    if has_transcript or has_objects:
        return events

    result = []
    for event in events:
        # Convert HH:MM:SS to seconds for comparison
        parts = event.timestamp.split(":")
        try:
            event_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except (IndexError, ValueError):
            result.append(event)
            continue

        action_lower = event.action.lower()
        has_pinch_keyword = any(kw in action_lower for kw in _PINCH_PENALTY_KEYWORDS)
        near_pinch = any(abs(event_sec - gt) <= _PINCH_WINDOW_SEC for gt in gesture_times)

        if has_pinch_keyword and near_pinch:
            penalized = min(event.confidence * _PINCH_ONLY_MULTIPLIER, _PINCH_ONLY_MULTIPLIER)
            result.append(Event(
                timestamp=event.timestamp,
                action=event.action,
                confidence=round(penalized, 4),
                thumbnail_path=event.thumbnail_path,
            ))
        else:
            result.append(event)

    return result


def _parse_retry_delay(error_text: str) -> float | None:
    """Extract retryDelay seconds from a Gemini 429 error message string."""
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+)s", error_text)
    if match:
        return float(match.group(1)) + 2  # small buffer
    return None


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
