"""
Gemini Pass 2 — overlap deduplication and chapter segmentation.

Deduplication strategy:
    Events from overlapping chunks are merged and sorted by timestamp.
    For any two events within `window_sec` of each other, the lower-confidence
    one is dropped. This is a timestamp-only approach — two genuinely different
    actions within 3s of each other may be incorrectly collapsed.

    Future improvement: prompt-side dedup — instruct Gemini not to annotate
    the last `overlap_sec` seconds of each non-final chunk. This eliminates
    the ambiguity but requires Gemini to be aware of the overlap boundary,
    complicating the per-chunk prompt.
"""

import json

import google.genai as genai

from .annotate import _parse_events, _sec_to_ts
from .models import Chapter, Event

_MODEL = "gemini-2.0-flash"


def _ts_to_sec(ts: str) -> float:
    """Convert HH:MM:SS to total seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    return float(ts)


def deduplicate_events(
    chunk_events: list[tuple[float, list[Event]]],
    window_sec: float = 3.0,
) -> list[Event]:
    """
    Merge events from all chunks into a deduplicated, time-sorted list.

    Args:
        chunk_events: list of (chunk_start_sec, events) from each chunk.
        window_sec: two events within this many seconds of each other
                    are considered duplicates; the lower-confidence one is dropped.
    """
    # Flatten all events
    all_events = [event for _, events in chunk_events for event in events]

    # Sort by absolute timestamp
    all_events.sort(key=lambda e: _ts_to_sec(e.timestamp))

    # Sliding dedup: walk through and suppress events too close to the previous kept one
    kept: list[Event] = []
    for event in all_events:
        t = _ts_to_sec(event.timestamp)
        if not kept:
            kept.append(event)
            continue
        prev = kept[-1]
        prev_t = _ts_to_sec(prev.timestamp)
        if t - prev_t < window_sec:
            # Within dedup window — keep the higher-confidence one
            if event.confidence > prev.confidence:
                kept[-1] = event
        else:
            kept.append(event)

    return kept


def _build_segmentation_prompt(events: list[Event], duration_sec: float) -> str:
    end_ts = _sec_to_ts(duration_sec)
    events_json = json.dumps([e.model_dump() for e in events], indent=2)

    return f"""You are analyzing a completed Apple Vision Pro recording session (duration: {end_ts}).

All detected events in chronological order:
{events_json}

Group these events into 3 to 8 logical chapters representing cohesive phases of activity.
Examples of good chapter titles: "App browsing", "Spatial anchor placement", "Video playback",
"Object scanning", "Calibration".

Chapter boundaries should reflect significant shifts in:
- Application or environment context
- Task type (browsing vs. creating vs. configuring vs. scanning)
- User attention or physical location

Rules:
- Chapters must be contiguous and together cover the full session
- First chapter starts at 00:00:00, last chapter ends at {end_ts}
- Every event must belong to exactly one chapter
- If there are very few events, 1–2 chapters is acceptable

Respond ONLY with a valid JSON array (no markdown fences):
[
  {{
    "id": 1,
    "title": "Short descriptive title",
    "start_ts": "HH:MM:SS",
    "end_ts": "HH:MM:SS",
    "events": [
      {{"timestamp": "HH:MM:SS", "action": "...", "confidence": 0.92}}
    ]
  }}
]"""


def segment_into_chapters(
    client: genai.Client,
    events: list[Event],
    duration_sec: float,
) -> list[Chapter]:
    """
    Single Gemini call: group all events into semantic chapters.
    Returns a list with one catch-all chapter if parsing fails.
    """
    if not events:
        return [Chapter(
            id=1,
            title="Session",
            start_ts="00:00:00",
            end_ts=_sec_to_ts(duration_sec),
            events=[],
        )]

    prompt = _build_segmentation_prompt(events, duration_sec)
    response = client.models.generate_content(model=_MODEL, contents=prompt)
    return _parse_chapters(response.text, events, duration_sec)


def _parse_chapters(text: str, events: list[Event], duration_sec: float) -> list[Chapter]:
    """
    Parse chapter JSON from Gemini response.
    Falls back to a single chapter containing all events if parsing fails.
    """
    if not text:
        return _fallback_chapter(events, duration_sec)

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(stripped)
        if not isinstance(data, list) or not data:
            return _fallback_chapter(events, duration_sec)

        chapters = []
        for item in data:
            chapter_events = []
            for ev in item.get("events", []):
                if isinstance(ev, dict) and "timestamp" in ev and "action" in ev:
                    chapter_events.append(Event(
                        timestamp=str(ev["timestamp"]),
                        action=str(ev["action"]),
                        confidence=float(ev.get("confidence", 0.5)),
                    ))
            chapters.append(Chapter(
                id=int(item.get("id", len(chapters) + 1)),
                title=str(item.get("title", "Untitled")),
                start_ts=str(item.get("start_ts", "00:00:00")),
                end_ts=str(item.get("end_ts", _sec_to_ts(duration_sec))),
                events=chapter_events,
            ))
        return chapters
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return _fallback_chapter(events, duration_sec)


def _fallback_chapter(events: list[Event], duration_sec: float) -> list[Chapter]:
    return [Chapter(
        id=1,
        title="Session",
        start_ts="00:00:00",
        end_ts=_sec_to_ts(duration_sec),
        events=events,
    )]
