# Telemetry

Telemetry refers to the structured sensor data captured alongside the video during an AVP recording session. It is assembled per-chunk and injected into the Gemini Pass 1 prompt as a JSON context block, giving the model grounded signal to correlate with what it sees in the video.

All telemetry processing lives in `capture_enrichment/telemetry.py`.

---

## Sources

### `device_pose.csv` — head movement

Each row is a timestamped head pose sample (`t_mono`, `x`, `y`, `z`, orientation quaternion). The telemetry builder divides the chunk into time buckets (default: 1s, configurable via `TELEMETRY_RESOLUTION_SEC`) and computes total displacement within each bucket.

Displacement is labeled:

| Label | Displacement |
|---|---|
| `static` | < 0.01 m |
| `gentle` | 0.01 – 0.10 m |
| `active` | > 0.10 m |

**Prompt field: `head_movement`**
```json
"head_movement": [
  { "t": 0.0, "displacement_m": 0.003, "label": "static" },
  { "t": 1.0, "displacement_m": 0.047, "label": "gentle" },
  { "t": 2.0, "displacement_m": 0.134, "label": "active" }
]
```
`t` is relative to the start of the chunk (not the session).

---

### `hand_pose_world.csv` — gestures

Each row is a timestamped world-space position for a single hand joint, for a given chirality (`left` / `right`). The file is wide — 222+ columns covering all tracked joints.

Pinch detection looks specifically at `thumbTip` and `indexFingerTip` position columns (found by regex at load time). A pinch is registered when the Euclidean distance between these two joints is below 20mm for at least 3 consecutive frames. One event is emitted at the midpoint of each pinch run.

**Prompt field: `gestures`**
```json
"gestures": [
  { "t": 4.2, "hand": "right", "type": "pinch" },
  { "t": 9.7, "hand": "left",  "type": "pinch" }
]
```
`t` is relative to the start of the chunk.

---

### `object_pose.csv` — tracked objects

Each row is a timestamped world-space position for a tracked object anchor, identified by `anchorID`. The set of tracked objects and their human-readable names comes from `metadata.json` (`tracked_objects` array).

For each tracked object, positions are averaged per time bucket and included in the prompt. Objects with no data in the chunk window are omitted entirely.

**Prompt field: `tracked_objects`**
```json
"tracked_objects": [
  {
    "name": "Blue Cube",
    "positions": [
      { "t": 0.0, "x": 0.1234, "y": 0.8801, "z": -0.4412 },
      { "t": 1.0, "x": 0.1251, "y": 0.8799, "z": -0.4408 }
    ]
  }
]
```

---

### `timecoded_transcript.json` — speech

A JSON array of transcript segments, each containing a `tokens` array with per-word `text`, `startSec`, and `endSec`. The telemetry builder filters tokens whose `startSec` falls within the chunk window and joins them into a plain string.

**Prompt field: `transcript`**
```json
"transcript": "open files app show me last week"
```

Returns an empty string if no speech occurred in the chunk.

---

## How it reaches Gemini

`build_telemetry()` assembles all four sources into a single dict:

```python
{
    "head_movement": [...],
    "gestures": [...],
    "tracked_objects": [...],
    "transcript": "..."
}
```

This dict is JSON-serialised and embedded directly in the Pass 1 prompt alongside the video chunk:

```
You are analyzing a segment of an Apple Vision Pro recording.
Segment: 00:00:00–00:00:30 within the full session.

Telemetry for this segment (JSON):
{
  "head_movement": [...],
  "gestures": [...],
  ...
}

Identify all discrete user actions visible in this video segment...
```

Gemini receives both the video frames and the sensor context in the same call. The intent is that corroborating signals — a pinch in `gestures` that coincides with a visible hand interaction in the video, or `active` head movement during a spatial scan — increase the reliability and specificity of the annotations Gemini produces.

Telemetry is **not** sent to Gemini Pass 2 (chapter segmentation). By that stage the events are already annotated; Pass 2 only receives the flat event list as text.

---

## Time alignment

All timestamps in `device_pose.csv`, `hand_pose_world.csv`, and `object_pose.csv` use `t_mono` — a monotonic clock value that is session-relative (approximately 0 at recording start). It is not device absolute uptime.

The telemetry builder uses `t_mono` directly as `session_t`. Inside the assembled telemetry dict, all `t` values are re-expressed relative to the chunk start (`t - chunk_start_sec`), so they are always 0-based regardless of where in the session the chunk falls. This matches how Gemini is told to interpret them in the prompt.
