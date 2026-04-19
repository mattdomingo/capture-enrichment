# capture-enrichment

Ingests Apple Vision Pro capture packages and produces structured, timestamped activity annotations — events grouped into semantic chapters, each with a thumbnail frame extracted from the source video.

```
uv run python -m capture_enrichment.handler \
  --input /path/to/session.capture \
  --telemetry-resolution 5.0
```

`uv run python tools/serve_viewer.py`

Output written to `<session_id>_result/` in the current directory.

---

## What it does

Given a `.capture` directory from an AVP recording session (video + sensor telemetry + spatial tracking data), the service:

1. Downsamples the video for efficient Gemini analysis
2. Breaks the session into overlapping chunks and runs Gemini **Pass 1** — event annotation — on each chunk in parallel with structured telemetry context
3. Deduplicates events at chunk boundaries
4. Runs Gemini **Pass 2** — semantic chapter segmentation — grouping all events into labeled phases
5. Extracts a JPEG thumbnail from the source video at each event's timecode
6. Writes a self-contained result folder

---

## Output

```
<session_id>_result/
├── result.json
└── thumbnails/
    ├── ch1_pt1.jpg
    ├── ch1_pt2.jpg
    ├── ch2_pt1.jpg
    └── ...
```

`thumbnail_path` in `result.json` is relative to the result folder root, so the folder is portable.

**`result.json` schema:**
```json
{
  "session_id": "D834FA49-D935-4FAA-A4BA-C816E3C162AC",
  "duration_seconds": 312.4,
  "processed_at": "2026-04-15T17:00:00Z",
  "chapters": [
    {
      "id": 1,
      "title": "Device calibration",
      "start_ts": "00:00:00",
      "end_ts": "00:01:32",
      "events": [
        {
          "timestamp": "00:00:04",
          "action": "User focuses on calibration target",
          "confidence": 0.92,
          "thumbnail_path": "thumbnails/ch1_pt1.jpg"
        },
        {
          "timestamp": "00:00:18",
          "action": "Pinch gesture detected, target confirmed",
          "confidence": 0.88,
          "thumbnail_path": "thumbnails/ch1_pt2.jpg"
        }
      ]
    }
  ]
}
```

---

## Architecture

### Full pipeline

```
.capture directory
        │
        ▼
┌───────────────────┐
│      ingest       │  Resolves and validates all paths inside the capture
│   (ingest.py)     │  package. Selects camera_sbs.mov → camera_left.mov
└───────────────────┘  fallback. Parses metadata + tracked object manifest.
        │
        ├── video (camera_sbs.mov or camera_left.mov)
        ├── tracking/device_pose.csv
        ├── tracking/hand_pose_world.csv
        ├── tracking/object_pose.csv
        ├── transcripts/timecoded_transcript.json
        └── metadata/metadata.json
        │
        ▼
┌───────────────────┐
│   video prep      │  FFmpeg: downsample to 480p @ 5fps (configurable).
│   (video.py)      │  Writes downsampled.mov to a temp directory.
└───────────────────┘  Plans overlapping 30s chunks with 5s overlap.
        │
        ▼  for each chunk
┌───────────────────────────────────────────────┐
│              Pass 1 — annotation              │
│                                               │
│  video.py    Extract chunk via FFmpeg         │
│              (stream copy, no re-encode)      │
│                                               │
│  telemetry   Build structured telemetry dict: │
│  .py         · head_movement (per-bucket      │
│                displacement, static/gentle/   │
│                active labels)                 │
│              · gestures (pinch events:        │
│                thumb-to-index < 20mm for      │
│                ≥ 3 consecutive frames)        │
│              · tracked_objects (per-bucket    │
│                averaged world positions)      │
│              · transcript (speech tokens      │
│                aligned to time window)        │
│                                               │
│  annotate    Upload chunk to Gemini Files API │
│  .py         Poll until ACTIVE (≤60s)         │
│              generate_content(video + prompt) │
│              → parse JSON event array         │
│              Delete uploaded file (always)    │
│              Retry up to 3× on 429,           │
│              honouring retryDelay hint        │
└───────────────────────────────────────────────┘
        │  list[Event] per chunk
        ▼
┌───────────────────┐
│   deduplication   │  Merge all chunk events, sort by timestamp.
│   (segment.py)    │  Within a 3s sliding window, keep the higher-
└───────────────────┘  confidence event and discard the other.
        │  deduplicated list[Event]
        ▼
┌───────────────────┐
│  Pass 2 —         │  Single Gemini call: group all events into 3–8
│  segmentation     │  logical chapters. Chapters must be contiguous,
│  (segment.py)     │  cover the full session, and assign every event
└───────────────────┘  to exactly one chapter. Falls back to a single
        │              catch-all chapter on parse failure.
        │  list[Chapter]
        ▼
┌───────────────────┐
│  thumbnail        │  For each chapter event, FFmpeg seeks to the
│  extraction       │  event's timecode in downsampled.mov and writes
│  (video.py)       │  a JPEG. Named ch{chapter_id}_pt{order}.jpg
└───────────────────┘  (1-indexed, relative paths stored in JSON).
        │
        ▼
┌───────────────────┐
│  output           │  Writes <session_id>_result/ to cwd:
│  (handler.py)     │  · result.json
└───────────────────┘  · thumbnails/ch*_pt*.jpg
```

### Module map

| Module | Responsibility |
|---|---|
| `handler.py` | CLI entrypoint, pipeline orchestration, output writing |
| `ingest.py` | Capture package resolution and validation |
| `video.py` | FFmpeg wrappers: downsample, chunk extract, thumbnail extract |
| `telemetry.py` | CSV → structured JSON for Gemini prompts |
| `annotate.py` | Gemini Pass 1: per-chunk event annotation |
| `segment.py` | Cross-chunk dedup + Gemini Pass 2 chapter segmentation |
| `models.py` | Pydantic models: `Event`, `Chapter`, `EnrichmentResult` |
| `config.py` | `pydantic-settings` config, reads from env / `.env` |

---

## Input package format

```
session.capture/
├── video/
│   ├── camera_sbs.mov          # side-by-side stereo (preferred)
│   └── camera_left.mov         # fallback if SBS not present
├── tracking/
│   ├── device_pose.csv         # head position + orientation (t_mono)
│   ├── hand_pose_world.csv     # per-joint world-space hand positions
│   └── object_pose.csv         # tracked object world positions
├── transcripts/
│   └── timecoded_transcript.json
└── metadata/
    └── metadata.json           # session ID, duration, tracked_objects list
```

All tracking timestamps are session-relative (`t_mono ≈ 0` at recording start).

---

## Tech stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.14 | ML ecosystem, Lambda-compatible |
| Packaging | [uv](https://github.com/astral-sh/uv) | Fast, reproducible |
| Video processing | FFmpeg via `ffmpeg-python` | Downsample, chunk, seek-extract |
| Telemetry | pandas + numpy | CSV windowing, gesture detection |
| Vision model | Gemini 2.5 Flash | Native video understanding, temporal reasoning |
| Config | pydantic-settings | Typed env vars, `.env` support |
| Lambda deploy *(Phase 2)* | Container image | FFmpeg + ML deps exceed zip size limit |
| Orchestration *(Phase 2)* | AWS Step Functions | Parallel chunk processing for long sessions |
| Storage *(Phase 2)* | S3 | Capture staging + result persistence |
| IaC *(Phase 2)* | AWS SAM | Lambda + API Gateway + Step Functions |

---

## Development

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- FFmpeg (`brew install ffmpeg` on macOS)
- A Google AI Studio API key with Gemini access

### Setup

```bash
uv sync
cp .env.example .env
# set GEMINI_API_KEY in .env
```

### Run locally

```bash
uv run python -m capture_enrichment.handler \
  --input /path/to/session.capture \
  --telemetry-resolution 5.0
```

### Test

```bash
uv run pytest
```

### Viewer

A lightweight static UI renders thumbnails and annotations side by side:

```bash
uv run python tools/serve_viewer.py
```

This indexes every `*_result/` folder in the cwd into `sessions.json`, starts a local server on `http://127.0.0.1:8000/viewer.html`, and opens it in your browser. Pick a session from the dropdown, scroll chapters in the sidebar, click a thumbnail to open the lightbox.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `CHUNK_DURATION_SEC` | `30` | Video chunk size fed to Gemini |
| `CHUNK_OVERLAP_SEC` | `5` | Overlap between adjacent chunks |
| `VIDEO_DOWNSAMPLE_FPS` | `5` | Frame rate for analysis video |
| `VIDEO_DOWNSAMPLE_HEIGHT` | `480` | Resolution (px) for analysis video |
| `TELEMETRY_RESOLUTION_SEC` | `1.0` | Telemetry bucket size (overridable via CLI) |
| `S3_RESULTS_BUCKET` | `""` | *(Phase 2)* S3 bucket for output |

---

## Roadmap

### Phase 2 — Lambda deployment

The `lambda_handler` stub in `handler.py` is the entry point for AWS deployment. The plan:

```
S3 upload trigger  ──►  capture-enrichment Lambda (container image)
                              │
                              ├── pull .capture.zip from S3
                              ├── run pipeline
                              ├── write result.json + thumbnails/ to S3
                              └── return S3 key in response
```

For sessions longer than ~15 minutes, chunk annotation is parallelised via **AWS Step Functions**: a Map state fans out one Lambda invocation per chunk, collects results, then runs dedup + segmentation in a final aggregation invocation.

### Potential improvements

- **Parallel chunk annotation** — even locally, Pass 1 chunks are independent and could run concurrently via `asyncio` + the async Gemini client, cutting wall-clock time significantly on long sessions
- **Prompt-side dedup** — instruct Gemini not to annotate the final `overlap_sec` of non-terminal chunks, eliminating the timestamp-window heuristic entirely
- **Sub-second thumbnail precision** — Pass 1 currently returns `HH:MM:SS`; requesting millisecond timestamps would let thumbnail seeks land on the exact frame Gemini identified
- **Streaming output** — emit chapter JSON as each chapter is confirmed rather than buffering the full result, useful for real-time UI integration

---

## Inspired by

[SentrySearch](https://github.com/ssrajadh/sentrysearch) — semantic video search engine whose chunk-based Gemini processing patterns informed this service's design.
