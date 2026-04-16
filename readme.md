# capture-enrichment

An AWS Lambda microservice that ingests Apple Vision Pro capture packages and returns structured, timestamped activity annotations — including semantic chapter segmentation.

   uv run python -m capture_enrichment.handler --input /path/to/session.capture --telemetry-resolution 5.0 

## What it does

Given a capture package from an Apple Vision Pro session (video + metadata + CSV telemetry), this service returns a JSON document describing what is happening at each point in the video, grouped into logical chapters.

**Input** — a capture package containing:
- Video file (`.mov` / `.mp4` from visionOS capture)
- Metadata file (device info, session duration, capture settings)
- CSV telemetry (gaze data, hand tracking, spatial anchors, etc.)

**Output** — structured JSON:
```json
{
  "session_id": "abc123",
  "duration_seconds": 312,
  "processed_at": "2026-04-09T17:00:00Z",
  "chapters": [
    {
      "id": 1,
      "title": "Device calibration",
      "start_ts": "00:00:00",
      "end_ts": "00:01:32",
      "events": [
        { "timestamp": "00:00:04", "action": "User focuses on calibration target", "confidence": 0.92 },
        { "timestamp": "00:00:18", "action": "Pinch gesture detected, target confirmed", "confidence": 0.88 }
      ]
    }
  ]
}
```

## Architecture

```
Main Repo
    │
    ▼ HTTP (API Gateway) or S3 trigger
┌─────────────────────────────┐
│   capture-enrichment Lambda │
│   (container image)         │
│                             │
│  1. Unpack capture package  │
│  2. Downsample video        │  ◄── FFmpeg (480p, 5fps)
│  3. Chunk into segments     │  ◄── 30s chunks, 5s overlap
│  4. Analyze via Gemini      │  ◄── Gemini 2.0 Flash (video-native)
│  5. Segment into chapters   │  ◄── LLM-driven semantic grouping
│  6. Return JSON             │
└─────────────────────────────┘
    │
    ▼ JSON written to S3 + returned in response
```

For sessions longer than ~15 minutes, processing is orchestrated via **AWS Step Functions** to parallelize chunk analysis across multiple Lambda invocations.

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | ML ecosystem, Lambda support |
| Packaging | [uv](https://github.com/astral-sh/uv) | Fast, reproducible |
| Lambda deploy | Container image | FFmpeg + ML deps exceed zip limits |
| Video processing | FFmpeg | Frame extraction, downscaling, trimming |
| Vision model | Gemini 2.0 Flash | Native video understanding, temporal reasoning |
| Orchestration | AWS Step Functions | Long-session parallel processing |
| Storage | S3 | Capture package staging + result persistence |
| IaC | AWS SAM | Lambda + API Gateway + Step Functions |

## Inspired by

[SentrySearch](https://github.com/ssrajadh/sentrysearch) — semantic video search engine whose chunk-based video processing and Gemini embedding patterns informed this service's design.

## Development

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker (for building the Lambda container image)
- FFmpeg
- AWS CLI with credentials configured

### Setup
```bash
uv sync
cp .env.example .env
# fill in GEMINI_API_KEY and AWS credentials
```

### Run locally
```bash
uv run python -m capture_enrichment.handler --input path/to/capture_package/
```

### Deploy
```bash
sam build && sam deploy
```

### Test
```bash
uv run pytest
```

## Input Package Format

The service expects a capture package as either:
- A directory containing the video, metadata JSON, and telemetry CSV
- A `.zip` archive of the above, uploaded to S3

```
capture_package/
├── video.mov           # Raw visionOS capture
├── metadata.json       # Session metadata (device, duration, settings)
└── telemetry.csv       # Gaze, hand tracking, spatial anchor data
```

## Chapter Segmentation

The service identifies natural breakpoints in the session — moments where the user's activity or environment changes significantly — and groups events into labeled chapters. Chapter boundaries are identified by prompting Gemini to reason about activity continuity across segments. This produces human-readable chapter titles like "App browsing", "Spatial anchor placement", or "Video playback".

## Environment Variables

See `.env.example` for required configuration.

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |
| `S3_RESULTS_BUCKET` | S3 bucket for output JSON |
| `CHUNK_DURATION_SEC` | Video chunk size in seconds (default: 30) |
| `CHUNK_OVERLAP_SEC` | Overlap between chunks (default: 5) |
| `VIDEO_DOWNSAMPLE_FPS` | Frame rate for analysis (default: 5) |
| `VIDEO_DOWNSAMPLE_HEIGHT` | Resolution for analysis (default: 480) |
