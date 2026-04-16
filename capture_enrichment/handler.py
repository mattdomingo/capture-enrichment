"""
CLI entrypoint and Lambda handler stub.

CLI usage:
    uv run python -m capture_enrichment.handler \\
        --input /path/to/session.capture \\
        [--telemetry-resolution 5.0]

Output (written to cwd):
    <session_id>_result/
        result.json
        thumbnails/
            <chapter_id>_<action_order>.jpg   (one per event, 1-indexed)
"""

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from .annotate import annotate_chunk, create_client
from .config import Config
from .ingest import load_capture_package, load_metadata, load_transcript_tokens
from .models import EnrichmentResult, VideoChunk
from .segment import deduplicate_events, segment_into_chapters
from .telemetry import build_telemetry, load_device_pose, load_hand_pose, load_object_pose
from .video import downsample_video, extract_chunk, extract_thumbnail, get_video_duration, plan_chunks, select_video

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input: Path = typer.Option(..., "--input", "-i", help="Path to .capture directory"),
    telemetry_resolution: Optional[float] = typer.Option(
        None,
        "--telemetry-resolution",
        help="Telemetry bucket size in seconds (overrides TELEMETRY_RESOLUTION_SEC)",
    ),
) -> None:
    cfg = Config()
    if telemetry_resolution is not None:
        cfg = cfg.model_copy(update={"telemetry_resolution_sec": telemetry_resolution})

    result_dir = process_capture(input, cfg)
    typer.echo(f"Output written to {result_dir}", err=True)


def process_capture(capture_path: Path, cfg: Config) -> Path:
    """
    Main pipeline:
      1. Ingest capture package
      2. Load tracking DataFrames
      3. Downsample video → temp dir
      4. Plan chunks
      5. For each chunk: extract video, build telemetry, annotate (Gemini Pass 1)
      6. Deduplicate events across overlapping chunks
      7. Segment into chapters (Gemini Pass 2)
      8. Extract thumbnails, named <chapter_id>_<action_order>.jpg
      9. Write <session_id>_result/ to cwd
     10. Return the result directory path
    """
    typer.echo(f"Loading capture package: {capture_path}", err=True)
    pkg = load_capture_package(capture_path)
    meta = load_metadata(pkg)
    tokens = load_transcript_tokens(pkg)

    session_id: str = meta["id"]
    metadata_duration: float = float(meta["duration"])

    typer.echo("Loading tracking data...", err=True)
    device_df = load_device_pose(pkg)
    hand_df = load_hand_pose(pkg)
    object_df = load_object_pose(pkg)

    client = create_client(cfg.gemini_api_key)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        typer.echo(f"Downsampling video ({cfg.video_downsample_height}p @ {cfg.video_downsample_fps}fps)...", err=True)
        downsampled = downsample_video(
            select_video(pkg),
            tmp / "downsampled.mov",
            fps=cfg.video_downsample_fps,
            height=cfg.video_downsample_height,
        )
        duration = get_video_duration(downsampled)
        typer.echo(f"Video duration: {duration:.1f}s", err=True)

        chunk_ranges = plan_chunks(duration, cfg.chunk_duration_sec, cfg.chunk_overlap_sec)
        typer.echo(f"Processing {len(chunk_ranges)} chunk(s)...", err=True)

        chunk_results: list[tuple[float, list]] = []
        for i, (start, end) in enumerate(chunk_ranges, 1):
            typer.echo(f"  Chunk {i}/{len(chunk_ranges)}: {start:.0f}s–{end:.0f}s", err=True)

            chunk_path = extract_chunk(downsampled, tmp / f"chunk_{i:03d}.mov", start, end)
            telemetry = build_telemetry(
                device_df,
                hand_df,
                object_df,
                tokens,
                pkg.tracked_objects,
                start,
                end,
                cfg.telemetry_resolution_sec,
            )
            chunk = VideoChunk(
                start_sec=start,
                end_sec=end,
                video_path=chunk_path,
                telemetry=telemetry,
            )
            events = annotate_chunk(client, chunk)
            typer.echo(f"    → {len(events)} event(s)", err=True)
            chunk_results.append((start, events))

        typer.echo("Deduplicating events...", err=True)
        all_events = deduplicate_events(chunk_results)
        typer.echo(f"  {len(all_events)} unique event(s)", err=True)

        typer.echo("Segmenting into chapters...", err=True)
        chapters = segment_into_chapters(client, all_events, duration)
        typer.echo(f"  {len(chapters)} chapter(s)", err=True)

        # Thumbnail extraction happens after segmentation so names reflect chapter/order,
        # and so thumbnail_path is absent from the Gemini segmentation prompt.
        result_dir = Path(f"{session_id}_result")
        thumbnails_subdir = result_dir / "thumbnails"
        thumbnails_subdir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Extracting thumbnails → {thumbnails_subdir}", err=True)
        for chapter in chapters:
            for order, event in enumerate(chapter.events, start=1):
                h, m, s = event.timestamp.split(":")
                offset_sec = int(h) * 3600 + int(m) * 60 + int(s)
                filename = f"{chapter.id}_{order}.jpg"
                extract_thumbnail(downsampled, thumbnails_subdir / filename, offset_sec)
                event.thumbnail_path = f"thumbnails/{filename}"

    result = EnrichmentResult(
        session_id=session_id,
        duration_seconds=metadata_duration,
        processed_at=datetime.now(timezone.utc).isoformat(),
        chapters=chapters,
    )
    (result_dir / "result.json").write_text(result.model_dump_json(indent=2))
    return result_dir


# ── Lambda handler stub (Phase 2) ────────────────────────────────────────────


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    AWS Lambda entrypoint (Phase 2).

    Expected event shape:
        {"capture_path": "s3://bucket/key.capture.zip"}
        or
        {"Records": [{"s3": {"bucket": {"name": "..."}, "object": {"key": "..."}}}]}
    """
    raise NotImplementedError("Lambda handler not yet implemented (Phase 2)")


if __name__ == "__main__":
    app()
