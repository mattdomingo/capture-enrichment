"""
CLI entrypoint and Lambda handler stub.

CLI usage:
    uv run python -m capture_enrichment.handler \\
        --input /path/to/session.capture \\
        [--output result.json] \\
        [--telemetry-resolution 5.0]
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
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write JSON to file (default: stdout)"),
    telemetry_resolution: Optional[float] = typer.Option(
        None,
        "--telemetry-resolution",
        help="Telemetry bucket size in seconds (overrides TELEMETRY_RESOLUTION_SEC)",
    ),
    thumbnails_dir: Optional[Path] = typer.Option(
        None,
        "--thumbnails-dir",
        help="Write per-event JPEG thumbnails to this directory",
    ),
) -> None:
    cfg = Config()
    if telemetry_resolution is not None:
        cfg = cfg.model_copy(update={"telemetry_resolution_sec": telemetry_resolution})

    result = process_capture(input, cfg, thumbnails_dir=thumbnails_dir)
    json_out = result.model_dump_json(indent=2)

    if output:
        output.write_text(json_out)
        typer.echo(f"Result written to {output}", err=True)
    else:
        typer.echo(json_out)


def process_capture(capture_path: Path, cfg: Config, thumbnails_dir: Optional[Path] = None) -> EnrichmentResult:
    """
    Main pipeline:
      1. Ingest capture package
      2. Load tracking DataFrames
      3. Downsample video → temp dir
      4. Plan chunks
      5. For each chunk: extract video, build telemetry, annotate (Gemini Pass 1)
      6. Deduplicate events across overlapping chunks
      7. Segment into chapters (Gemini Pass 2)
      8. Return EnrichmentResult
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

        if thumbnails_dir is not None:
            thumbnails_dir.mkdir(parents=True, exist_ok=True)
            typer.echo(f"Extracting thumbnails → {thumbnails_dir}", err=True)
            for event in all_events:
                h, m, s = event.timestamp.split(":")
                offset_sec = int(h) * 3600 + int(m) * 60 + int(s)
                safe_ts = event.timestamp.replace(":", "-")
                dst = thumbnails_dir / f"{safe_ts}.jpg"
                extract_thumbnail(downsampled, dst, offset_sec)
                event.thumbnail_path = str(dst)

        typer.echo("Segmenting into chapters...", err=True)
        chapters = segment_into_chapters(client, all_events, duration)
        typer.echo(f"  {len(chapters)} chapter(s)", err=True)

    return EnrichmentResult(
        session_id=session_id,
        duration_seconds=metadata_duration,
        processed_at=datetime.now(timezone.utc).isoformat(),
        chapters=chapters,
    )


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
