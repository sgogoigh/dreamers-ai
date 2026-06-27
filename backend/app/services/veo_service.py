"""Step 4 — generate the trailer video, segment by segment, with progress hooks.

The real path reuses the existing low-level Veo helpers (`generate_segment`,
`extract_last_frame`, `stitch`, ...) from scripts/trailer/step4_generate_video,
re-implementing only the orchestration loop so we can emit a `progress_cb` after
every beat (the CLI version just prints). The chaining logic — last-frame
seeding, native extension, and the cross-cut anchor reference image — mirrors
the CLI exactly.

The MOCK path produces a real, playable placeholder clip per beat with the
bundled ffmpeg (solid color + silent audio, same codec/size/fps), then runs the
*real* `stitch()`. So the whole assemble-and-serve path is exercised for free.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, List, Optional

from ..config import settings
from ..models import TrailerPrompts

# Vendored low-level Veo building blocks (app/pipeline/video.py).
from ..pipeline.video import (
    generate_segment,
    extract_last_frame,
    _to_genai_image,
    _save,
    stitch,
)

# progress_cb(event: dict) -> None ; events documented inline below.
ProgressCb = Callable[[dict], None]

_MOCK_COLORS = [
    "0x1b2a4a", "0x3a1b4a", "0x4a1b1b", "0x1b4a2a",
    "0x4a431b", "0x1b3a4a", "0x2a2a2a", "0x4a2a1b",
]


def _noop(_: dict) -> None:
    pass


def _make_mock_clip(path: Path, color: str, seconds: int) -> None:
    """Create a playable seconds-long clip (h264 + silent aac, 1280x720@24)."""
    ffmpeg = _ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s=1280x720:r=24:d={seconds}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(seconds),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"mock clip ffmpeg failed:\n{res.stderr[-1500:]}")


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run_generation(
    *,
    prompts: TrailerPrompts,
    out_dir: Path,
    native_extend: bool = False,
    use_anchor: bool = True,
    seg_seconds: int = 8,
    progress_cb: Optional[ProgressCb] = None,
) -> Path:
    """Generate + chain + stitch the trailer. Returns the final mp4 path.

    Emits, via progress_cb:
      {"type":"segment","beat_no":n,"status":"running"|"completed","detail":...,
       "video_filename":"segment_0N.mp4"}
      {"type":"stitch","status":"running"}
      {"type":"final","video_filename":"trailer.mp4"}
    """
    cb = progress_cb or _noop
    out_dir.mkdir(parents=True, exist_ok=True)

    if settings.MOCK:
        return _run_mock(prompts, out_dir, seg_seconds, cb)
    return _run_real(prompts, out_dir, native_extend, use_anchor, seg_seconds, cb)


# ---------------------------------------------------------------------------
def _run_mock(prompts: TrailerPrompts, out_dir: Path, seg_seconds: int, cb: ProgressCb) -> Path:
    segment_paths: List[Path] = []
    for i, seg in enumerate(prompts.segments):
        fname = f"segment_{seg.beat_no:02d}.mp4"
        path = out_dir / fname
        cb({"type": "segment", "beat_no": seg.beat_no, "status": "running",
            "detail": "mock render", "video_filename": fname})
        if not (path.exists() and path.stat().st_size > 0):
            _make_mock_clip(path, _MOCK_COLORS[i % len(_MOCK_COLORS)], seg_seconds)
        segment_paths.append(path)
        cb({"type": "segment", "beat_no": seg.beat_no, "status": "completed",
            "detail": "mock clip ready", "video_filename": fname})

    cb({"type": "stitch", "status": "running"})
    final = stitch(segment_paths, out_dir / "trailer.mp4")
    cb({"type": "final", "video_filename": final.name})
    return final


# ---------------------------------------------------------------------------
def _run_real(
    prompts: TrailerPrompts,
    out_dir: Path,
    native_extend: bool,
    use_anchor: bool,
    seg_seconds: int,
    cb: ProgressCb,
) -> Path:
    from google import genai

    from ..config import require_gemini_key

    client = genai.Client(api_key=require_gemini_key())

    segment_paths: List[Path] = []
    prev_video = None
    prev_last_frame = None
    anchor_img = None

    for seg in prompts.segments:
        fname = f"segment_{seg.beat_no:02d}.mp4"
        path = out_dir / fname

        # Resume: never re-pay for a clip that already exists on disk.
        if path.exists() and path.stat().st_size > 0:
            cb({"type": "segment", "beat_no": seg.beat_no, "status": "completed",
                "detail": "reused existing clip", "video_filename": fname})
            segment_paths.append(path)
            prev_video = None
            prev_last_frame = extract_last_frame(
                path, out_dir / f"_lastframe_{seg.beat_no:02d}.png")
            if anchor_img is None and use_anchor:
                anchor_img = _to_genai_image(prev_last_frame)
            continue

        kind = "continuation" if seg.continues_previous else "fresh cut"
        cb({"type": "segment", "beat_no": seg.beat_no, "status": "running",
            "detail": f"generating ({kind})", "video_filename": fname})

        image = video = None
        if seg.continues_previous:
            if native_extend and prev_video is not None:
                video = prev_video
            elif prev_last_frame is not None:
                image = _to_genai_image(prev_last_frame)

        refs = (
            [anchor_img]
            if (use_anchor and anchor_img is not None and image is None and video is None)
            else None
        )

        gv = generate_segment(
            client, prompt=seg.prompt, negative_prompt=seg.negative_prompt,
            image=image, video=video, reference_images=refs,
            seg_seconds=seg_seconds, label=f"beat {seg.beat_no}",
        )
        _save(client, gv, path)
        segment_paths.append(path)

        prev_video = gv.video
        prev_last_frame = extract_last_frame(
            path, out_dir / f"_lastframe_{seg.beat_no:02d}.png")
        if anchor_img is None and use_anchor:
            anchor_img = _to_genai_image(prev_last_frame)

        cb({"type": "segment", "beat_no": seg.beat_no, "status": "completed",
            "detail": "clip ready", "video_filename": fname})

    cb({"type": "stitch", "status": "running"})
    final = stitch(segment_paths, out_dir / "trailer.mp4")
    cb({"type": "final", "video_filename": final.name})
    return final
