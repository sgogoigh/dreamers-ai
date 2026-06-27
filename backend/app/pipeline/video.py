"""Step 4 — low-level Veo 3.1 clip generation + chaining + ffmpeg stitch helpers.

Vendored from scripts/trailer/step4_generate_video.py. The orchestration loop
(with progress callbacks) lives in app/services/veo_service.py; these are the
reusable building blocks it calls.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

from .constants import VEO_MODEL, RESOLUTION, ASPECT_RATIO, SEG_SECONDS


# --- low-level helpers -----------------------------------------------------
def _poll(client, operation, label: str, every: int = 10):
    while not operation.done:
        print(f"  [{label}] generating...")
        time.sleep(every)
        operation = client.operations.get(operation)
    return operation


def _save(client, generated_video, path: Path) -> Path:
    client.files.download(file=generated_video.video)
    generated_video.video.save(str(path))
    return path


def extract_last_frame(mp4_path: Path, out_png: Path):
    """Grab the final frame of a clip as a PIL image (for last-frame chaining)."""
    import imageio.v3 as iio
    frames = iio.imread(mp4_path, plugin="pyav")  # (n, h, w, c)
    last = frames[-1]
    iio.imwrite(out_png, last)
    from PIL import Image
    return Image.open(out_png)


def _to_genai_image(pil_image):
    import io
    from google.genai import types
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return types.Image(image_bytes=buf.getvalue(), mime_type="image/png")


# --- segment generation ----------------------------------------------------
def generate_segment(
    client,
    prompt: str,
    negative_prompt: str = "",
    image=None,                 # genai Image: first frame (last frame of prev clip)
    video=None,                 # previous generated video (native extension)
    reference_images=None,      # list[genai Image] anchor for cross-cut consistency
    seg_seconds: int = SEG_SECONDS,
    label: str = "seg",
):
    from google.genai import types

    cfg = dict(
        resolution=RESOLUTION,
        aspect_ratio=ASPECT_RATIO,
        number_of_videos=1,
        duration_seconds=str(seg_seconds),
    )
    # Veo 3.1 rejects negative_prompt when the request is image/video/reference
    # conditioned ("Negative prompt is not supported in your use case"), so only
    # send it for pure text-to-video.
    conditioned = (image is not None) or (video is not None) or bool(reference_images)
    if negative_prompt and not conditioned:
        cfg["negative_prompt"] = negative_prompt
    if reference_images:
        cfg["reference_images"] = [
            types.VideoGenerationReferenceImage(image=img, reference_type="asset")
            for img in reference_images[:3]
        ]

    kwargs = dict(model=VEO_MODEL, prompt=prompt, config=types.GenerateVideosConfig(**cfg))
    if video is not None:
        kwargs["video"] = video            # native extension
    elif image is not None:
        kwargs["image"] = image            # last-frame seeding / image-to-video

    op = client.models.generate_videos(**kwargs)
    op = _poll(client, op, label)
    return op.response.generated_videos[0]


# --- stitching --------------------------------------------------------------
def stitch(segment_paths: List[Path], out_path: Path) -> Path:
    """Concatenate clips (keeping audio) using the ffmpeg bundled with imageio-ffmpeg."""
    import subprocess
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    listfile = out_path.with_suffix(".concat.txt")
    listfile.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in segment_paths), encoding="utf-8")
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
           "-c", "copy", str(out_path)]
    print("  stitching:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # fall back to re-encode if stream-copy concat fails (codec mismatch)
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
               "-c:v", "libx264", "-c:a", "aac", str(out_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg stitch failed:\n{res.stderr[-2000:]}")
    listfile.unlink(missing_ok=True)
    return out_path
