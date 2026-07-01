"""Step 4 — low-level Veo 3.1 clip generation + chaining + ffmpeg stitch helpers.

Vendored from scripts/trailer/step4_generate_video.py. The orchestration loop
(with progress callbacks) lives in app/services/veo_service.py; these are the
reusable building blocks it calls.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import List

from .constants import VEO_MODEL, RESOLUTION, ASPECT_RATIO, SEG_SECONDS, snap_duration

# All post-processed clips (fades, title card) are normalised to these specs so a
# stream-copy concat with the Veo/mock clips stays clean (the stitch re-encode
# path is the safety net if a real clip still differs).
_W, _H, _FPS = 1280, 720, 24


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
        duration_seconds=str(snap_duration(seg_seconds)),  # Veo accepts only 4/6/8
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


# --- transitions & title card ----------------------------------------------
def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: List[str], what: str) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg {what} failed:\n{res.stderr[-2000:]}")


def apply_fades(src: Path, out: Path, seconds: int, *,
                fade_in: bool = False, fade_out: bool = False, dur: float = 1.0) -> Path:
    """Re-encode `src` with a fade-in and/or fade-out (video only; audio copied).

    The `fade` filter is a core ffmpeg filter (no libfreetype needed). `seconds`
    is the clip length, used to place the fade-out at the tail.
    """
    if not (fade_in or fade_out):
        return src
    filters = []
    if fade_in:
        filters.append(f"fade=t=in:st=0:d={dur}")
    if fade_out:
        filters.append(f"fade=t=out:st={max(0.0, seconds - dur):.3f}:d={dur}")
    cmd = [_ffmpeg(), "-y", "-i", str(src), "-vf", ",".join(filters),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(out)]
    _run(cmd, "fade")
    return out


def _title_still(freeze_frame, title: str, out_png: Path) -> Path:
    """Compose the title-card still: (darkened) freeze frame + centered title.

    `freeze_frame` is a PIL Image (as returned by `extract_last_frame`) or None.
    Text is drawn with PIL (reliable legible glyphs) rather than ffmpeg drawtext,
    which requires a libfreetype-enabled build we can't assume.
    """
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont

    if freeze_frame is not None:
        base = freeze_frame.convert("RGB").resize((_W, _H))
        base = ImageEnhance.Brightness(base).enhance(0.35)  # dim so the title reads
    else:
        base = Image.new("RGB", (_W, _H), (8, 8, 12))

    draw = ImageDraw.Draw(base)
    text = (title or "").strip().upper()
    font = None
    for name in ("arialbd.ttf", "Arial Bold.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(name, size=int(_H * 0.11))
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text(((_W - tw) / 2 - box[0], (_H - th) / 2 - box[1]), text,
              fill=(240, 240, 235), font=font)
    base.save(out_png)
    return out_png


def make_title_card(freeze_frame, title: str, out_path: Path,
                    seconds: int = 4, *, fade_out: bool = True) -> Path:
    """Build a title-card clip (still + title + fade-out + silent audio).

    `freeze_frame` is a PIL Image (or None for a black card). Rendered
    deterministically — no Veo call — so the title is always legible. Silent
    stereo aac + h264 match the other clips for a clean concat.
    """
    still = _title_still(freeze_frame, title, out_path.with_suffix(".title.png"))
    vf = f"fps={_FPS},format=yuv420p"
    if fade_out:
        vf += f",fade=t=out:st={max(0.0, seconds - 1.0):.3f}:d=1.0"
    cmd = [
        _ffmpeg(), "-y",
        "-loop", "1", "-t", str(seconds), "-i", str(still),
        "-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=44100:cl=stereo",
        "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(out_path),
    ]
    _run(cmd, "title card")
    still.unlink(missing_ok=True)
    return out_path


# --- stitching --------------------------------------------------------------
def stitch(segment_paths: List[Path], out_path: Path, *, reencode: bool = False) -> Path:
    """Concatenate clips (keeping audio) using the ffmpeg bundled with imageio-ffmpeg.

    `reencode=True` skips the stream-copy attempt and re-encodes directly — use it
    when the inputs are heterogeneous (post-processed fades / composed title card)
    so the concat is guaranteed clean.
    """
    ffmpeg = _ffmpeg()
    listfile = out_path.with_suffix(".concat.txt")
    listfile.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in segment_paths), encoding="utf-8")

    if not reencode:
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
               "-c", "copy", str(out_path)]
        print("  stitching:", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            listfile.unlink(missing_ok=True)
            return out_path
        # fall through to re-encode on codec mismatch

    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
           "-c:v", "libx264", "-c:a", "aac", str(out_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg stitch failed:\n{res.stderr[-2000:]}")
    listfile.unlink(missing_ok=True)
    return out_path
