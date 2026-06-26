"""Step 4 — generate the trailer as chained Veo 3.1 clips, then stitch to one file.

Chaining strategy (per beat's `continues_previous` flag):
  * continues_previous = True  -> seamless continuation. We seed the next clip
    with the LAST FRAME of the previous clip (image-to-video), OR use Veo's
    native video extension (`video=prev`) when --native-extend is set.
  * continues_previous = False -> a fresh cut. We still pass a reusable ANCHOR
    reference image (the trailer's first frame) so the protagonist/look stays
    consistent across cuts (Veo reference_images, type "asset").

Veo clips are 720p while chaining (extension input constraint). Final stitch
concatenates all segments (video + native audio) with ffmpeg.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types

from .config import (VEO_MODEL, RESOLUTION, ASPECT_RATIO, SEG_SECONDS,
                     TrailerPrompts, require_gemini_key)


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


# --- orchestration ----------------------------------------------------------
def generate_trailer(
    prompts: TrailerPrompts,
    out_dir: str = "trailer_out",
    native_extend: bool = False,
    use_anchor: bool = True,
    seg_seconds: int = SEG_SECONDS,
) -> Path:
    client = genai.Client(api_key=require_gemini_key())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    segment_paths: List[Path] = []
    prev_video = None
    prev_last_frame = None
    anchor_img = None

    for seg in prompts.segments:
        label = f"beat {seg.beat_no}"
        path = out / f"segment_{seg.beat_no:02d}.mp4"

        # Resume: reuse an already-generated clip instead of paying to regenerate it.
        if path.exists() and path.stat().st_size > 0:
            print(f"{label}: reusing existing {path.name}")
            segment_paths.append(path)
            prev_video = None  # video object not recoverable from disk
            prev_last_frame = extract_last_frame(path, out / f"_lastframe_{seg.beat_no:02d}.png")
            if anchor_img is None and use_anchor:
                anchor_img = _to_genai_image(prev_last_frame)
            continue

        image = video = None
        if seg.continues_previous:
            if native_extend and prev_video is not None:
                video = prev_video
            elif prev_last_frame is not None:
                image = _to_genai_image(prev_last_frame)
            print(f"{label}: continuation ({'native-extend' if video else 'last-frame seed'})")
        else:
            print(f"{label}: fresh cut")

        refs = [anchor_img] if (use_anchor and anchor_img is not None and image is None and video is None) else None

        gv = generate_segment(
            client, prompt=seg.prompt, negative_prompt=seg.negative_prompt,
            image=image, video=video, reference_images=refs,
            seg_seconds=seg_seconds, label=label,
        )
        path = _save(client, gv, out / f"segment_{seg.beat_no:02d}.mp4")
        segment_paths.append(path)
        print(f"  saved {path}")

        # update chaining state
        prev_video = gv.video
        prev_last_frame = extract_last_frame(path, out / f"_lastframe_{seg.beat_no:02d}.png")
        if anchor_img is None and use_anchor:
            anchor_img = _to_genai_image(prev_last_frame)  # first clip's end = look anchor

    final = stitch(segment_paths, out / "trailer.mp4")
    print(f"\n[OK] Trailer assembled: {final}  ({len(segment_paths)} x {seg_seconds}s)")
    return final


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-file", required=True, help="trailer_prompts.json from step 3")
    p.add_argument("--out-dir", default="trailer_out")
    p.add_argument("--native-extend", action="store_true",
                   help="use Veo native video extension for continuation beats")
    p.add_argument("--no-anchor", action="store_true", help="disable cross-cut anchor reference image")
    a = p.parse_args()

    prompts = TrailerPrompts.model_validate_json(open(a.prompts_file, encoding="utf-8").read())
    generate_trailer(prompts, a.out_dir, native_extend=a.native_extend, use_anchor=not a.no_anchor)
