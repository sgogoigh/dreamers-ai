"""Execute a ChainPlan into clips and stitch a coherent trailer.

Two backends:
  * mock (default) — ffmpeg placeholder clips (color + silent audio), a composed
    title card, and fades. Free, fast, exercises the ENTIRE assemble path so the
    plan produces a real playable mp4 without spending on Veo.
  * veo            — real Veo 3.1 with the proven chaining: `continue` steps seed
    from the previous clip's last frame; `fresh` steps carry an anchor reference
    image for cross-cut consistency; the title card is composed (no Veo call).

Both honor the plan's per-step duration, fades, and title card.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, List, Optional

from .chain_plan import ChainPlan, RenderStep
from .config import ASPECT_RATIO, RESOLUTION, VEO_MODEL, snap_duration, require_gemini_key

_W, _H, _FPS = 1280, 720, 24
_MOCK_COLORS = ["0x1b2a4a", "0x3a1b4a", "0x4a1b1b", "0x1b4a2a",
                "0x4a431b", "0x1b3a4a", "0x2a2a2a", "0x4a2a1b"]
ProgressCb = Callable[[str], None]


# --- ffmpeg / imaging helpers ----------------------------------------------
def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: List[str], what: str) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg {what} failed:\n{res.stderr[-1800:]}")


def _mock_clip(path: Path, color: str, seconds: int) -> None:
    _run([_ffmpeg(), "-y",
          "-f", "lavfi", "-i", f"color=c={color}:s={_W}x{_H}:r={_FPS}:d={seconds}",
          "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(seconds),
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
         "mock clip")


def _extract_last_frame(mp4: Path, out_png: Path):
    import imageio.v3 as iio
    from PIL import Image
    frames = iio.imread(mp4, plugin="pyav")
    iio.imwrite(out_png, frames[-1])
    return Image.open(out_png)


def _apply_fades(src: Path, out: Path, seconds: int, *, fade_in: bool, fade_out: bool) -> Path:
    if not (fade_in or fade_out):
        return src
    filt = []
    if fade_in:
        filt.append("fade=t=in:st=0:d=1.0")
    if fade_out:
        filt.append(f"fade=t=out:st={max(0.0, seconds - 1.0):.3f}:d=1.0")
    _run([_ffmpeg(), "-y", "-i", str(src), "-vf", ",".join(filt),
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(out)], "fade")
    return out


def _load_font(size: int):
    from PIL import ImageFont
    for name in ("arialbd.ttf", "Arial Bold.ttf", "arial.ttf", "segoeui.ttf",
                 "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_tracked(draw, cx: int, y: int, text: str, font, fill, tracking: int, shadow=None):
    """Draw letter-spaced text centered on cx, with an optional drop shadow."""
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * max(0, len(text) - 1)
    x = cx - total / 2
    for ch, w in zip(text, widths):
        if shadow:
            off, col = shadow
            draw.text((x + off, y + off), ch, font=font, fill=col)
        draw.text((x, y), ch, font=font, fill=fill)
        x += w + tracking


def _vignette(base):
    """Darken the edges radially for a cinematic title card."""
    from PIL import Image, ImageDraw, ImageFilter
    mask = Image.new("L", (_W, _H), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([-_W * 0.25, -_H * 0.25, _W * 1.25, _H * 1.25], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(120))
    dark = Image.new("RGB", (_W, _H), (0, 0, 0))
    return Image.composite(base, dark, mask)


def _title_card(freeze, title: str, out: Path, seconds: int, *, fade_out: bool,
                subtitle: str = "") -> Path:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    if freeze is not None:
        base = freeze.convert("RGB").resize((_W, _H))
        base = base.filter(ImageFilter.GaussianBlur(3))          # soften the footage
        base = ImageEnhance.Brightness(base).enhance(0.42)
        base = ImageEnhance.Contrast(base).enhance(1.05)
        base = _vignette(base)
    else:
        base = _vignette(Image.new("RGB", (_W, _H), (12, 12, 16)))

    draw = ImageDraw.Draw(base)
    title = (title or "").strip().upper()
    tfont = _load_font(int(_H * 0.12))
    tbox = draw.textbbox((0, 0), title, font=tfont)
    th = tbox[3] - tbox[1]
    ty = int(_H * 0.42) - th
    _draw_tracked(draw, _W // 2, ty, title, tfont, (243, 240, 232),
                  tracking=int(_H * 0.02), shadow=(3, (0, 0, 0)))

    # thin rule under the title
    rule_y = int(_H * 0.42) + int(_H * 0.06)
    draw.line([(_W * 0.36, rule_y), (_W * 0.64, rule_y)], fill=(210, 205, 195), width=2)

    # small dim tagline (wrapped), if short enough to read on a card
    sub = (subtitle or "").strip()
    if sub:
        sfont = _load_font(int(_H * 0.035))
        words, lines, cur = sub.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if draw.textlength(trial, font=sfont) > _W * 0.6 and cur:
                lines.append(cur); cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        y = rule_y + int(_H * 0.04)
        for ln in lines[:2]:                                     # at most two lines
            _draw_tracked(draw, _W // 2, y, ln, sfont, (198, 194, 186),
                          tracking=int(_H * 0.006), shadow=(2, (0, 0, 0)))
            y += int(_H * 0.055)

    still = out.with_suffix(".title.png")
    base.save(still)
    vf = f"fps={_FPS},format=yuv420p"
    if fade_out:
        vf += f",fade=t=out:st={max(0.0, seconds - 1.0):.3f}:d=1.0"
    _run([_ffmpeg(), "-y", "-loop", "1", "-t", str(seconds), "-i", str(still),
          "-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=44100:cl=stereo",
          "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
          "-shortest", str(out)], "title card")
    still.unlink(missing_ok=True)
    return out


def _stitch(paths: List[Path], out: Path, *, reencode: bool) -> Path:
    ff = _ffmpeg()
    listfile = out.with_suffix(".concat.txt")
    listfile.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in paths), encoding="utf-8")
    if not reencode:
        r = subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
                            "-c", "copy", str(out)], capture_output=True, text=True)
        if r.returncode == 0:
            listfile.unlink(missing_ok=True); return out
    _run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
          "-c:v", "libx264", "-c:a", "aac", str(out)], "stitch")
    listfile.unlink(missing_ok=True)
    return out


# --- Veo helpers ------------------------------------------------------------
def _to_image(pil):
    import io
    from google.genai import types
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    return types.Image(image_bytes=buf.getvalue(), mime_type="image/png")


def _veo_clip(client, step: RenderStep, image=None, refs=None) -> Path:
    from google.genai import types
    import time
    cfg = dict(resolution=RESOLUTION, aspect_ratio=ASPECT_RATIO, number_of_videos=1,
               duration_seconds=str(snap_duration(step.seconds)))
    conditioned = image is not None or bool(refs)
    if step.negative_prompt and not conditioned:
        cfg["negative_prompt"] = step.negative_prompt
    if refs:
        cfg["reference_images"] = [
            types.VideoGenerationReferenceImage(image=r, reference_type="asset") for r in refs[:3]
        ]
    kwargs = dict(model=VEO_MODEL, prompt=step.prompt, config=types.GenerateVideosConfig(**cfg))
    if image is not None:
        kwargs["image"] = image
    op = client.models.generate_videos(**kwargs)
    while not op.done:
        time.sleep(10)
        op = client.operations.get(op)
    gv = op.response.generated_videos[0]
    client.files.download(file=gv.video)
    return gv.video


# --- the executor -----------------------------------------------------------
def render(plan: ChainPlan, out_dir: Path, *, mode: str = "mock",
           progress_cb: Optional[ProgressCb] = None) -> Path:
    """Execute the plan → per-beat clips → stitched trailer.mp4. Returns the mp4."""
    cb = progress_cb or (lambda _m: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    veo_client = None
    if mode == "veo":
        from google import genai
        veo_client = genai.Client(api_key=require_gemini_key())

    clip_paths: List[Path] = []
    prev_last_frame = None
    anchor_img = None

    for i, step in enumerate(plan.steps):
        path = out_dir / f"segment_{step.beat_no:02d}.mp4"
        if path.exists() and path.stat().st_size > 0:               # resume
            cb(f"beat {step.beat_no}: reuse existing")
            clip_paths.append(path)
            if not step.is_title_card:
                prev_last_frame = _extract_last_frame(path, out_dir / f"_lf_{step.beat_no:02d}.png")
                if anchor_img is None:
                    anchor_img = prev_last_frame
            continue

        if step.strategy == "title":
            cb(f"beat {step.beat_no}: compose title card")
            _title_card(prev_last_frame, step.title_text, path, step.seconds,
                        fade_out=step.fade_out, subtitle=step.subtitle)
            clip_paths.append(path)
            continue

        cb(f"beat {step.beat_no}: {step.strategy} ({step.seconds}s, {mode})")
        if mode == "veo":
            image = _to_image(prev_last_frame) if (step.strategy == "continue" and prev_last_frame is not None) else None
            # Veo 3.1 asset reference images (Ingredients) are only supported at the
            # native 8s length; on shorter beats we drop the anchor (the prompt still
            # carries the exact cast description via the graph) so pacing is kept.
            want_anchor = step.use_anchor and anchor_img is not None and image is None
            if want_anchor and snap_duration(step.seconds) != 8:
                cb(f"beat {step.beat_no}: anchor skipped (asset refs need 8s; this is {step.seconds}s)")
                want_anchor = False
            refs = [_to_image(anchor_img)] if want_anchor else None
            video = _veo_clip(veo_client, step, image=image, refs=refs)
            video.save(str(path))
        else:  # mock
            _mock_clip(path, _MOCK_COLORS[i % len(_MOCK_COLORS)], step.seconds)

        if step.fade_in or step.fade_out:
            tmp = path.with_suffix(".fade.mp4")
            _apply_fades(path, tmp, step.seconds, fade_in=step.fade_in, fade_out=step.fade_out)
            tmp.replace(path)

        clip_paths.append(path)
        prev_last_frame = _extract_last_frame(path, out_dir / f"_lf_{step.beat_no:02d}.png")
        if anchor_img is None:
            anchor_img = prev_last_frame

    cb("stitching")
    reencode = any(s.fade_in or s.fade_out or s.is_title_card for s in plan.steps)
    final = _stitch(clip_paths, out_dir / "trailer.mp4", reencode=reencode)
    cb(f"done → {final}")
    return final
