"""Background trailer-generation jobs + live progress (SSE).

Video generation is long-running and blocking (sync Veo calls + ffmpeg), so a
job runs in a worker thread via `asyncio.to_thread`. Each progress event does
two things:
  1. Persists incremental state into the project JSON (storage is lock-guarded),
     so plain polling of `GET .../trailer` always reflects reality.
  2. Fans out to any connected SSE subscribers for that project.

One active job per project; starting a second while one runs is rejected.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from . import storage
from .config import settings
from .models import (
    JobStatus,
    Project,
    ProjectStatus,
    SegmentState,
    TrailerJob,
)
from .services import veo_service


def _segment_url(project_id: str, beat_no: int) -> str:
    return f"/api/projects/{project_id}/segments/{beat_no}"


def _video_url(project_id: str) -> str:
    return f"/api/projects/{project_id}/trailer/video"


class JobManager:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._running: set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # --- SSE subscription -------------------------------------------------
    def subscribe(self, project_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(project_id, []).append(q)
        return q

    def unsubscribe(self, project_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(project_id)
        if subs and q in subs:
            subs.remove(q)
            if not subs:
                self._subscribers.pop(project_id, None)

    def _publish(self, project_id: str, event: dict) -> None:
        for q in self._subscribers.get(project_id, []):
            q.put_nowait(event)

    def is_running(self, project_id: str) -> bool:
        return project_id in self._running

    # --- lifecycle --------------------------------------------------------
    async def start(self, project: Project, *, native_extend: bool, use_anchor: bool) -> TrailerJob:
        if project.id in self._running:
            raise RuntimeError("A trailer job is already running for this project.")
        self._loop = asyncio.get_running_loop()

        job = TrailerJob(
            job_id=storage.new_id(),
            status=JobStatus.pending,
            progress=0.0,
            message="queued",
            segments=[
                SegmentState(
                    beat_no=s.beat_no,
                    continues_previous=s.continues_previous,
                    status=JobStatus.pending,
                )
                for s in project.prompts.segments
            ],
            native_extend=native_extend,
            use_anchor=use_anchor,
            created_at=storage.now_iso(),
            updated_at=storage.now_iso(),
        )
        project.trailer = job
        project.status = ProjectStatus.generating
        storage.save(project)

        self._running.add(project.id)
        asyncio.create_task(self._run(project.id, native_extend, use_anchor))
        return job

    async def _run(self, project_id: str, native_extend: bool, use_anchor: bool) -> None:
        try:
            project = storage.load(project_id)
            seg_seconds = project.seg_seconds
            prompts = project.prompts
            out_dir = storage.project_artifacts_dir(project_id)

            # Mark running.
            self._update(project_id, status=JobStatus.running, message="generating", progress=0.01)

            def progress_cb(event: dict) -> None:
                # Runs in the worker thread → persist synchronously (lock-guarded),
                # then schedule SSE fan-out on the event loop.
                enriched = self._apply_event(project_id, event)
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._publish, project_id, enriched)

            await asyncio.to_thread(
                veo_service.run_generation,
                prompts=prompts,
                out_dir=out_dir,
                native_extend=native_extend,
                use_anchor=use_anchor,
                seg_seconds=seg_seconds,
                progress_cb=progress_cb,
            )

            self._update(
                project_id,
                status=JobStatus.completed,
                message="completed",
                progress=1.0,
                video_url=_video_url(project_id),
                project_status=ProjectStatus.completed,
            )
            self._publish(project_id, {"type": "status", "status": "completed",
                                       "progress": 1.0, "video_url": _video_url(project_id)})
        except Exception as exc:  # noqa: BLE001
            self._update(
                project_id,
                status=JobStatus.failed,
                message="failed",
                error=f"{type(exc).__name__}: {exc}",
                project_status=ProjectStatus.failed,
            )
            self._publish(project_id, {"type": "status", "status": "failed",
                                       "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self._running.discard(project_id)
            self._publish(project_id, {"type": "close"})

    # --- state mutation (thread-safe via storage lock) --------------------
    def _apply_event(self, project_id: str, event: dict) -> dict:
        """Fold one veo_service event into the persisted job; return an enriched
        event (with computed progress) for SSE."""
        project = storage.load(project_id)
        job = project.trailer
        etype = event.get("type")

        if etype == "segment":
            for seg in job.segments:
                if seg.beat_no == event["beat_no"]:
                    seg.status = (
                        JobStatus.completed
                        if event["status"] == "completed"
                        else JobStatus.running
                    )
                    seg.detail = event.get("detail", "")
                    if event["status"] == "completed":
                        seg.video_url = _segment_url(project_id, seg.beat_no)
                    break
            job.message = f"beat {event['beat_no']}: {event.get('detail','')}"
        elif etype == "stitch":
            job.message = "stitching segments"
        elif etype == "final":
            job.message = "stitched"

        total_steps = len(job.segments) + 1  # +1 for the stitch step
        done = sum(1 for s in job.segments if s.status == JobStatus.completed)
        if etype == "final":
            done = total_steps
        job.progress = round(min(done / total_steps, 1.0), 3)
        job.updated_at = storage.now_iso()
        storage.save(project)

        return {
            "type": etype,
            "beat_no": event.get("beat_no"),
            "status": event.get("status"),
            "detail": event.get("detail", ""),
            "progress": job.progress,
            "message": job.message,
        }

    def _update(
        self,
        project_id: str,
        *,
        status: Optional[JobStatus] = None,
        message: Optional[str] = None,
        progress: Optional[float] = None,
        error: Optional[str] = None,
        video_url: Optional[str] = None,
        project_status: Optional[ProjectStatus] = None,
    ) -> None:
        project = storage.load(project_id)
        job = project.trailer
        if status is not None:
            job.status = status
        if message is not None:
            job.message = message
        if progress is not None:
            job.progress = progress
        if error is not None:
            job.error = error
        if video_url is not None:
            job.video_url = video_url
        job.updated_at = storage.now_iso()
        if project_status is not None:
            project.status = project_status
        storage.save(project)


# Process-wide singleton.
job_manager = JobManager()
