"""The trailer pipeline, exposed step by step so a frontend can drive (and let
the user review/edit between) every stage:

  step 1  POST   /projects/{id}/draft           optional adapter/manual draft
  step 2  POST   /projects/{id}/script          generate trailer script
          PUT    /projects/{id}/script          replace with user-edited script
          POST   /projects/{id}/script/revise   AI revision from feedback
  step 3  POST   /projects/{id}/prompts         build Veo prompts
          PUT    /projects/{id}/prompts         replace with user-edited prompts
  step 4  POST   /projects/{id}/trailer         start video generation (async)
          GET    /projects/{id}/trailer         poll job status
          GET    /projects/{id}/trailer/events  live progress (SSE)
          GET    /projects/{id}/trailer/video   final stitched mp4
          GET    /projects/{id}/segments/{n}    individual segment mp4
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse

from .. import storage
from ..deps import get_project
from ..jobs import job_manager
from ..models import (
    DraftRequest,
    DraftSource,
    GenerateTrailerRequest,
    Project,
    ProjectStatus,
    ReviseScriptRequest,
    ScriptRevision,
    TrailerJob,
    TrailerPrompts,
    TrailerScript,
)
from ..services import adapter_service, gemini_service

router = APIRouter(prefix="/projects", tags=["pipeline"])


# --- step 1: draft ---------------------------------------------------------
@router.post("/{project_id}/draft", response_model=Project)
def make_draft(body: DraftRequest, project: Project = Depends(get_project)) -> Project:
    if body.mode == DraftSource.manual:
        if not body.text.strip():
            raise HTTPException(422, "text is required when mode=manual")
        project.draft = body.text
        project.draft_source = DraftSource.manual
    else:  # adapter
        try:
            project.draft = adapter_service.generate_draft(
                concept=project.concept,
                genre=project.genre,
                tone=project.tone,
                max_new_tokens=body.max_new_tokens,
            )
        except adapter_service.AdapterUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
        project.draft_source = DraftSource.adapter
    if project.status == ProjectStatus.created:
        project.status = ProjectStatus.drafted
    return storage.save(project)


# --- step 2: script --------------------------------------------------------
@router.post("/{project_id}/script", response_model=TrailerScript)
def generate_script(project: Project = Depends(get_project)) -> TrailerScript:
    script = gemini_service.generate_script(
        concept=project.concept,
        draft=project.draft or "",
        genre=project.genre,
        tone=project.tone,
        n_segments=project.n_segments,
    )
    project.script = script
    project.prompts = None  # invalidate downstream artifacts
    project.status = ProjectStatus.script_ready
    storage.save(project)
    return script


@router.put("/{project_id}/script", response_model=TrailerScript)
def replace_script(body: TrailerScript, project: Project = Depends(get_project)) -> TrailerScript:
    """Accept a user-edited script wholesale (validated against the schema)."""
    for i, beat in enumerate(body.beats, start=1):
        beat.beat_no = i
    if body.beats:
        body.beats[0].continues_previous = False
    project.script = body
    project.prompts = None
    project.revisions.append(
        ScriptRevision(timestamp=storage.now_iso(), feedback="(manual edit)", source="manual")
    )
    project.status = ProjectStatus.script_ready
    storage.save(project)
    return body


@router.post("/{project_id}/script/revise", response_model=TrailerScript)
def revise_script(body: ReviseScriptRequest, project: Project = Depends(get_project)) -> TrailerScript:
    if project.script is None:
        raise HTTPException(409, "No script to revise. Generate a script first.")
    revised = gemini_service.revise_script(script=project.script, feedback=body.feedback)
    project.script = revised
    project.prompts = None
    project.revisions.append(
        ScriptRevision(timestamp=storage.now_iso(), feedback=body.feedback, source="ai")
    )
    project.status = ProjectStatus.script_ready
    storage.save(project)
    return revised


# --- step 3: prompts -------------------------------------------------------
@router.post("/{project_id}/prompts", response_model=TrailerPrompts)
def build_prompts(project: Project = Depends(get_project)) -> TrailerPrompts:
    if project.script is None:
        raise HTTPException(409, "No script. Generate or edit a script first.")
    prompts = gemini_service.build_prompts(
        script=project.script, seg_seconds=project.seg_seconds
    )
    project.prompts = prompts
    project.status = ProjectStatus.prompts_ready
    storage.save(project)
    return prompts


@router.put("/{project_id}/prompts", response_model=TrailerPrompts)
def replace_prompts(body: TrailerPrompts, project: Project = Depends(get_project)) -> TrailerPrompts:
    body.segments.sort(key=lambda s: s.beat_no)
    if body.segments:
        body.segments[0].continues_previous = False
    project.prompts = body
    project.status = ProjectStatus.prompts_ready
    storage.save(project)
    return body


# --- step 4: trailer video -------------------------------------------------
@router.post("/{project_id}/trailer", response_model=TrailerJob, status_code=status.HTTP_202_ACCEPTED)
async def start_trailer(
    body: GenerateTrailerRequest, project: Project = Depends(get_project)
) -> TrailerJob:
    if project.prompts is None or not project.prompts.segments:
        raise HTTPException(409, "No prompts. Build Veo prompts first.")
    if job_manager.is_running(project.id):
        raise HTTPException(409, "A trailer job is already running for this project.")
    return await job_manager.start(
        project, native_extend=body.native_extend, use_anchor=body.use_anchor
    )


@router.get("/{project_id}/trailer", response_model=TrailerJob)
def trailer_status(project: Project = Depends(get_project)) -> TrailerJob:
    if project.trailer is None:
        raise HTTPException(404, "No trailer job for this project yet.")
    return project.trailer


@router.get("/{project_id}/trailer/events")
async def trailer_events(project: Project = Depends(get_project)):
    """Server-Sent Events stream of live job progress.

    Emits the current snapshot first, then live events, closing on terminal
    state. Frontend: `new EventSource('.../trailer/events')`.
    """
    project_id = project.id
    queue = job_manager.subscribe(project_id)

    async def event_gen():
        # initial snapshot so a late subscriber still gets current state
        snapshot = storage.load(project_id)
        if snapshot and snapshot.trailer:
            yield _sse({"type": "snapshot", **json.loads(snapshot.trailer.model_dump_json())})
            if snapshot.trailer.status in ("completed", "failed"):
                yield _sse({"type": "close"})
                job_manager.unsubscribe(project_id, queue)
                return
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # comment frame keeps the connection open
                    continue
                yield _sse(event)
                if event.get("type") == "close":
                    break
        finally:
            job_manager.unsubscribe(project_id, queue)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# --- artifact serving ------------------------------------------------------
@router.get("/{project_id}/trailer/video")
def get_video(project: Project = Depends(get_project)) -> FileResponse:
    path = storage.project_artifacts_dir(project.id) / "trailer.mp4"
    if not path.exists():
        raise HTTPException(404, "Trailer video not generated yet.")
    return FileResponse(path, media_type="video/mp4", filename=f"{project.id}_trailer.mp4")


@router.get("/{project_id}/segments/{beat_no}")
def get_segment(beat_no: int, project: Project = Depends(get_project)) -> FileResponse:
    path = storage.project_artifacts_dir(project.id) / f"segment_{beat_no:02d}.mp4"
    if not path.exists():
        raise HTTPException(404, f"Segment {beat_no} not generated yet.")
    return FileResponse(path, media_type="video/mp4", filename=f"{project.id}_segment_{beat_no:02d}.mp4")
