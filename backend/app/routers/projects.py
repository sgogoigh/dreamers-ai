"""Project CRUD."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, status

from .. import storage
from ..config import settings
from ..deps import get_project
from ..models import (
    CreateProjectRequest,
    MessageResponse,
    Project,
    ProjectSummary,
)

router = APIRouter(prefix="/projects", tags=["projects"])


def _summary(p: Project) -> ProjectSummary:
    return ProjectSummary(
        id=p.id,
        concept=p.concept,
        genre=p.genre,
        tone=p.tone,
        status=p.status,
        created_at=p.created_at,
        updated_at=p.updated_at,
        has_script=p.script is not None,
        has_prompts=p.prompts is not None,
        has_video=bool(p.trailer and p.trailer.video_url),
    )


@router.post("", response_model=Project, status_code=status.HTTP_201_CREATED)
def create_project(body: CreateProjectRequest) -> Project:
    return storage.create(
        concept=body.concept,
        genre=body.genre,
        tone=body.tone,
        n_segments=body.n_segments or settings.N_SEGMENTS,
        seg_seconds=body.seg_seconds or settings.SEG_SECONDS,
    )


@router.get("", response_model=List[ProjectSummary])
def list_projects() -> List[ProjectSummary]:
    return [_summary(p) for p in storage.list_all()]


@router.get("/{project_id}", response_model=Project)
def get_one(project: Project = Depends(get_project)) -> Project:
    return project


@router.delete("/{project_id}", response_model=MessageResponse)
def delete_project(project: Project = Depends(get_project)) -> MessageResponse:
    storage.delete(project.id)
    return MessageResponse(detail=f"Project '{project.id}' deleted")
