"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import HTTPException, Path

from . import storage
from .models import Project


def get_project(project_id: str = Path(..., description="12-char project id")) -> Project:
    project = storage.load(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return project
