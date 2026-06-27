"""JSON-file-on-disk persistence for Projects.

One project == one file at `data/projects/<id>.json`. Video artifacts live in
`data/projects/<id>/trailer_out/`. A module-level RLock serializes the
read-modify-write cycles so the background video job and incoming requests can't
clobber each other (single-process uvicorn).
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .config import settings
from .models import Project

_LOCK = threading.RLock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _project_path(project_id: str) -> Path:
    return settings.projects_dir / f"{project_id}.json"


def project_artifacts_dir(project_id: str) -> Path:
    d = settings.projects_dir / project_id / "trailer_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _validate_id(project_id: str) -> bool:
    # ids are 12 hex chars; reject anything else to prevent path traversal.
    return len(project_id) == 12 and all(c in "0123456789abcdef" for c in project_id)


def save(project: Project) -> Project:
    with _LOCK:
        project.updated_at = now_iso()
        path = _project_path(project.id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic on same filesystem
    return project


def load(project_id: str) -> Optional[Project]:
    if not _validate_id(project_id):
        return None
    with _LOCK:
        path = _project_path(project_id)
        if not path.exists():
            return None
        return Project.model_validate_json(path.read_text(encoding="utf-8"))


def delete(project_id: str) -> bool:
    if not _validate_id(project_id):
        return False
    with _LOCK:
        path = _project_path(project_id)
        existed = path.exists()
        path.unlink(missing_ok=True)
        # best-effort artifact cleanup
        art = settings.projects_dir / project_id
        if art.exists():
            import shutil

            shutil.rmtree(art, ignore_errors=True)
        return existed


def list_all() -> List[Project]:
    with _LOCK:
        out: List[Project] = []
        for path in settings.projects_dir.glob("*.json"):
            try:
                out.append(Project.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue  # skip corrupt/partial files
    out.sort(key=lambda p: p.created_at, reverse=True)
    return out


def create(
    *,
    concept: str,
    genre: str,
    tone: str,
    n_segments: int,
    seg_seconds: int,
) -> Project:
    ts = now_iso()
    project = Project(
        id=new_id(),
        created_at=ts,
        updated_at=ts,
        concept=concept,
        genre=genre,
        tone=tone,
        n_segments=n_segments,
        seg_seconds=seg_seconds,
    )
    return save(project)
