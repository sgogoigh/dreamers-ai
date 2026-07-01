"""API + domain models.

Domain models for the *content* (TrailerScript / TrailerPrompts / beats) come
from the vendored pipeline package so request/response payloads and the
generator share one schema. Everything else (Project, jobs, requests) is defined
here.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# Content schema lives in the vendored pipeline package (app/pipeline/schema.py).
from .pipeline.schema import (  # noqa: F401
    TrailerBeat,
    TrailerScript,
    SegmentPrompt,
    TrailerPrompts,
    CastMember,
    Slot,
    TrailerBlueprint,
    default_blueprint,
)


# --- lifecycle -------------------------------------------------------------
class ProjectStatus(str, Enum):
    created = "created"            # just a concept
    drafted = "drafted"           # optional adapter/manual draft attached
    script_ready = "script_ready"  # trailer script generated
    prompts_ready = "prompts_ready"  # Veo prompts built
    generating = "generating"     # video generation in flight
    completed = "completed"       # trailer.mp4 ready
    failed = "failed"             # last operation failed


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class DraftSource(str, Enum):
    adapter = "adapter"
    manual = "manual"


# --- sub-objects -----------------------------------------------------------
class SegmentState(BaseModel):
    beat_no: int
    status: JobStatus = JobStatus.pending
    continues_previous: bool = False
    duration_seconds: int = 8
    is_title_card: bool = False
    video_url: Optional[str] = None
    detail: str = ""


class TrailerJob(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.pending
    progress: float = 0.0  # 0..1
    message: str = ""
    segments: List[SegmentState] = Field(default_factory=list)
    video_url: Optional[str] = None
    error: Optional[str] = None
    native_extend: bool = False
    use_anchor: bool = True
    created_at: str
    updated_at: str


class ScriptRevision(BaseModel):
    timestamp: str
    feedback: str
    source: str = "ai"  # "ai" | "manual"


# --- the central entity ----------------------------------------------------
class Project(BaseModel):
    id: str
    created_at: str
    updated_at: str
    status: ProjectStatus = ProjectStatus.created

    # inputs
    concept: str
    genre: str = ""
    tone: str = ""
    n_segments: int
    seg_seconds: int

    # structural contract for the trailer (the shot blueprint / graph). Defaults
    # to the canonical 8-shot structure; editable via the blueprint endpoints.
    blueprint: TrailerBlueprint = Field(default_factory=default_blueprint)

    # step 1 (optional)
    draft: Optional[str] = None
    draft_source: Optional[DraftSource] = None

    # step 2 / revisions
    script: Optional[TrailerScript] = None
    revisions: List[ScriptRevision] = Field(default_factory=list)

    # step 3
    prompts: Optional[TrailerPrompts] = None

    # step 4
    trailer: Optional[TrailerJob] = None


# --- request bodies --------------------------------------------------------
class CreateProjectRequest(BaseModel):
    concept: str = Field(min_length=3, description="One-line movie concept")
    genre: str = ""
    tone: str = ""
    n_segments: int = Field(default=0, ge=0, le=16,
                            description="0 = use server default")
    seg_seconds: int = Field(default=0, ge=0, le=8,
                             description="0 = use server default (max 8s/clip)")


class DraftRequest(BaseModel):
    mode: DraftSource = DraftSource.manual
    text: str = Field(default="", description="Required when mode=manual")
    max_new_tokens: int = Field(default=400, ge=64, le=1024)


class ReviseScriptRequest(BaseModel):
    feedback: str = Field(min_length=2,
                          description="Natural-language note on what to change")


class GenerateTrailerRequest(BaseModel):
    native_extend: bool = Field(
        default=False,
        description="Use Veo native video extension for continuation beats")
    use_anchor: bool = Field(
        default=True,
        description="Carry an anchor reference image across hard cuts")


# --- response envelopes ----------------------------------------------------
class ConfigResponse(BaseModel):
    mock: bool
    gemini_available: bool
    adapter_available: bool
    gemini_model: str
    veo_model: str
    adapter_repo: str
    base_model: str
    default_segments: int
    default_seg_seconds: int
    resolution: str
    aspect_ratio: str


class ProjectSummary(BaseModel):
    id: str
    concept: str
    genre: str
    tone: str
    status: ProjectStatus
    created_at: str
    updated_at: str
    has_script: bool
    has_prompts: bool
    has_video: bool


class MessageResponse(BaseModel):
    detail: str
