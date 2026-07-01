"""Typed nodes and edges for the trailer Scene Graph.

The scene graph is the single source of truth for a trailer. It models the film
as a *typed graph* rather than a flat list:

  nodes  = Concept (root) · Style (visual bible) · Entity (cast/objects) ·
           Location · Beat (one shot ≈ one clip)
  edges  = NEXT (order) · CONTINUES_FROM (seamed chain) · TRANSITIONS_TO (fade) ·
           APPEARS_IN (entity→beat) · LOCATED_IN (beat→location) ·
           STYLED_BY (beat→style) · TITLE_OF (title beat→concept)

The edges are what make it a graph and not a list: continuity, entity reuse,
location and style are all first-class *relationships* the downstream stages
(chain planning, prompt serialization) traverse to produce a coherent video.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Dict, List, Literal, Union

from pydantic import BaseModel, Field

# Fade transitions are applied in post (ffmpeg), not by Veo.
Transition = Literal["none", "fade_in", "fade_out"]
Pace = Literal["slow", "medium", "fast"]
Section = Literal["hook", "setup", "escalation", "climax", "title_card", "stinger"]
EntityRole = Literal["character", "object", "vehicle", "prop", "crowd"]


class EdgeType(str, Enum):
    NEXT = "NEXT"                     # beat -> beat : narrative order
    CONTINUES_FROM = "CONTINUES_FROM"  # beat -> beat : seamed continuation (last-frame chain)
    TRANSITIONS_TO = "TRANSITIONS_TO"  # beat -> beat : deliberate fade/interpolation
    APPEARS_IN = "APPEARS_IN"        # entity -> beat : keeps a subject consistent across cuts
    LOCATED_IN = "LOCATED_IN"        # beat -> location
    STYLED_BY = "STYLED_BY"          # beat/concept -> style
    TITLE_OF = "TITLE_OF"            # title-card beat -> concept


# --- nodes -----------------------------------------------------------------
class Concept(BaseModel):
    kind: Literal["concept"] = "concept"
    id: str
    label: str = ""
    title: str
    logline: str
    genre: str = ""
    tone: str = ""


class Style(BaseModel):
    kind: Literal["style"] = "style"
    id: str
    label: str = ""
    palette: str = ""
    film_stock: str = ""
    lens: str = ""
    lighting: str = ""
    grade: str = ""
    mood: str = ""


class Entity(BaseModel):
    kind: Literal["entity"] = "entity"
    id: str
    label: str = ""
    name: str
    role: EntityRole = "character"
    description: str = Field(default="", description="Fixed look reused verbatim across beats")


class Location(BaseModel):
    kind: Literal["location"] = "location"
    id: str
    label: str = ""
    name: str
    description: str = ""


class Beat(BaseModel):
    kind: Literal["beat"] = "beat"
    id: str
    label: str = ""
    beat_no: int
    section: Section = "escalation"
    pace: Pace = "medium"
    target_seconds: int = 8
    continues_previous: bool = False
    transition_in: Transition = "none"
    transition_out: Transition = "none"
    is_title_card: bool = False
    # creative content
    visual: str = ""
    voiceover: str = ""
    dialogue: str = ""
    on_screen_text: str = ""
    mood: str = ""
    # filled by the serialize stage (graph -> Veo prose)
    prompt: str = ""
    negative_prompt: str = ""


Node = Annotated[
    Union[Concept, Style, Entity, Location, Beat],
    Field(discriminator="kind"),
]


class Edge(BaseModel):
    source: str
    target: str
    type: EdgeType
    attrs: Dict[str, str] = Field(default_factory=dict)

    def key(self) -> tuple:
        return (self.source, self.target, self.type)
