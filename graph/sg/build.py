"""Stage 1 — carve a user idea into a Scene Graph.

One Gemini call produces a flat, model-friendly `GraphSpec` (title, style, cast,
locations, and the beats — each declaring WHICH entities appear and WHERE). We
then *construct the graph*: nodes for every element and typed edges wiring them
together (NEXT, CONTINUES_FROM, APPEARS_IN, LOCATED_IN, STYLED_BY, TITLE_OF).

The trailer's STRUCTURE (role, pace, duration, continuity, fades, title card) is
owned by a fixed 8-shot blueprint and stamped onto the beats deterministically —
Gemini only supplies creative content and the entity/location connections.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .config import snap_duration
from .gemini import structured
from .model import (
    Beat, Concept, EdgeType, Entity, Location, Style,
)
from .scene_graph import SceneGraph


# --- the fixed structural blueprint (the "shot skeleton") ------------------
def default_blueprint() -> List[dict]:
    """Canonical 8-shot trailer: fade-in hook → continued setup → fast
    escalation cuts (one 6s slow-down) → climax tease → 4s title card."""
    return [
        dict(beat_no=1, section="hook", pace="slow", target_seconds=8,
             continues_previous=False, transition_in="fade_in", transition_out="none",
             is_title_card=False,
             intent="Cinematic establishing setting; build atmosphere; the main "
                    "character appears only in shadow/silhouette — no full reveal."),
        dict(beat_no=2, section="setup", pace="medium", target_seconds=8,
             continues_previous=True, transition_in="none", transition_out="none",
             is_title_card=False,
             intent="Continue directly from shot 1; introduce the character(s) and a "
                    "first short line; keep building tension."),
        dict(beat_no=3, section="escalation", pace="fast", target_seconds=8,
             continues_previous=False, transition_in="none", transition_out="none",
             is_title_card=False,
             intent="Hard cut. Fast, energetic beat with a short line; new location."),
        dict(beat_no=4, section="escalation", pace="slow", target_seconds=6,
             continues_previous=False, transition_in="none", transition_out="none",
             is_title_card=False,
             intent="A brief slower 'normal' beat to let the trailer breathe."),
        dict(beat_no=5, section="escalation", pace="fast", target_seconds=8,
             continues_previous=False, transition_in="none", transition_out="none",
             is_title_card=False,
             intent="Hard cut back to fast pacing; short punchy line; rising stakes."),
        dict(beat_no=6, section="escalation", pace="fast", target_seconds=8,
             continues_previous=False, transition_in="none", transition_out="none",
             is_title_card=False,
             intent="Fast cut; peak of the escalation; short line."),
        dict(beat_no=7, section="climax", pace="fast", target_seconds=8,
             continues_previous=False, transition_in="none", transition_out="none",
             is_title_card=False,
             intent="Fast climax tease — hint at the biggest moment without resolving."),
        dict(beat_no=8, section="title_card", pace="slow", target_seconds=4,
             continues_previous=False, transition_in="none", transition_out="fade_out",
             is_title_card=True,
             intent="The movie title, held on screen with an aesthetic fade to black."),
    ]


# --- the Gemini-facing spec (flat, no structural fields) -------------------
class SpecStyle(BaseModel):
    palette: str = Field(description="Color palette / grade")
    film_stock: str = Field(default="", description="Film stock or digital look")
    lens: str = Field(default="", description="Lens/camera character")
    lighting: str = Field(default="", description="Lighting philosophy")
    grade: str = Field(default="", description="Overall color grade")
    mood: str = Field(default="", description="Global mood in a few words")


class SpecEntity(BaseModel):
    id: str = Field(description="Short stable id, e.g. 'e1'")
    name: str
    role: str = Field(default="character", description="character|object|vehicle|prop|crowd")
    description: str = Field(description="Fixed physical look: age, build, hair, wardrobe")


class SpecLocation(BaseModel):
    id: str = Field(description="Short stable id, e.g. 'l1'")
    name: str
    description: str = ""


class SpecBeat(BaseModel):
    beat_no: int
    visual: str = Field(description="What we SEE (subject, action, staging; no camera jargon)")
    voiceover: str = Field(default="", description="Narrator line, if any (short)")
    dialogue: str = Field(default="", description="Spoken character line, if any (short)")
    on_screen_text: str = Field(default="", description="Caption/title text, if any")
    mood: str = Field(default="", description="Emotional tone of the beat")
    entity_ids: List[str] = Field(default_factory=list,
                                  description="ids of cast/objects that appear in this beat")
    location_id: str = Field(default="", description="id of the location for this beat")


class GraphSpec(BaseModel):
    title: str
    logline: str
    genre: str = ""
    tone: str = ""
    style: SpecStyle
    cast: List[SpecEntity] = Field(default_factory=list)
    locations: List[SpecLocation] = Field(default_factory=list)
    beats: List[SpecBeat]


_SYSTEM = (
    "You are a film-trailer story architect. From a one-line idea you design a "
    "small consistent world (title, visual style, a 2-4 person cast with FIXED "
    "descriptions, locations) and fill a fixed shot blueprint with beats, "
    "declaring which cast/objects appear in each beat and where. Output ONLY JSON "
    "matching the schema."
)


def _blueprint_lines(bp: List[dict]) -> str:
    out = []
    for s in bp:
        chain = "CONTINUES previous shot" if s["continues_previous"] else "fresh cut"
        title = " [MOVIE TITLE CARD]" if s["is_title_card"] else ""
        out.append(f"  beat {s['beat_no']} — {s['section']}, pace={s['pace']}, "
                   f"~{s['target_seconds']}s, {chain}{title}: {s['intent']}")
    return "\n".join(out)


_PROMPT = """Design a movie trailer from this idea, filling the SHOT BLUEPRINT exactly.

IDEA: {idea}
GENRE: {genre}
TONE: {tone}

SHOT BLUEPRINT — produce EXACTLY {n} beats, one per slot below, in order. Fill
each beat's creative content so it fulfils that slot's role. For every beat, list
the `entity_ids` of the cast/objects that appear in it and its `location_id`, so
the trailer's world is connected and consistent.
{blueprint}

RULES:
- Define a SMALL, CONSISTENT cast (2-4) with FIXED physical descriptions; reuse
  the SAME ids across beats so recurring characters stay identical across cuts.
- Give 1-3 locations with stable ids.
- Keep voiceover/dialogue SHORT (a beat is a few seconds). Very short beats may
  have no spoken line.
- The title-card beat's `on_screen_text` is the film TITLE only.
- Provide a coherent visual `style` that applies to the whole trailer.
"""


def build_spec(idea: str, *, genre: str = "", tone: str = "",
               blueprint: List[dict] | None = None) -> GraphSpec:
    bp = blueprint or default_blueprint()
    prompt = _PROMPT.format(idea=idea, genre=genre or "(infer)", tone=tone or "(infer)",
                            n=len(bp), blueprint=_blueprint_lines(bp))
    return structured(prompt, GraphSpec, system=_SYSTEM)


def build_graph(spec: GraphSpec, *, blueprint: List[dict] | None = None) -> SceneGraph:
    """Construct the typed SceneGraph from a spec + the structural blueprint."""
    bp = blueprint or default_blueprint()
    bp_by_no = {s["beat_no"]: s for s in bp}
    g = SceneGraph()

    # concept (root) + style
    g.add_node(Concept(id="concept", label=spec.title, title=spec.title,
                       logline=spec.logline, genre=spec.genre, tone=spec.tone))
    g.add_node(Style(id="style", label="visual style", palette=spec.style.palette,
                     film_stock=spec.style.film_stock, lens=spec.style.lens,
                     lighting=spec.style.lighting, grade=spec.style.grade,
                     mood=spec.style.mood))
    g.add_edge("concept", "style", EdgeType.STYLED_BY)

    # entities + locations
    valid_entities = set()
    for e in spec.cast:
        role = e.role if e.role in ("character", "object", "vehicle", "prop", "crowd") else "character"
        g.add_node(Entity(id=e.id, label=e.name, name=e.name, role=role, description=e.description))
        valid_entities.add(e.id)
    valid_locations = set()
    for loc in spec.locations:
        g.add_node(Location(id=loc.id, label=loc.name, name=loc.name, description=loc.description))
        valid_locations.add(loc.id)

    # beats (structure stamped from the blueprint) + wiring
    spec_beats = {b.beat_no: b for b in spec.beats}
    prev_id = None
    for slot in bp:
        n = slot["beat_no"]
        sb = spec_beats.get(n)
        beat_id = f"beat:{n}"
        beat = Beat(
            id=beat_id, label=f"beat {n} ({slot['section']})", beat_no=n,
            section=slot["section"], pace=slot["pace"],
            target_seconds=snap_duration(slot["target_seconds"]),
            continues_previous=slot["continues_previous"] and n > 1,
            transition_in=slot["transition_in"], transition_out=slot["transition_out"],
            is_title_card=slot["is_title_card"],
            visual=(sb.visual if sb else ""),
            voiceover=(sb.voiceover if sb else ""),
            dialogue=(sb.dialogue if sb else ""),
            on_screen_text=(spec.title if slot["is_title_card"] else (sb.on_screen_text if sb else "")),
            mood=(sb.mood if sb else spec.tone),
        )
        g.add_node(beat)

        if prev_id is not None:
            g.add_edge(prev_id, beat_id, EdgeType.NEXT)
        if beat.continues_previous and prev_id is not None:
            g.add_edge(beat_id, prev_id, EdgeType.CONTINUES_FROM)
        if slot["is_title_card"]:
            g.add_edge(beat_id, "concept", EdgeType.TITLE_OF)
        g.add_edge(beat_id, "style", EdgeType.STYLED_BY)

        if sb:
            for eid in sb.entity_ids:
                if eid in valid_entities:
                    g.add_edge(eid, beat_id, EdgeType.APPEARS_IN)
            if sb.location_id in valid_locations:
                g.add_edge(beat_id, sb.location_id, EdgeType.LOCATED_IN)

        prev_id = beat_id

    return g


def idea_to_graph(idea: str, *, genre: str = "", tone: str = "",
                  blueprint: List[dict] | None = None) -> SceneGraph:
    """Convenience: one call, idea → constructed SceneGraph."""
    bp = blueprint or default_blueprint()
    spec = build_spec(idea, genre=genre, tone=tone, blueprint=bp)
    return build_graph(spec, blueprint=bp)
