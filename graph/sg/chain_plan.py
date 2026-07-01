"""Stage 3 — derive the video CHAINING PLAN from the graph (pure, no LLM).

Walks the beats in NEXT order and reads the graph's edges/attrs to decide, for
each beat, HOW it is produced and welded to its neighbour:

  * strategy = "title"     if the beat is a title card (composed, not generated)
              = "continue"  if it has a CONTINUES_FROM edge (seed from the prev
                            clip's last frame — the seam is welded)
              = "fresh"     otherwise (hard cut; carry a consistency anchor)
  * anchor              — recurring entities (APPEARS_IN ≥ 2) mean we reuse a
                          reference image across cuts to keep the cast consistent
  * fade_in / fade_out  — from the beat's transition attrs (applied in post)

This is the graph-driven analog of "chain the clips": continuity is not a flag on
a list, it is an edge in the graph that the planner resolves into a concrete
render step.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel

from .scene_graph import SceneGraph


class RenderStep(BaseModel):
    beat_no: int
    beat_id: str
    strategy: Literal["fresh", "continue", "title"]
    seconds: int
    prompt: str = ""
    negative_prompt: str = ""
    fade_in: bool = False
    fade_out: bool = False
    is_title_card: bool = False
    title_text: str = ""
    subtitle: str = ""                         # tagline shown under the title
    continue_from_beat: Optional[int] = None   # beat_no this one seeds from
    use_anchor: bool = False                   # carry a reference image across the cut
    anchor_entities: List[str] = []            # recurring entity names anchored here


class ChainPlan(BaseModel):
    steps: List[RenderStep]

    def summary(self) -> str:
        rows = []
        for s in self.steps:
            edge = (f"⤶ continues beat {s.continue_from_beat}" if s.strategy == "continue"
                    else ("◆ title card" if s.strategy == "title" else "✂ hard cut"))
            anchor = f"  anchor[{', '.join(s.anchor_entities)}]" if s.use_anchor else ""
            fades = ("  fade-in" if s.fade_in else "") + ("  fade-out" if s.fade_out else "")
            rows.append(f"  beat {s.beat_no}: {s.seconds}s  {edge}{anchor}{fades}")
        return "\n".join(rows)

    def total_seconds(self) -> int:
        return sum(s.seconds for s in self.steps)


def build_plan(g: SceneGraph) -> ChainPlan:
    recurring = {e.id for e in g.recurring_entities()}
    concept = g.concept()
    tagline = (concept.logline if concept else "").strip()
    steps: List[RenderStep] = []

    for beat in g.ordered_beats():
        pred = g.predecessor(beat.id)
        ents = g.entities_in(beat.id)
        anchored = [e.name for e in ents if e.id in recurring]

        if beat.is_title_card:
            strategy = "title"
        elif pred is not None:
            strategy = "continue"
        else:
            strategy = "fresh"

        steps.append(RenderStep(
            beat_no=beat.beat_no,
            beat_id=beat.id,
            strategy=strategy,
            seconds=beat.target_seconds,
            prompt=beat.prompt,
            negative_prompt=beat.negative_prompt,
            fade_in=(beat.transition_in == "fade_in"),
            fade_out=(beat.transition_out == "fade_out"),
            is_title_card=beat.is_title_card,
            title_text=(beat.on_screen_text if beat.is_title_card else ""),
            subtitle=(tagline if beat.is_title_card else ""),
            continue_from_beat=(pred.beat_no if pred else None),
            # a fresh cut with recurring cast should carry the anchor reference
            use_anchor=(strategy == "fresh" and bool(anchored)),
            anchor_entities=anchored,
        ))
    return ChainPlan(steps=steps)
