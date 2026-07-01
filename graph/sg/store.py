"""Persist the Scene Graph as JSON, reload it, and export a Mermaid diagram.

The graph IS the stored artifact — nodes + typed edges round-trip losslessly, so
a trailer's whole structure (world, cast, shots, continuity) can be saved,
inspected, hand-edited, and re-rendered later.
"""
from __future__ import annotations

from pathlib import Path

from .model import Beat, EdgeType
from .scene_graph import SceneGraph


def save_graph(g: SceneGraph, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(g.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_graph(path: Path) -> SceneGraph:
    return SceneGraph.model_validate_json(Path(path).read_text(encoding="utf-8"))


_EDGE_STYLE = {
    EdgeType.NEXT: "-->|next|",
    EdgeType.CONTINUES_FROM: "-.->|continues|",
    EdgeType.TRANSITIONS_TO: "-.->|transitions|",
    EdgeType.APPEARS_IN: "-->|appears in|",
    EdgeType.LOCATED_IN: "-->|located in|",
    EdgeType.STYLED_BY: "-->|styled by|",
    EdgeType.TITLE_OF: "-->|title of|",
}


def _safe(node_id: str) -> str:
    return node_id.replace(":", "_").replace("-", "_")


def to_mermaid(g: SceneGraph) -> str:
    """A Mermaid `graph LR` rendering — paste into any Mermaid viewer."""
    lines = ["graph LR"]
    for n in g.nodes:
        nid = _safe(n.id)
        if n.kind == "beat":
            b: Beat = n  # type: ignore[assignment]
            tag = "◆" if b.is_title_card else ("⤶" if b.continues_previous else "✂")
            lines.append(f'  {nid}["{tag} beat {b.beat_no}\\n{b.section} · {b.pace} · {b.target_seconds}s"]')
        elif n.kind == "entity":
            lines.append(f'  {nid}(["{n.name}"])')          # rounded = entity
        elif n.kind == "location":
            lines.append(f'  {nid}[/"{n.name}"/]')           # parallelogram = location
        elif n.kind == "style":
            lines.append(f'  {nid}{{{{"style"}}}}')          # hexagon = style
        elif n.kind == "concept":
            lines.append(f'  {nid}[("{n.title}")]')          # stadium = concept root
    for e in g.edges:
        arrow = _EDGE_STYLE.get(e.type, "-->")
        lines.append(f"  {_safe(e.source)} {arrow} {_safe(e.target)}")
    return "\n".join(lines)


def graph_summary(g: SceneGraph) -> str:
    c = g.concept()
    parts = []
    if c:
        parts.append(f'"{c.title}" — {c.logline}  [{c.genre} / {c.tone}]')
    counts = g.appearance_counts()
    ents = ", ".join(f"{e.name}×{counts.get(e.id, 0)}" for e in g.entities())
    parts.append(f"cast: {ents or '(none)'}")
    parts.append(f"locations: {', '.join(l.name for l in g.locations()) or '(none)'}")
    parts.append(f"nodes: {len(g.nodes)}  edges: {len(g.edges)}  beats: {len(g.of_kind('beat'))}")
    return "\n".join(parts)
