"""Graph-native trailer engine.

Pipeline:  idea → SceneGraph (Gemini) → prompts (Gemini) → chain plan (graph algo)
           → stored graph JSON + Mermaid → rendered, stitched trailer.mp4

The scene graph is the brain: nodes hold the world, typed edges connect the parts
(order, continuation, cast reuse, location, style), and every later stage is a
traversal of that graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .build import idea_to_graph, build_graph, build_spec, default_blueprint, GraphSpec
from .chain_plan import ChainPlan, build_plan
from .render import render
from .scene_graph import SceneGraph
from .serialize import serialize_prompts
from .store import graph_summary, load_graph, save_graph, to_mermaid


@dataclass
class TrailerResult:
    graph: SceneGraph
    plan: ChainPlan
    graph_json: Path
    mermaid: Path
    video: Optional[Path]


def generate_trailer(
    idea: str, *, genre: str = "", tone: str = "", out_dir: Path,
    render_mode: str = "mock", do_render: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> TrailerResult:
    """End-to-end: idea → graph → prompts → plan → store → (render)."""
    cb = progress_cb or (lambda _m: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cb("building scene graph from idea…")
    graph = idea_to_graph(idea, genre=genre, tone=tone)
    problems = graph.validate_graph()
    if problems:
        raise RuntimeError("graph failed validation: " + "; ".join(problems))

    cb("serializing beats → Veo prompts…")
    serialize_prompts(graph)

    cb("deriving chain plan from graph…")
    plan = build_plan(graph)

    graph_json = save_graph(graph, out_dir / "scene_graph.json")
    mermaid = out_dir / "scene_graph.mmd"
    mermaid.write_text(to_mermaid(graph), encoding="utf-8")

    video = None
    if do_render:
        cb(f"rendering ({render_mode})…")
        video = render(plan, out_dir, mode=render_mode, progress_cb=cb)

    return TrailerResult(graph=graph, plan=plan, graph_json=graph_json,
                         mermaid=mermaid, video=video)


__all__ = [
    "generate_trailer", "TrailerResult",
    "idea_to_graph", "build_graph", "build_spec", "default_blueprint", "GraphSpec",
    "serialize_prompts", "build_plan", "ChainPlan", "render",
    "SceneGraph", "save_graph", "load_graph", "to_mermaid", "graph_summary",
]
