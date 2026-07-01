"""CLI: turn an idea into a graph-driven trailer.

    python run.py "a lighthouse keeper discovers the fog is alive" \
        --genre "psychological horror" --tone dread --out out/fog

    # real Veo render (spends on Veo; needs GEMINI_API_KEY):
    python run.py "..." --veo

    # graph + prompts + plan only, no video:
    python run.py "..." --no-render
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sg import (build_plan, generate_trailer, graph_summary, load_graph,
                render, save_graph, to_mermaid)


def _render_stored(graph_path: Path, out_dir: Path, mode: str):
    """Render an already-built (and serialized) graph — no new Gemini calls."""
    g = load_graph(graph_path)
    problems = g.validate_graph()
    if problems:
        raise SystemExit("graph failed validation: " + "; ".join(problems))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_graph(g, out_dir / "scene_graph.json")
    (out_dir / "scene_graph.mmd").write_text(to_mermaid(g), encoding="utf-8")
    plan = build_plan(g)
    video = render(plan, out_dir, mode=mode, progress_cb=lambda m: print(f"  · {m}"))
    return g, plan, video


def main() -> None:
    p = argparse.ArgumentParser(description="Graph-native trailer generator")
    p.add_argument("idea", nargs="?", default="", help="one-line movie idea")
    p.add_argument("--genre", default="")
    p.add_argument("--tone", default="")
    p.add_argument("--out", default="out/trailer", help="output directory")
    p.add_argument("--veo", action="store_true", help="render with real Veo 3.1 (costs $)")
    p.add_argument("--no-render", action="store_true", help="skip video; graph + plan only")
    p.add_argument("--from-graph", default="", metavar="PATH",
                   help="render an existing scene_graph.json (skips Gemini build)")
    a = p.parse_args()
    mode = "veo" if a.veo else "mock"

    if a.from_graph:
        g, plan, video = _render_stored(Path(a.from_graph), Path(a.out), mode)
        graph_json = Path(a.out) / "scene_graph.json"
        mermaid = Path(a.out) / "scene_graph.mmd"
    else:
        if not a.idea:
            raise SystemExit("provide an IDEA, or use --from-graph PATH")
        res = generate_trailer(
            a.idea, genre=a.genre, tone=a.tone, out_dir=Path(a.out),
            render_mode=mode, do_render=not a.no_render,
            progress_cb=lambda m: print(f"  · {m}"),
        )
        g, plan, video = res.graph, res.plan, res.video
        graph_json, mermaid = res.graph_json, res.mermaid

    print("\n" + graph_summary(g))
    print("\nCHAIN PLAN")
    print(plan.summary())
    print(f"\ntotal: {plan.total_seconds()}s")
    print(f"\ngraph  → {graph_json}")
    print(f"mermaid→ {mermaid}")
    if video:
        print(f"video  → {video}")


if __name__ == "__main__":
    main()
