"""End-to-end tests for the graph-native trailer engine.

`test_offline_*` build a graph from a hand-authored spec (no network) so the whole
graph → plan → store → render path is exercised for free in CI.

`test_live_end_to_end` runs the REAL Gemini pipeline (idea → graph → prompts) when
GEMINI_API_KEY is available, then renders with the free mock backend. It is
skipped automatically if no key is present.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from sg import (
    build_graph, build_plan, default_blueprint, generate_trailer,
    load_graph, save_graph, serialize_prompts, to_mermaid,
)
from sg.build import GraphSpec, SpecBeat, SpecEntity, SpecLocation, SpecStyle
from sg.config import gemini_key
from sg.model import EdgeType
from sg.render import render


def _fake_spec() -> GraphSpec:
    """A deterministic 8-beat spec (what Gemini would return), offline."""
    beats = []
    for n in range(1, 9):
        beats.append(SpecBeat(
            beat_no=n, visual=f"visual for beat {n}",
            dialogue=("We're not alone." if n in (3, 5) else ""),
            voiceover=("In a world of fog…" if n == 1 else ""),
            mood="dread",
            entity_ids=["e1"] + (["e2"] if n >= 2 else []),
            location_id="l1",
        ))
    return GraphSpec(
        title="THE FOG", logline="A keeper learns the fog is alive.",
        genre="horror", tone="dread",
        style=SpecStyle(palette="cold teal", film_stock="35mm", lens="anamorphic",
                        lighting="low-key", grade="desaturated", mood="oppressive"),
        cast=[SpecEntity(id="e1", name="Cass", role="character",
                         description="50s, weathered, grey beard, oilskin coat"),
              SpecEntity(id="e2", name="Mara", role="character",
                         description="20s, red parka, short dark hair")],
        locations=[SpecLocation(id="l1", name="Lighthouse", description="storm-battered tower")],
        beats=beats,
    )


def _dur(mp4: Path) -> float:
    import imageio_ffmpeg, re
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(mp4)],
                       capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r)
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


# --- offline graph structure -----------------------------------------------
def test_offline_graph_shape_and_edges():
    g = build_graph(_fake_spec())
    assert not g.validate_graph()                       # structurally healthy
    assert g.concept().title == "THE FOG"
    assert len(g.of_kind("beat")) == 8

    beats = g.ordered_beats()
    assert [b.beat_no for b in beats] == list(range(1, 9))   # NEXT ordering
    assert beats[0].continues_previous is False
    assert beats[1].continues_previous is True               # blueprint slot 2

    # CONTINUES_FROM edge exists for beat 2 -> beat 1
    assert g.predecessor("beat:2") is not None
    assert g.predecessor("beat:2").beat_no == 1
    assert g.predecessor("beat:3") is None                   # hard cut

    # APPEARS_IN wiring: Cass in all 8, Mara in 7 → both recurring
    counts = g.appearance_counts()
    assert counts["e1"] == 8 and counts["e2"] == 7
    names = {e.name for e in g.recurring_entities()}
    assert names == {"Cass", "Mara"}

    # structural stamping from the blueprint
    assert beats[3].target_seconds == 6                      # slot 4 slow-down
    assert beats[7].is_title_card is True
    assert beats[7].target_seconds == 4
    assert beats[0].transition_in == "fade_in"
    assert beats[7].transition_out == "fade_out"


def test_offline_chain_plan_reflects_graph():
    g = build_graph(_fake_spec())
    for b in g.of_kind("beat"):
        b.prompt = f"prompt {b.beat_no}"
    plan = build_plan(g)
    assert [s.beat_no for s in plan.steps] == list(range(1, 9))
    assert plan.steps[0].strategy == "fresh" and plan.steps[0].fade_in
    assert plan.steps[1].strategy == "continue" and plan.steps[1].continue_from_beat == 1
    assert plan.steps[2].strategy == "fresh"
    assert plan.steps[2].use_anchor is True                  # recurring cast on a hard cut
    assert plan.steps[7].strategy == "title" and plan.steps[7].fade_out
    assert plan.total_seconds() == 8 + 8 + 8 + 6 + 8 + 8 + 8 + 4


def test_offline_store_roundtrip_and_mermaid(tmp_path):
    g = build_graph(_fake_spec())
    p = save_graph(g, tmp_path / "g.json")
    g2 = load_graph(p)
    assert g2.model_dump() == g.model_dump()                 # lossless roundtrip
    mmd = to_mermaid(g)
    assert mmd.startswith("graph LR")
    assert "beat 1" in mmd and "continues" in mmd and "appears in" in mmd


def test_offline_render_mock_produces_video(tmp_path):
    g = build_graph(_fake_spec())
    for b in g.of_kind("beat"):
        b.prompt = f"prompt {b.beat_no}"
    plan = build_plan(g)
    final = render(plan, tmp_path, mode="mock")
    assert final.exists()
    # 8 clips: 8+8+8+6+8+8+8+4 = 58s (± container rounding)
    assert abs(_dur(final) - 58) <= 2
    assert (tmp_path / "segment_08.mp4").exists()             # composed title card


# --- live pipeline (real Gemini; mock render) -------------------------------
@pytest.mark.skipif(not gemini_key(), reason="GEMINI_API_KEY not set")
def test_live_end_to_end(tmp_path):
    res = generate_trailer(
        "a deep-sea diver finds a drowned city that remembers her",
        genre="sci-fi mystery", tone="awe and dread",
        out_dir=tmp_path, render_mode="mock", do_render=True,
    )
    g = res.graph
    assert not g.validate_graph()
    assert len(g.of_kind("beat")) == 8
    assert g.concept().title                                  # a real title
    assert g.entities()                                       # a real cast
    assert any(g.appearance_counts().get(e.id, 0) >= 2 for e in g.entities())  # reuse
    # every beat got a real, non-trivial Veo prompt
    assert all(len(b.prompt) > 40 for b in g.ordered_beats())
    # plan coherence
    assert res.plan.steps[0].fade_in
    assert res.plan.steps[-1].strategy == "title"
    # artifacts
    assert res.graph_json.exists() and res.mermaid.exists()
    assert res.video and res.video.exists()
    assert abs(_dur(res.video) - res.plan.total_seconds()) <= 3
