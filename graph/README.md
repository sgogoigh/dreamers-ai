# graph — a graph-native trailer engine

Turns a **one-line idea** into a **coherent chained trailer** by modelling the
film as a **typed scene graph** and letting every stage traverse that graph. It's
the GitNexus philosophy applied to video: *model the thing as a typed graph of
entities and relationships, persist it, and read from it.*

```
idea → SceneGraph (Gemini) → Veo prompts (Gemini) → chain plan (graph algorithm)
     → stored graph JSON + Mermaid → rendered, stitched trailer.mp4
```

## Why a graph (not a list)

A trailer is not a flat list of shots — it's a web of relationships. The graph
makes those relationships **first-class edges** that the downstream stages read:

| Node | holds |
|---|---|
| **Concept** (root) | title, logline, genre, tone |
| **Style** | palette, film stock, lens, lighting, grade, mood |
| **Entity** | a character/object/vehicle with a *fixed* description |
| **Location** | a setting |
| **Beat** | one shot ≈ one clip: role, pace, duration, transitions, content, prompt |

| Edge | meaning | used for |
|---|---|---|
| `NEXT` | narrative order | sequencing the beats |
| `CONTINUES_FROM` | seamed continuation | last-frame chaining (weld the seam) |
| `APPEARS_IN` | entity → beat | re-inject the SAME description every cut; recurring cast → anchor reference |
| `LOCATED_IN` | beat → location | grounding the prompt |
| `STYLED_BY` | → style | one coherent look across all beats |
| `TITLE_OF` / `TRANSITIONS_TO` | title card / fades | composed title, fade in/out |

Continuity stops being a boolean on a row and becomes an **edge the planner
resolves** into a concrete render step. Character consistency stops being "please
re-describe" and becomes a **graph query** (`entities_in(beat)`), so beat 2's
prompt literally re-states beat 1's exact wardrobe.

## Structure = a fixed blueprint, content = the model

The 8-shot structure (fade-in hook → continued setup → fast escalation cuts with
a 6s slow-down → climax tease → 4s title card) is a **blueprint** stamped onto the
beats deterministically. Gemini only fills creative content **and the
connections** (which entities appear in which beat, where). Durations snap to
Veo's valid 4/6/8s; fades and the (legible) title are rendered in post.

## Quick start

```bash
pip install -r requirements.txt
cp ../.env ./.env            # provides GEMINI_API_KEY  (already done)

# idea → graph → prompts → plan → stored graph → mock-rendered trailer (free)
python run.py "a lighthouse keeper discovers the fog is alive" \
    --genre "psychological horror" --tone dread --out out/fog

python run.py "..." --no-render     # graph + prompts + plan only, no video
python run.py "..." --veo           # render with REAL Veo 3.1 (spends $)
```

Outputs in `--out`: `scene_graph.json` (the persisted graph), `scene_graph.mmd`
(a Mermaid diagram — paste into any Mermaid viewer), and `trailer.mp4`.

## Rendering

- **mock** (default): ffmpeg placeholder clips + composed title + fades, stitched.
  Exercises the whole plan → video path for free.
- **veo**: real Veo 3.1 with the proven chaining — `continue` steps seed from the
  previous clip's last frame, `fresh` cuts carry an anchor reference image for
  cross-cut cast consistency, the title card is composed (no Veo call). Uses the
  same `GEMINI_API_KEY`.

  **Veo 3.1 constraint (found live):** asset reference images (Ingredients) are
  only accepted at the native **8s** length — a 6s beat + reference image returns
  `400: Your use case is currently not supported`. So the renderer drops the anchor
  on any non-8s beat (the graph still injects the exact cast description into the
  prompt, so consistency mostly holds); the 6s pacing is preserved. Native video
  extension (`video=`) is likewise unsupported on the Developer API, which is why
  continuations use last-frame seeding, not `video=`.

To render an already-built graph with Veo without re-paying for the Gemini stages
(and resuming past any clips already on disk):

```bash
python run.py --from-graph out/fog/scene_graph.json --out out/fog_veo --veo
```

## Layout

```
graph/
  run.py                 # CLI: idea → trailer
  sg/
    model.py             # typed nodes + typed edges (the data model)
    scene_graph.py       # SceneGraph container + traversal/queries
    build.py             # stage 1: idea → GraphSpec (Gemini) → constructed graph
    serialize.py         # stage 2: graph-connected beats → Veo prompts (Gemini)
    chain_plan.py        # stage 3: graph → render plan (pure graph algorithm)
    render.py            # execute plan → clips → stitched mp4 (mock | veo)
    store.py             # JSON persist/load + Mermaid export + summary
    config.py gemini.py  # env + model ids; structured-output client
  tests/test_end_to_end.py
```

## Tests

```bash
python -m pytest tests -q            # offline graph tests (no network) always run
python -m pytest tests -q -k live    # real Gemini end-to-end (needs GEMINI_API_KEY)
```

Offline tests build a graph from a hand-authored spec and check the edges,
ordering, appearance counts, chain plan, JSON roundtrip, and a real mock render.
The live test runs the actual Gemini pipeline, then renders with the free mock
backend and asserts the produced trailer is coherent (~58s, title card last).
