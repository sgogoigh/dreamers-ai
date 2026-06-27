# Dreamers-AI Trailer Backend

A **fully self-contained** FastAPI service implementing the trailer pipeline as a
**review-friendly, step-by-step REST + SSE API**. It vendors the entire pipeline
under [`app/pipeline/`](app/pipeline/) (a copy of the logic in
`../scripts/trailer/`), so the backend depends on **nothing outside this folder**
and can be lifted out and deployed on its own.

```
concept
  │  (step 1, optional)  fine-tuned LoRA adapter  → rough draft scene   [GPU only]
  │  (step 2)            gemini-3.5-flash         → structured trailer script  ← editable / AI-revisable
  │  (step 3)            gemini-3.5-flash         → chained Veo 3.1 prompts     ← editable
  └  (step 4)            veo-3.1                  → 8×8s clips, chained + stitched → trailer.mp4
```

Why step-by-step: each stage is a separate endpoint so a frontend can show the
user the script, let them edit it or ask the AI to revise it, *then* build
prompts, review those, and only then spend money on video.

---

## Quick start

```powershell
# deps — reuse the parent project's venv, or make your own (python -m venv .venv)
..\venv\Scripts\python.exe -m pip install -r requirements.txt

# run in MOCK mode — no API keys, no cost, fully functional (stub Gemini/Veo,
# real ffmpeg placeholder clips). Best for frontend dev + CI.
.\run.ps1 -Mock

# run live (reads GEMINI_API_KEY from backend/.env, falling back to ../.env;
# step 4 calls paid Veo).
.\run.ps1
```

Then open **http://localhost:8000/docs** for interactive OpenAPI docs.

### Modes
| Mode | `DREAMERS_MOCK` | Gemini (steps 2/3) | Veo (step 4) | Adapter (step 1) |
|------|----------------|--------------------|--------------|------------------|
| Mock | `1` | stub fixtures | ffmpeg color clips | stub seed |
| Live | `0` | real API | real API ($) | GPU only, else `503` |

---

## The flow, end to end

| # | Method & path | What it does |
|---|---|---|
| — | `GET /api/health` | liveness + mock flag |
| — | `GET /api/config` | model ids, defaults, capabilities (drives the UI) |
| — | `POST /api/projects` | create a project from a concept |
| — | `GET /api/projects` / `GET /api/projects/{id}` | list / fetch |
| — | `DELETE /api/projects/{id}` | delete project + artifacts |
| 1 | `POST /api/projects/{id}/draft` | optional draft — `{"mode":"manual","text":...}` or `{"mode":"adapter"}` (GPU) |
| 2 | `POST /api/projects/{id}/script` | generate the structured trailer script |
| 2 | `PUT /api/projects/{id}/script` | replace with a user-edited script (validated) |
| 2 | `POST /api/projects/{id}/script/revise` | AI revision — `{"feedback":"make it scarier"}` |
| 3 | `POST /api/projects/{id}/prompts` | build chained Veo prompts from the script |
| 3 | `PUT /api/projects/{id}/prompts` | replace with user-edited prompts |
| 4 | `POST /api/projects/{id}/trailer` | **start** async video generation (202) |
| 4 | `GET /api/projects/{id}/trailer` | poll job status / progress |
| 4 | `GET /api/projects/{id}/trailer/events` | **SSE** live progress |
| 4 | `GET /api/projects/{id}/trailer/video` | download the stitched `trailer.mp4` |
| 4 | `GET /api/projects/{id}/segments/{n}` | download an individual clip |

Editing a script clears downstream prompts; the lifecycle status
(`created → drafted → script_ready → prompts_ready → generating → completed/failed`)
tracks exactly where a project is.

---

## Frontend integration

- **CORS** is open by default (`DREAMERS_CORS_ORIGINS=*`); set it to your origins in prod.
- **Capability discovery:** call `GET /api/config` on load. `adapter_available:false`
  means hide/disable the "generate draft with the model" button (no GPU) — manual
  draft + script-from-concept still work. `mock:true` lets you badge the UI.
- **Typed contracts:** the full schema is at `/openapi.json` — generate a client with
  `openapi-typescript` / `orval` and you get end-to-end types for `TrailerScript`,
  `SegmentPrompt`, `Project`, etc.
- **Long job, two ways to watch it:**

```js
// Option A — Server-Sent Events (recommended)
const es = new EventSource(`/api/projects/${id}/trailer/events`);
es.onmessage = (e) => {
  const ev = JSON.parse(e.data);          // {type, beat_no, status, progress, message}
  if (ev.type === "close") es.close();
  updateProgressBar(ev.progress);
};

// Option B — polling (works everywhere)
const poll = setInterval(async () => {
  const job = await (await fetch(`/api/projects/${id}/trailer`)).json();
  if (["completed", "failed"].includes(job.status)) clearInterval(poll);
}, 1500);
```

- **Playing video:** point a `<video>` tag straight at
  `/api/projects/${id}/trailer/video` (served as `video/mp4`).

Typical UI wizard:
1. concept form → `POST /projects`
2. (optional) draft step → `POST /projects/{id}/draft`
3. show script (`POST /script`); offer **Edit** (`PUT /script`) and **Ask AI to revise**
   (`POST /script/revise`)
4. show prompts (`POST /prompts`); offer **Edit** (`PUT /prompts`)
5. **Generate trailer** (`POST /trailer`) → progress bar via SSE → `<video>`

---

## Persistence & layout

JSON-on-disk (no DB). One project per file; survives restarts; human-inspectable.

```
backend/
  app/
    config.py          # standalone settings + env loading + require_gemini_key
    models.py          # API models + content schema (from app/pipeline/schema)
    storage.py         # atomic JSON project store (lock-guarded)
    jobs.py            # async video job runner + SSE fan-out
    deps.py            # get_project dependency
    pipeline/          # VENDORED pipeline — no deps outside this folder
      constants.py     #   model ids + trailer defaults
      schema.py        #   TrailerScript / TrailerPrompts / beats
      adapter.py       #   step 1 (LoRA draft, GPU)
      script_writer.py #   step 2 (Gemini trailer script)
      prompt_builder.py#   step 3 (Gemini Veo prompts)
      video.py         #   step 4 (Veo clip + chain + ffmpeg stitch helpers)
    services/
      adapter_service.py  # step 1 wrapper (GPU-aware, degrades gracefully)
      gemini_service.py   # steps 2 & 3 + AI revision (+ mock fixtures)
      veo_service.py      # step 4 orchestration w/ progress (+ ffmpeg mock)
    routers/ meta.py projects.py pipeline.py
  tests/               # pytest suite (runs entirely in mock mode)
  data/projects/       # <id>.json + <id>/trailer_out/*.mp4
```

---

## Tests

```powershell
..\venv\Scripts\python.exe -m pytest
```

The suite runs in **mock mode** (no keys, no cost): project CRUD, all pipeline
steps + their guards, script editing/revision, schema validation, the full async
video job to completion, artifact serving, and both SSE paths (snapshot + live).
