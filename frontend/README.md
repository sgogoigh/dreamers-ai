# Dreamers · AI Trailer Studio — Frontend

A single-page **Next.js** (App Router, TypeScript) UI for the trailer backend.
Pick genre / tone / scene count from dropdowns, describe your idea, and watch it
become a structured trailer script and a finished, downloadable trailer video —
on one screen, no scrolling.

## Features
- **Big animated hero title** with a moving gradient + drifting aurora background.
- **Dark, glassy** panels (backdrop blur, gradient borders, shine).
- Metadata via **dropdowns** (genre, tone, scenes) + an idea textarea.
- One **Generate** button runs the whole chain: project → script → Veo prompts →
  trailer render, with **live progress** (SSE, polling fallback).
- **Script** and **Trailer** tabs; **Download** buttons for the `.mp4` and the
  script `.json`.
- **Backend status pill** (bottom-right): green = live, red = offline (polls
  `/api/health` every 4s).
- Tab favicon and the heading icon are the **same** SVG (`app/icon.svg` +
  `lib/icon.ts`).

## Run

1. Start the backend (from `../backend`):
   ```powershell
   ..\..\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000   # add DREAMERS_MOCK=1 for free mode
   ```
2. Start the frontend:
   ```bash
   npm install
   npm run dev            # http://localhost:3000
   ```

Point the UI at a non-default backend by copying `.env.local.example` to
`.env.local` and setting `NEXT_PUBLIC_API_BASE`.

> CORS: the backend ships with permissive CORS, so the dev server on `:3000`
> talks to the API on `:8000` directly (fetch + EventSource). Lock down
> `DREAMERS_CORS_ORIGINS` on the backend for production.

## Layout
```
frontend/
  app/
    layout.tsx     # metadata + globals
    page.tsx       # the whole single-page studio (client component)
    globals.css    # dark/animated theme
    icon.svg       # favicon (identical markup to the heading icon)
  lib/
    api.ts         # typed backend client (fetch + SSE URLs)
    icon.ts        # shared SVG markup for the heading
```
