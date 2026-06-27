// Thin typed client for the Dreamers trailer backend.
// Base URL is configurable; defaults to the local FastAPI server. The backend
// sends permissive CORS headers, so cross-origin fetch + SSE work directly.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";

export interface Beat {
  beat_no: number;
  section: string;
  setting: string;
  visual: string;
  voiceover?: string;
  dialogue?: string;
  on_screen_text?: string;
  mood: string;
  continues_previous: boolean;
}
export interface TrailerScript {
  title: string;
  logline: string;
  genre: string;
  tone: string;
  beats: Beat[];
}
export interface SegmentState {
  beat_no: number;
  status: string;
  continues_previous: boolean;
  video_url?: string | null;
  detail?: string;
}
export interface TrailerJob {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  message: string;
  segments: SegmentState[];
  video_url?: string | null;
  error?: string | null;
}
export interface Project {
  id: string;
  status: string;
  concept: string;
  n_segments: number;
}
export interface BackendConfig {
  mock: boolean;
  gemini_available: boolean;
  adapter_available: boolean;
  gemini_model: string;
  veo_model: string;
  default_segments: number;
  default_seg_seconds: number;
}

async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const d = await r.json();
      if (d?.detail) detail = typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail);
    } catch {}
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

export const api = {
  health: () => getJSON<{ status: string; mock: boolean }>("/api/health"),
  config: () => getJSON<BackendConfig>("/api/config"),
  createProject: (b: {
    concept: string;
    genre: string;
    tone: string;
    n_segments: number;
    seg_seconds: number;
  }) => postJSON<Project>("/api/projects", b),
  generateScript: (id: string) => postJSON<TrailerScript>(`/api/projects/${id}/script`, {}),
  buildPrompts: (id: string) => postJSON<unknown>(`/api/projects/${id}/prompts`, {}),
  startTrailer: (id: string, b: { native_extend?: boolean; use_anchor?: boolean }) =>
    postJSON<TrailerJob>(`/api/projects/${id}/trailer`, b),
  trailerStatus: (id: string) => getJSON<TrailerJob>(`/api/projects/${id}/trailer`),
  videoUrl: (id: string) => `${API_BASE}/api/projects/${id}/trailer/video`,
  eventsUrl: (id: string) => `${API_BASE}/api/projects/${id}/trailer/events`,
};
