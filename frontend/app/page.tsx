"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, API_BASE, TrailerScript } from "@/lib/api";
import { ICON_SVG } from "@/lib/icon";

const GENRES = [
  "Action", "Adventure", "Comedy", "Crime", "Drama", "Fantasy", "Horror",
  "Psychological Horror", "Mystery", "Romance", "Sci-Fi", "Thriller", "Western",
];
const TONES = [
  "Dark & gritty", "Epic & hopeful", "Tense & paranoid", "Dread & isolation",
  "Slick & stylish", "Whimsical & playful", "Melancholic & wondrous",
  "Bleak & unsettling", "Heartwarming", "High-octane",
];
const SCENES: [string, string][] = [
  ["3", "3 — quick (~24s)"], ["4", "4"], ["5", "5"], ["6", "6"], ["8", "8 — full (~64s)"],
];

interface Prog { text: string; err: boolean; indet: boolean; pct: number | null }

function prettyMsg(m?: string): string {
  if (!m) return "";
  const t = m.replace(/^beat (\d+):\s*/i, "Clip $1 — ");
  return t.charAt(0).toUpperCase() + t.slice(1);
}

export default function Home() {
  const [genre, setGenre] = useState("Psychological Horror");
  const [tone, setTone] = useState("Dread & isolation");
  const [scenes, setScenes] = useState("3");
  const [concept, setConcept] = useState("");

  const [live, setLive] = useState(false);
  const [modeNote, setModeNote] = useState("");

  const [busy, setBusy] = useState(false);
  const [prog, setProg] = useState<Prog | null>(null);
  const [script, setScript] = useState<TrailerScript | null>(null);
  const [tab, setTab] = useState<"script" | "trailer">("script");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [trailerReady, setTrailerReady] = useState(false);
  const [scriptHref, setScriptHref] = useState<string | null>(null);
  const [scriptName, setScriptName] = useState("trailer_script.json");

  const esRef = useRef<EventSource | null>(null);
  const doneRef = useRef(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---- backend status + config ----
  useEffect(() => {
    let alive = true;
    const ping = async () => {
      try {
        await api.health();
        if (alive) setLive(true);
      } catch {
        if (alive) setLive(false);
      }
    };
    const loadCfg = async () => {
      try {
        const c = await api.config();
        if (alive)
          setModeNote(
            `${c.mock ? "Mock mode · no API cost" : "Live mode · Gemini + Veo"} · ${c.veo_model}`,
          );
      } catch {}
    };
    ping();
    loadCfg();
    const t = setInterval(ping, 4000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  // ---- cleanup ----
  useEffect(() => {
    return () => {
      esRef.current?.close();
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const finish = useCallback((id: string) => {
    if (doneRef.current) return;
    doneRef.current = true;
    esRef.current?.close();
    esRef.current = null;
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    setVideoUrl(api.videoUrl(id));
    setTrailerReady(true);
    setTab("trailer");
    setProg({ text: "✨ Trailer ready — enjoy the premiere.", err: false, indet: false, pct: 1 });
    setBusy(false);
  }, []);

  const streamProgress = useCallback(
    (id: string) => {
      const es = new EventSource(api.eventsUrl(id));
      esRef.current = es;
      es.onmessage = (ev) => {
        let d: any;
        try {
          d = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (d.type === "close") {
          if (!doneRef.current) {
            es.close();
            esRef.current = null;
          }
          return;
        }
        if (typeof d.progress === "number") {
          setProg({ text: prettyMsg(d.message) || "Rendering…", err: false, indet: false, pct: d.progress });
        } else if (d.message) {
          setProg({ text: prettyMsg(d.message), err: false, indet: true, pct: null });
        }
        if (d.type === "final" || (d.type === "status" && d.status === "completed")) finish(id);
        if (d.type === "snapshot" && d.status === "completed") finish(id);
        if (d.type === "status" && d.status === "failed") {
          setProg({ text: "⚠ " + (d.error || "Render failed"), err: true, indet: false, pct: 0 });
          setTab("script");
          setBusy(false);
          es.close();
          esRef.current = null;
        }
      };
      es.onerror = () => {
        // SSE dropped — fall back to polling the job status.
        if (doneRef.current) return;
        es.close();
        esRef.current = null;
        if (pollRef.current) return;
        pollRef.current = setInterval(async () => {
          try {
            const j = await api.trailerStatus(id);
            if (typeof j.progress === "number")
              setProg({ text: prettyMsg(j.message) || "Rendering…", err: false, indet: false, pct: j.progress });
            if (j.status === "completed") finish(id);
            if (j.status === "failed") {
              if (pollRef.current) clearInterval(pollRef.current);
              pollRef.current = null;
              setProg({ text: "⚠ " + (j.error || "Render failed"), err: true, indet: false, pct: 0 });
              setTab("script");
              setBusy(false);
            }
          } catch {}
        }, 2000);
      };
    },
    [finish],
  );

  const generate = useCallback(async () => {
    if (busy) return;
    const idea = concept.trim();
    if (idea.length < 5) {
      setProg({ text: "⚠ Describe your idea in a sentence or two first.", err: true, indet: false, pct: 0 });
      setTab("script");
      return;
    }
    if (!live) {
      setProg({ text: "⚠ Backend is offline — start the server first.", err: true, indet: false, pct: 0 });
      setTab("script");
      return;
    }

    // reset
    doneRef.current = false;
    esRef.current?.close();
    esRef.current = null;
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = null;
    setBusy(true);
    setScript(null);
    setTrailerReady(false);
    setVideoUrl(null);
    setTab("script");

    try {
      setProg({ text: "Creating your project…", err: false, indet: true, pct: null });
      const proj = await api.createProject({
        concept: idea,
        genre,
        tone,
        n_segments: Number(scenes),
        seg_seconds: 8,
      });
      const id = proj.id;

      setProg({ text: "✍️ Writing your trailer script…", err: false, indet: true, pct: null });
      const sc = await api.generateScript(id);
      setScript(sc);
      const json = JSON.stringify(sc, null, 2);
      setScriptHref("data:application/json;charset=utf-8," + encodeURIComponent(json));
      setScriptName((sc.title || "trailer").replace(/[^a-z0-9]+/gi, "_").toLowerCase() + "_script.json");

      setProg({ text: "🎨 Storyboarding cinematic shots…", err: false, indet: true, pct: null });
      await api.buildPrompts(id);

      setProg({ text: "🎬 Rolling camera — generating clips…", err: false, indet: true, pct: null });
      await api.startTrailer(id, { use_anchor: true });
      streamProgress(id);
    } catch (e) {
      setProg({ text: "⚠ " + (e instanceof Error ? e.message : String(e)), err: true, indet: false, pct: 0 });
      setTab("script");
      setBusy(false);
    }
  }, [busy, concept, genre, tone, scenes, live, streamProgress]);

  return (
    <>
      <div className="bg">
        <span className="blob b1" />
        <span className="blob b2" />
        <span className="blob b3" />
      </div>

      <div className="app">
        <header>
          <div className="brand">
            <span className="icon" dangerouslySetInnerHTML={{ __html: ICON_SVG }} />
            <h1>Dreamers</h1>
          </div>
          <div className="tagline">AI Trailer Studio · Idea → Script → Film</div>
        </header>

        <main>
          {/* INPUT */}
          <section className="card">
            <h2>Your Vision</h2>
            <div className="form">
              <div className="row">
                <div>
                  <label htmlFor="genre">Genre</label>
                  <select id="genre" value={genre} onChange={(e) => setGenre(e.target.value)}>
                    {GENRES.map((g) => (
                      <option key={g}>{g}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label htmlFor="tone">Tone</label>
                  <select id="tone" value={tone} onChange={(e) => setTone(e.target.value)}>
                    {TONES.map((t) => (
                      <option key={t}>{t}</option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="row">
                <div>
                  <label htmlFor="scenes">Scenes</label>
                  <select id="scenes" value={scenes} onChange={(e) => setScenes(e.target.value)}>
                    {SCENES.map(([v, l]) => (
                      <option key={v} value={v}>
                        {l}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label htmlFor="pace">Pace</label>
                  <select id="pace" defaultValue="8">
                    <option value="8">8s clips (cinematic)</option>
                  </select>
                </div>
              </div>
              <div className="idea">
                <label htmlFor="concept">Describe your idea</label>
                <textarea
                  id="concept"
                  value={concept}
                  onChange={(e) => setConcept(e.target.value)}
                  onKeyDown={(e) => {
                    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generate();
                  }}
                  placeholder="A lighthouse keeper on a desolate island discovers the fog rolling in from the sea is alive — and hunting him…"
                />
              </div>
              <button className="go" onClick={generate} disabled={busy}>
                {busy && <span className="spin" />}
                {busy ? "Dreaming…" : "Generate Trailer"}
              </button>
            </div>
          </section>

          {/* OUTPUT */}
          <section className="card">
            <h2>Studio Output</h2>
            <div className="out">
              <div className="tabs">
                <button
                  className="tab"
                  aria-selected={tab === "script"}
                  onClick={() => setTab("script")}
                >
                  📝 Script
                </button>
                <button
                  className="tab"
                  aria-selected={tab === "trailer"}
                  disabled={!trailerReady}
                  onClick={() => trailerReady && setTab("trailer")}
                >
                  🎬 Trailer
                </button>
              </div>

              {prog && (
                <>
                  <div className="stage">
                    <span className={prog.err ? "err" : ""}>{prog.text}</span>
                  </div>
                  <div className={"bar" + (prog.indet ? " indet" : "")}>
                    <i style={{ width: prog.pct != null ? `${Math.round(prog.pct * 100)}%` : undefined }} />
                  </div>
                </>
              )}

              <div className="panes">
                {/* script */}
                <div className={"pane" + (tab === "script" ? " active" : "")}>
                  {!script ? (
                    <div className="empty">
                      <div className="big">🎞️</div>
                      <p>
                        Pick your genre &amp; tone, describe your idea, and hit <b>Generate</b>. Your
                        trailer script appears here first — then the film.
                      </p>
                    </div>
                  ) : (
                    <div className="script-scroll">
                      <h3 className="sc-title">{script.title}</h3>
                      <p className="sc-logline">{script.logline}</p>
                      {script.beats.map((b) => (
                        <div className="beat" key={b.beat_no}>
                          <div className="bh">
                            <span className="chip">{b.section}</span>
                            <span className="setting">{b.setting}</span>
                          </div>
                          <div className="visual">{b.visual}</div>
                          {b.voiceover && (
                            <div className="line">
                              <b>VO:</b> &ldquo;{b.voiceover}&rdquo;
                            </div>
                          )}
                          {b.dialogue && (
                            <div className="line">
                              <b>Dialogue:</b> &ldquo;{b.dialogue}&rdquo;
                            </div>
                          )}
                          {b.on_screen_text && <span className="ost">&ldquo;{b.on_screen_text}&rdquo;</span>}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* trailer */}
                <div className={"pane" + (tab === "trailer" ? " active" : "")}>
                  <div className="video-wrap">
                    {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
                    <video src={videoUrl ?? undefined} controls playsInline />
                  </div>
                  <div className="dl-row">
                    <a className="dl primary" href={videoUrl ?? "#"} download={`trailer.mp4`}>
                      ⬇ Download Trailer
                    </a>
                    <a className="dl ghost" href={scriptHref ?? "#"} download={scriptName}>
                      ⬇ Script (.json)
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </main>
        <div style={{ height: 6 }} />
      </div>

      <div className="footnote">{modeNote}</div>
      <div className={"status" + (live ? " live" : "")}>
        <span className="dot" />
        <span className="txt">{live ? "Backend Live" : "Backend Offline"}</span>
      </div>
    </>
  );
}
