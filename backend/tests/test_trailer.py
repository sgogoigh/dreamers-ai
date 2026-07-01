"""Step 4: async video job lifecycle, artifact serving, and SSE progress.

The video job runs as an asyncio background task (`create_task` + `to_thread`).
The sync Starlette TestClient's portal loop doesn't reliably pump such tasks to
completion between requests, so the job-driving tests here run against the ASGI
app through an async httpx client on a single, continuously-running event loop —
exactly how uvicorn behaves in production.
"""
import asyncio
import json

import httpx
import pytest
from httpx import ASGITransport

from app.main import app


def _run(coro):
    return asyncio.run(coro)


async def _async_client():
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _make_project_with_prompts(ac) -> str:
    r = await ac.post("/api/projects", json={
        "concept": "A lighthouse keeper discovers the fog is alive",
        "genre": "Psychological Horror", "tone": "dread", "n_segments": 3,
    })
    pid = r.json()["id"]
    assert (await ac.post(f"/api/projects/{pid}/script")).status_code == 200
    assert (await ac.post(f"/api/projects/{pid}/prompts")).status_code == 200
    return pid


async def _poll_until_done(ac, pid, timeout=90):
    for _ in range(int(timeout * 2)):
        job = (await ac.get(f"/api/projects/{pid}/trailer")).json()
        if job["status"] in ("completed", "failed"):
            return job
        await asyncio.sleep(0.5)
    raise AssertionError(f"job did not finish; last={job}")


# --- guards (sync TestClient is fine here — no background work) ------------
def test_trailer_requires_prompts(client, project_id):
    assert client.post(f"/api/projects/{project_id}/trailer", json={}).status_code == 409


def test_trailer_status_404_before_start(client, project_with_prompts):
    assert client.get(f"/api/projects/{project_with_prompts}/trailer").status_code == 404


# --- full async flow -------------------------------------------------------
def test_full_trailer_flow_and_serving():
    async def scenario():
        async with await _async_client() as ac:
            pid = await _make_project_with_prompts(ac)

            r = await ac.post(f"/api/projects/{pid}/trailer", json={"use_anchor": True})
            assert r.status_code == 202
            job = r.json()
            assert job["status"] in ("pending", "running")
            assert len(job["segments"]) == 8              # blueprint-driven
            assert job["segments"][-1]["is_title_card"] is True

            final = await _poll_until_done(ac, pid)
            assert final["status"] == "completed", final.get("error")
            assert final["progress"] == 1.0
            assert final["video_url"]
            assert all(s["status"] == "completed" for s in final["segments"])
            assert all(s["video_url"] for s in final["segments"])
            # the composed title card is served like any other segment
            title_seg = await ac.get(f"/api/projects/{pid}/segments/8")
            assert title_seg.status_code == 200 and len(title_seg.content) > 500

            v = await ac.get(f"/api/projects/{pid}/trailer/video")
            assert v.status_code == 200
            assert v.headers["content-type"] == "video/mp4"
            assert len(v.content) > 1000

            seg = await ac.get(f"/api/projects/{pid}/segments/1")
            assert seg.status_code == 200 and len(seg.content) > 500
            assert (await ac.get(f"/api/projects/{pid}/segments/99")).status_code == 404

            assert (await ac.get(f"/api/projects/{pid}")).json()["status"] == "completed"

            # restarting after completion is allowed (reuses existing clips)
            assert (await ac.post(f"/api/projects/{pid}/trailer", json={})).status_code == 202
            await _poll_until_done(ac, pid)

    _run(scenario())


def test_sse_snapshot_after_completion():
    async def scenario():
        async with await _async_client() as ac:
            pid = await _make_project_with_prompts(ac)
            await ac.post(f"/api/projects/{pid}/trailer", json={})
            await _poll_until_done(ac, pid)

            events = []
            async with ac.stream("GET", f"/api/projects/{pid}/trailer/events") as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    events.append(json.loads(line[len("data:"):].strip()))
                    if events[-1].get("type") == "close":
                        break
            types = [e["type"] for e in events]
            assert "snapshot" in types
            assert events[0]["status"] == "completed"
            assert "close" in types

    _run(scenario())


# --- live publish stream + concurrency guard (drive manager directly) ------
def test_live_sse_event_stream_via_job_manager(client, project_with_prompts):
    from app import storage
    from app.jobs import job_manager

    pid = project_with_prompts
    project = storage.load(pid)

    async def scenario():
        q = job_manager.subscribe(pid)
        await job_manager.start(project, native_extend=False, use_anchor=True)
        collected = []
        while True:
            ev = await asyncio.wait_for(q.get(), timeout=90)
            collected.append(ev)
            if ev.get("type") == "close":
                break
        return collected

    events = _run(scenario())
    types = [e["type"] for e in events]
    assert "segment" in types
    assert "final" in types
    assert types[-1] == "close"
    progresses = [e["progress"] for e in events if "progress" in e]
    assert progresses == sorted(progresses)
    assert progresses[-1] == 1.0


def test_cannot_start_while_running(client, project_with_prompts):
    from app import storage
    from app.jobs import job_manager

    pid = project_with_prompts
    project = storage.load(pid)

    async def scenario():
        await job_manager.start(project, native_extend=False, use_anchor=True)
        running = job_manager.is_running(pid)
        for _ in range(360):
            if not job_manager.is_running(pid):
                break
            await asyncio.sleep(0.5)
        return running

    assert _run(scenario()) is True
