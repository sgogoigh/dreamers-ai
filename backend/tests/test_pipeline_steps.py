"""Steps 1-3: draft, script (+ revise + manual edit), prompts — and their guards."""


# --- step 1: draft ---------------------------------------------------------
def test_manual_draft(client, project_id):
    r = client.post(f"/api/projects/{project_id}/draft",
                    json={"mode": "manual", "text": "INT. TOWER - NIGHT\nThomas waits."})
    assert r.status_code == 200
    p = r.json()
    assert p["draft_source"] == "manual"
    assert "Thomas" in p["draft"]
    assert p["status"] == "drafted"


def test_manual_draft_requires_text(client, project_id):
    r = client.post(f"/api/projects/{project_id}/draft", json={"mode": "manual", "text": "  "})
    assert r.status_code == 422


def test_adapter_draft_in_mock_returns_seed(client, project_id):
    # In MOCK mode the adapter is "available" and returns a deterministic seed.
    r = client.post(f"/api/projects/{project_id}/draft", json={"mode": "adapter"})
    assert r.status_code == 200
    assert r.json()["draft_source"] == "adapter"
    assert r.json()["draft"]


# --- step 2: script --------------------------------------------------------
def test_generate_script_shape(client, project_id):
    r = client.post(f"/api/projects/{project_id}/script")
    assert r.status_code == 200
    s = r.json()
    assert s["title"] and s["logline"] and s["beats"]
    assert len(s["beats"]) == 3                       # honored n_segments
    assert s["beats"][0]["continues_previous"] is False
    assert [b["beat_no"] for b in s["beats"]] == [1, 2, 3]  # renumbered


def test_revise_requires_existing_script(client, project_id):
    r = client.post(f"/api/projects/{project_id}/script/revise",
                    json={"feedback": "make it scarier"})
    assert r.status_code == 409


def test_revise_records_history(client, project_id):
    client.post(f"/api/projects/{project_id}/script")
    r = client.post(f"/api/projects/{project_id}/script/revise",
                    json={"feedback": "make the ending more hopeful"})
    assert r.status_code == 200
    proj = client.get(f"/api/projects/{project_id}").json()
    assert any(rev["source"] == "ai" and "hopeful" in rev["feedback"]
               for rev in proj["revisions"])


def test_manual_script_edit_revalidates_and_renumbers(client, project_id):
    script = client.post(f"/api/projects/{project_id}/script").json()
    # scramble beat numbers + flip first continues flag; server must fix both
    script["beats"][0]["beat_no"] = 99
    script["beats"][0]["continues_previous"] = True
    script["title"] = "USER EDITED TITLE"
    r = client.put(f"/api/projects/{project_id}/script", json=script)
    assert r.status_code == 200
    out = r.json()
    assert out["title"] == "USER EDITED TITLE"
    assert out["beats"][0]["beat_no"] == 1
    assert out["beats"][0]["continues_previous"] is False
    proj = client.get(f"/api/projects/{project_id}").json()
    assert any(rev["source"] == "manual" for rev in proj["revisions"])


def test_invalid_section_rejected_on_manual_edit(client, project_id):
    script = client.post(f"/api/projects/{project_id}/script").json()
    script["beats"][0]["section"] = "not_a_real_section"
    r = client.put(f"/api/projects/{project_id}/script", json=script)
    assert r.status_code == 422


# --- step 3: prompts -------------------------------------------------------
def test_prompts_require_script(client, project_id):
    assert client.post(f"/api/projects/{project_id}/prompts").status_code == 409


def test_build_prompts_aligns_with_script(client, project_id):
    client.post(f"/api/projects/{project_id}/script")
    r = client.post(f"/api/projects/{project_id}/prompts")
    assert r.status_code == 200
    segs = r.json()["segments"]
    assert len(segs) == 3
    assert segs[0]["continues_previous"] is False
    assert all(s["prompt"] for s in segs)


def test_generating_script_invalidates_prompts(client, project_id):
    client.post(f"/api/projects/{project_id}/script")
    client.post(f"/api/projects/{project_id}/prompts")
    assert client.get(f"/api/projects/{project_id}").json()["prompts"] is not None
    # regenerating the script should clear stale prompts
    client.post(f"/api/projects/{project_id}/script")
    assert client.get(f"/api/projects/{project_id}").json()["prompts"] is None


def test_edit_prompts(client, project_id):
    client.post(f"/api/projects/{project_id}/script")
    prompts = client.post(f"/api/projects/{project_id}/prompts").json()
    prompts["segments"][1]["prompt"] = "EDITED PROMPT TEXT"
    r = client.put(f"/api/projects/{project_id}/prompts", json=prompts)
    assert r.status_code == 200
    assert r.json()["segments"][1]["prompt"] == "EDITED PROMPT TEXT"
