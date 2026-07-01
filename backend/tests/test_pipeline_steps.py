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


# --- step 1.5: blueprint ---------------------------------------------------
def test_default_blueprint_shape(client, project_id):
    r = client.get(f"/api/projects/{project_id}/blueprint")
    assert r.status_code == 200
    slots = r.json()["slots"]
    assert len(slots) == 8
    assert [s["beat_no"] for s in slots] == list(range(1, 9))
    assert slots[0]["transition_in"] == "fade_in"
    assert slots[1]["continues_previous"] is True       # slot 2 continues slot 1
    assert slots[3]["target_seconds"] == 6              # slot 4 slow-down (5s -> 6)
    assert slots[7]["is_title_card"] is True
    assert slots[7]["transition_out"] == "fade_out"
    assert slots[7]["target_seconds"] == 4


def test_edit_blueprint_invalidates_downstream_and_renumbers(client, project_id):
    client.post(f"/api/projects/{project_id}/script")
    bp = client.get(f"/api/projects/{project_id}/blueprint").json()
    bp["slots"] = bp["slots"][:4]                       # trim to 4 slots
    bp["slots"][0]["beat_no"] = 99                      # server must renumber
    r = client.put(f"/api/projects/{project_id}/blueprint", json=bp)
    assert r.status_code == 200
    out = r.json()
    assert [s["beat_no"] for s in out["slots"]] == [1, 2, 3, 4]
    proj = client.get(f"/api/projects/{project_id}").json()
    assert proj["n_segments"] == 4
    assert proj["script"] is None and proj["prompts"] is None  # invalidated


# --- step 2: script --------------------------------------------------------
def test_generate_script_shape(client, project_id):
    r = client.post(f"/api/projects/{project_id}/script")
    assert r.status_code == 200
    s = r.json()
    assert s["title"] and s["logline"] and s["beats"]
    assert len(s["beats"]) == 8                        # blueprint-driven (8 slots)
    assert [b["beat_no"] for b in s["beats"]] == list(range(1, 9))  # renumbered
    assert s["cast"]                                    # a consistent cast is emitted
    # structural fields stamped from the blueprint
    assert s["beats"][0]["continues_previous"] is False
    assert s["beats"][0]["transition_in"] == "fade_in"
    assert s["beats"][1]["continues_previous"] is True
    assert s["beats"][3]["target_seconds"] == 6
    assert s["beats"][7]["is_title_card"] is True
    assert s["beats"][7]["on_screen_text"] == s["title"]


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
    assert len(segs) == 8
    assert segs[0]["continues_previous"] is False
    assert all(s["prompt"] for s in segs)
    # structural fields carried from the beats/blueprint
    assert segs[0]["transition_in"] == "fade_in"
    assert segs[3]["duration_seconds"] == 6
    assert segs[7]["is_title_card"] is True
    assert segs[7]["duration_seconds"] == 4
    assert segs[7]["title_text"]                       # title propagated for rendering
    assert all(s["duration_seconds"] in (4, 6, 8) for s in segs)  # Veo-valid


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
