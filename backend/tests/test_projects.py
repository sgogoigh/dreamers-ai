def test_create_applies_defaults(client):
    r = client.post("/api/projects", json={"concept": "Robots learn to dream"})
    assert r.status_code == 201
    p = r.json()
    assert p["id"] and len(p["id"]) == 12
    assert p["status"] == "created"
    assert p["n_segments"] >= 1   # default applied (0 -> server default)
    assert p["seg_seconds"] >= 1


def test_create_rejects_short_concept(client):
    r = client.post("/api/projects", json={"concept": "x"})
    assert r.status_code == 422


def test_get_and_list(client, project_id):
    r = client.get(f"/api/projects/{project_id}")
    assert r.status_code == 200
    assert r.json()["id"] == project_id

    listing = client.get("/api/projects").json()
    assert any(item["id"] == project_id for item in listing)
    summary = next(item for item in listing if item["id"] == project_id)
    for key in ("has_script", "has_prompts", "has_video", "status"):
        assert key in summary


def test_get_missing_is_404(client):
    assert client.get("/api/projects/deadbeefcafe").status_code == 404


def test_bad_id_is_404(client):
    # malformed id must not 500 or traverse paths
    assert client.get("/api/projects/../etc").status_code in (404, 422)
    assert client.get("/api/projects/zzz").status_code == 404


def test_delete(client):
    pid = client.post("/api/projects", json={"concept": "to be deleted soon"}).json()["id"]
    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert client.get(f"/api/projects/{pid}").status_code == 404
