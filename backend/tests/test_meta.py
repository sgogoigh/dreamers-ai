def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["mock"] is True


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["docs"] == "/docs"


def test_config_exposes_capabilities(client):
    cfg = client.get("/api/config").json()
    # Frontend relies on these fields to render capabilities/defaults.
    for key in (
        "mock", "gemini_available", "adapter_available", "gemini_model",
        "veo_model", "default_segments", "default_seg_seconds", "resolution",
        "aspect_ratio",
    ):
        assert key in cfg, key
    assert cfg["gemini_model"]
    assert cfg["veo_model"]
    assert cfg["default_seg_seconds"] <= 8  # Veo clip cap
