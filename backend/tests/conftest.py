"""Test config: force MOCK mode + an isolated temp data dir BEFORE the app loads.

`app.config.Settings` reads env at import time, so these must be set before any
`app.*` import. Importing this conftest first (pytest does) guarantees that.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ["DREAMERS_MOCK"] = "1"
_TMP = Path(tempfile.mkdtemp(prefix="dreamers_test_"))
os.environ["DREAMERS_DATA_DIR"] = str(_TMP)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def project_id(client: TestClient) -> str:
    r = client.post(
        "/api/projects",
        json={
            "concept": "A lighthouse keeper discovers the fog is alive",
            "genre": "Psychological Horror",
            "tone": "dread, isolation",
            "n_segments": 3,
        },
    )
    assert r.status_code == 201
    return r.json()["id"]


@pytest.fixture
def project_with_prompts(client: TestClient, project_id: str) -> str:
    assert client.post(f"/api/projects/{project_id}/script").status_code == 200
    assert client.post(f"/api/projects/{project_id}/prompts").status_code == 200
    return project_id
