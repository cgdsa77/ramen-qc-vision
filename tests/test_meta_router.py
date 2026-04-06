from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.webapp.routers.meta import build_meta_router


def test_health_and_runtime_info():
    app = FastAPI()
    app.include_router(build_meta_router(Path(__file__).resolve().parents[1]))
    c = TestClient(app)
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
    r2 = c.get("/api/runtime-info")
    assert r2.status_code == 200
    body = r2.json()
    assert "cors_origins_wildcard" in body
