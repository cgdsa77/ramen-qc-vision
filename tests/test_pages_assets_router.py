from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.webapp.routes.pages_and_assets import register_pages_and_asset_routes


def test_stretch_pose_catalog_structure():
    app = FastAPI()
    pr = Path(__file__).resolve().parents[1]
    web = pr / "web"
    register_pages_and_asset_routes(app, pr, web)
    c = TestClient(app)
    r = c.get("/api/stretch_pose_catalog")
    assert r.status_code == 200
    data = r.json()
    assert data.get("success") is True
    assert isinstance(data.get("videos"), list)
    if data["videos"]:
        assert "name" in data["videos"][0]
