import os

import pytest

from src.webapp.config import cors_allows_credentials, get_cors_origins, is_path_exempt_from_session


def test_cors_default_is_wildcard(monkeypatch):
    monkeypatch.delenv("RAMEN_CORS_ORIGINS", raising=False)
    assert get_cors_origins() == ["*"]


def test_cors_custom_list(monkeypatch):
    monkeypatch.setenv("RAMEN_CORS_ORIGINS", "http://127.0.0.1:8000, http://localhost:3000 ")
    assert get_cors_origins() == ["http://127.0.0.1:8000", "http://localhost:3000"]


def test_cors_credentials_rule():
    assert cors_allows_credentials(["*"]) is False
    assert cors_allows_credentials(["http://a"]) is True


def test_session_exempt_paths():
    assert is_path_exempt_from_session("/api/auth/login") is True
    assert is_path_exempt_from_session("/api/health") is True
    assert is_path_exempt_from_session("/docs") is True
    assert is_path_exempt_from_session("/api/detect_video") is False
