"""Session 解析（供需登录的路由与拆分出的路由模块共用）。"""
from __future__ import annotations

from typing import Optional

from fastapi import Request

try:
    from src import auth_db
except ImportError:
    auth_db = None  # type: ignore


def auth_session_id(request: Request) -> Optional[str]:
    return request.cookies.get("ramen_session")


def auth_current_user(request: Request):
    if auth_db is None:
        return None
    uid = auth_session_id(request)
    return auth_db.get_session(uid) if uid else None
