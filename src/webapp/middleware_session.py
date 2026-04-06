"""
可选：整站要求登录后才能调大部分 /api（除白名单）。
启用: 环境变量 RAMEN_REQUIRE_SESSION_FOR_API=1
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.webapp.config import api_session_required, is_path_exempt_from_session


def build_session_enforcement_middleware():
    if not api_session_required():
        return None

    class SessionEnforcementMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            if not path.startswith("/api/") or is_path_exempt_from_session(path):
                return await call_next(request)
            try:
                from src import auth_db

                sid = request.cookies.get("ramen_session")
                user = auth_db.get_session(sid) if sid else None
                if not user:
                    return JSONResponse(
                        status_code=401,
                        content={"success": False, "error": "unauthorized", "message": "请先登录后再调用此接口"},
                    )
            except Exception:
                return JSONResponse(
                    status_code=503,
                    content={"success": False, "error": "auth_unavailable", "message": "认证模块不可用"},
                )
            return await call_next(request)

    return SessionEnforcementMiddleware
