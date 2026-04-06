"""
Web 服务配置（中期：CORS / 可选 API 会话校验）

环境变量:
  RAMEN_CORS_ORIGINS
    - 未设置或为空: 等价于 ["*"]（与改前一致，本地开发最省事）
    - 逗号分隔列表: 如 http://127.0.0.1:8000,http://localhost:5173
    - 单一路径写完整 origin（含协议与端口）

  RAMEN_REQUIRE_SESSION_FOR_API
    - 设为 1/true/on 时: 对 /api/* 要求已登录（Cookie ramen_session），
      白名单见 _API_SESSION_EXEMPT_PREFIXES（登录、文档、部分公开接口）
"""
from __future__ import annotations

import os
from typing import List, Tuple


def get_cors_origins() -> List[str]:
    raw = (os.environ.get("RAMEN_CORS_ORIGINS") or "").strip()
    if not raw:
        return ["*"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts if parts else ["*"]


def cors_allows_credentials(origins: List[str]) -> bool:
    """浏览器规范：origins 含 * 时不应与 allow_credentials=True 同用；由调用方降级 credentials。"""
    return "*" not in origins


# 当 RAMEN_REQUIRE_SESSION_FOR_API=1 时，以下路径前缀不要求 Session（需与前端实际访问路径一致）
_API_SESSION_EXEMPT_PREFIXES: Tuple[str, ...] = (
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/session",
    "/api/health",
    "/api/runtime-info",
    "/docs",
    "/openapi.json",
    "/redoc",
)


def api_session_required() -> bool:
    return os.environ.get("RAMEN_REQUIRE_SESSION_FOR_API", "").lower() in ("1", "true", "yes", "on")


def is_path_exempt_from_session(path: str) -> bool:
    if path in ("/", "/favicon.ico"):
        return True
    for prefix in _API_SESSION_EXEMPT_PREFIXES:
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
            return True
    # 静态与页面路由（无 /api 前缀的由 Starlette 处理；此处只拦 /api/*）
    return False
