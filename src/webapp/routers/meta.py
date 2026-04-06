"""
元信息路由（中期拆分示例）：健康检查与运行时配置（无敏感信息）
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from src.webapp.config import api_session_required, get_cors_origins


def build_meta_router(project_root: Path) -> APIRouter:
    r = APIRouter(tags=["meta"])

    @r.get("/api/health")
    async def health():
        return {"status": "ok", "project": str(project_root)}

    @r.get("/api/runtime-info")
    async def runtime_info():
        origins = get_cors_origins()
        return {
            "cors_origins_wildcard": origins == ["*"],
            "cors_origin_count": len(origins),
            "api_session_enforcement": api_session_required(),
        }

    return r
