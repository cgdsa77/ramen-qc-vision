"""异步检测/评分任务状态（由 start_web 与各路由共享）。"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

_detect_jobs: Dict[str, Any] = {}
_score_jobs: Dict[str, Any] = {}
_upload_score_result: Any = None
_upload_score_stage: Optional[str] = None
_jobs_lock = threading.Lock()
_MAX_BG_JOBS = int(os.environ.get("RAMEN_MAX_BG_JOBS", "200"))


def prune_job_store(store: Dict[str, Any]) -> None:
    while len(store) > _MAX_BG_JOBS:
        k = next(iter(store))
        del store[k]


def env_default_async_detect() -> bool:
    return os.environ.get("RAMEN_DEFAULT_ASYNC_DETECT", "").lower() in ("1", "true", "yes", "on")


def resolve_async_mode(async_mode: Optional[bool]) -> bool:
    if async_mode is None:
        return env_default_async_detect()
    return bool(async_mode)
