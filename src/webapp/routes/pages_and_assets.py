"""
Static page routes and ivcam / stretch raw / stretch_pose_catalog APIs.
Registered from start_web.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse


def register_pages_and_asset_routes(app: FastAPI, project_root: Path, web_dir: Path) -> None:
    """Register /, detection pages, ivcam, stretch raw, catalog."""
    # 根路径：毕设主页面（整合导航）
    @app.get("/")
    async def root():
        web_file = web_dir / "index.html"
        if web_file.exists():
            return FileResponse(str(web_file))
        web_file = web_dir / "video_detection.html"
        if web_file.exists():
            return FileResponse(str(web_file))
        return {"message": "Ramen QC System", "status": "running", "web_interface": "/web/index.html"}

    # 流程检测
    def _no_cache_file_response(path: str):
        r = FileResponse(path)
        r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        r.headers["Pragma"] = "no-cache"
        return r

    @app.get("/stretch-detection")
    async def stretch_detection():
        """抻面动作检测"""
        web_file = web_dir / "video_detection.html"
        if web_file.exists():
            return _no_cache_file_response(str(web_file))
        return {"message": "页面未找到"}

    @app.get("/boiling-detection")
    async def boiling_detection():
        """下面及捞面检测"""
        web_file = web_dir / "boiling_scooping_detection.html"
        if web_file.exists():
            return _no_cache_file_response(str(web_file))
        return {"message": "页面未找到"}

    # 预处理视频页面
    @app.get("/video-skeleton")
    async def video_skeleton():
        """抻面预处理视频展示页面（带骨架线）"""
        web_file = web_dir / "video_with_skeleton.html"
        if web_file.exists():
            return FileResponse(str(web_file))
        return {"message": "预处理视频页面未找到"}

    @app.get("/video-skeleton-boiling")
    async def video_skeleton_boiling():
        """下面及捞面预处理视频展示页面（xl 带骨架线）"""
        web_file = web_dir / "video_with_skeleton_boiling.html"
        if web_file.exists():
            return FileResponse(str(web_file))
        return {"message": "下面及捞面预处理视频页面未找到"}

    @app.get("/realtime-monitor")
    async def realtime_monitor():
        """实时监测页面（摄像头 + 检测流）"""
        web_file = web_dir / "realtime_monitor.html"
        if web_file.exists():
            return FileResponse(str(web_file))
        return {"message": "实时监测页面未找到"}

    @app.get("/realtime-skeleton")
    async def realtime_skeleton():
        """旧入口：已与「实时监测」合并（检测+骨架双画面），避免重复页面。"""
        return RedirectResponse(url="/realtime-monitor", status_code=302)

    # ivcam 目录：检测框与骨架线分两个子文件夹（data/ivcam/检测框、data/ivcam/骨架线）
    ivcam_dir = project_root / "data" / "ivcam"
    ivcam_subdirs = ("检测框", "骨架线")
    _video_ext = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    _image_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    def _ivcam_collect(subdir_path):
        videos, images = [], []
        if not subdir_path.exists():
            return videos, images
        for f in subdir_path.iterdir():
            if not f.is_file():
                continue
            suf = f.suffix.lower()
            if suf in _video_ext:
                videos.append(f.name)
            elif suf in _image_ext:
                images.append(f.name)
        videos.sort()
        images.sort()
        return videos, images

    @app.get("/api/ivcam/list")
    async def api_ivcam_list():
        """列出 data/ivcam 下「检测框」「骨架线」两个子目录中的视频与图片。"""
        ivcam_dir.mkdir(parents=True, exist_ok=True)
        for sub in ivcam_subdirs:
            (ivcam_dir / sub).mkdir(parents=True, exist_ok=True)
        detection_v, detection_i = _ivcam_collect(ivcam_dir / "检测框")
        skeleton_v, skeleton_i = _ivcam_collect(ivcam_dir / "骨架线")
        return {
            "success": True,
            "detection": {"videos": detection_v, "images": detection_i},
            "skeleton": {"videos": skeleton_v, "images": skeleton_i},
            "videos": detection_v + skeleton_v,
            "images": detection_i + skeleton_i,
        }

    @app.get("/api/ivcam/file/{file_path:path}")
    async def api_ivcam_file(file_path: str):
        """返回 data/ivcam 下文件内容。file_path 可为「检测框/xxx.mp4」或「骨架线/xxx.mp4」，禁止路径穿越。"""
        from urllib.parse import unquote
        file_path = unquote(file_path or "")
        if not file_path or ".." in file_path:
            raise HTTPException(status_code=400, detail="invalid path")
        parts = file_path.replace("\\", "/").strip("/").split("/")
        if len(parts) == 1:
            subdir, name = "", parts[0]
        elif len(parts) == 2 and parts[0] in ivcam_subdirs:
            subdir, name = parts[0], parts[1]
        else:
            raise HTTPException(status_code=400, detail="invalid path")
        if not name or "/" in name:
            raise HTTPException(status_code=400, detail="invalid filename")
        if subdir:
            path = ivcam_dir / subdir / name
        else:
            path = ivcam_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        try:
            path.resolve().relative_to(ivcam_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(str(path), filename=name)

    # data/raw/抻面 示范视频：与 ivcam 并列，用于骨架线页直接加载 cm13～cm17 等本地 mp4
    import re as _re_stretch_raw
    stretch_raw_抻面_dir = project_root / "data" / "raw" / "抻面"
    _stretch_demo_name_pat = _re_stretch_raw.compile(r"^cm\d+$", _re_stretch_raw.I)

    def _stretch_raw_cm_sort_key(name: str):
        m = _re_stretch_raw.search(r"(\d+)$", name, _re_stretch_raw.I)
        return (int(m.group(1)) if m else 0, name.lower())

    @app.get("/api/stretch_raw_videos/list")
    async def api_stretch_raw_videos_list():
        """列出 data/raw/抻面 下 cm*.mp4，供骨架线示范页与综合评分数据准备。"""
        out = []
        if stretch_raw_抻面_dir.exists():
            for f in stretch_raw_抻面_dir.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() not in (".mp4", ".mov", ".avi", ".mkv"):
                    continue
                stem = f.stem
                if _stretch_demo_name_pat.match(stem):
                    out.append(stem)
        out.sort(key=_stretch_raw_cm_sort_key)
        return {"success": True, "videos": out}

    @app.get("/api/stretch_raw_video/file/{video_name}")
    async def api_stretch_raw_video_file(video_name: str):
        """安全返回 data/raw/抻面 下视频（仅允许 cm+数字  stem）。"""
        from urllib.parse import unquote
        name = unquote(video_name or "").strip()
        if not name or not _stretch_demo_name_pat.match(name):
            raise HTTPException(status_code=400, detail="invalid video name")
        for ext in (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"):
            path = stretch_raw_抻面_dir / f"{name}{ext}"
            if path.is_file():
                try:
                    path.resolve().relative_to(stretch_raw_抻面_dir.resolve())
                except ValueError:
                    raise HTTPException(status_code=404, detail="file not found")
                return FileResponse(str(path), filename=path.name, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
        raise HTTPException(status_code=404, detail="file not found")

    @app.get("/api/stretch_pose_catalog")
    async def api_stretch_pose_catalog():
        """抻面预处理/骨架页用：cm1～cm17 各自是否已有原片、骨架 JSON、预渲染带骨架 mp4。"""
        processed_dir = project_root / "data" / "processed_videos" / "抻面"
        raw_dir = project_root / "data" / "raw" / "抻面"
        kp_dir = project_root / "data" / "scores" / "抻面" / "hand_keypoints"
        items = []
        for i in range(1, 18):
            name = f"cm{i}"
            has_p = (processed_dir / f"{name}_with_skeleton.mp4").is_file()
            has_raw = False
            if raw_dir.exists():
                for ext in (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"):
                    if (raw_dir / f"{name}{ext}").is_file():
                        has_raw = True
                        break
            has_kp = (kp_dir / f"hand_keypoints_{name}.json").is_file() if kp_dir.exists() else False
            if has_p:
                label = "已预处理"
            elif has_raw and has_kp:
                label = "可生成预处理"
            elif has_raw:
                label = "有原片"
            else:
                label = "未检测到文件"
            items.append({
                "name": name,
                "has_processed": has_p,
                "has_raw": has_raw,
                "has_keypoints": has_kp,
                "label": label,
            })
        return {"success": True, "videos": items}
