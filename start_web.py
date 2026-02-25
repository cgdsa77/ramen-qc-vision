#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""启动Web服务器 - 简化版"""
import json
import os
import sys
import threading
import uuid
from pathlib import Path

# 异步检测/评分任务状态（job_id -> 状态字典），供前端轮询进度
_detect_jobs = {}
_score_jobs = {}
_upload_score_result = None   # 最近一次上传视频的评分结果，供综合评分页 source=upload 展示
_upload_score_stage = None    # "stretch" | "boiling_scooping"
_jobs_lock = threading.Lock()

# 项目根目录 = start_web.py 所在目录（不依赖当前工作目录）
_script_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
os.chdir(_script_dir)
project_root = Path(_script_dir)
sys.path.insert(0, str(project_root))

# 启动时即加载 AI API 密钥（与 start_web.py 同目录的 configs）
_ai_api_key_from_file = None
_ai_config_load_error = None
_secret_path = project_root / "configs" / "ai_api_secret.json"
if _secret_path.exists():
    try:
        with open(_secret_path, "r", encoding="utf-8-sig") as f:
            _data = json.load(f)
            _raw = (_data.get("api_key") or _data.get("secret_key") or "")
            _ai_api_key_from_file = _raw.strip() if isinstance(_raw, str) else None
    except Exception as e:
        _ai_config_load_error = str(e)
        import traceback
        traceback.print_exc()
else:
    _ai_config_load_error = f"文件不存在: {_secret_path}"

print("="*60)
print("启动Ramen QC检测系统")
print("="*60)
if _ai_api_key_from_file:
    print(f"[AI] 已从 {_secret_path} 加载 api_key，AI 分析可用")
else:
    print(f"[AI] 未加载 api_key。路径: {_secret_path}，原因: {_ai_config_load_error or '文件内无 api_key/secret_key'}")

try:
    # 导入FastAPI
    from fastapi import FastAPI, UploadFile, File, Request, Query, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    import json
    
    # 创建应用
    app = FastAPI(title="Ramen QC API")

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc), "message": "服务器内部错误: " + str(exc)}
        )

    # CORS支持
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 静态文件
    web_dir = project_root / "web"
    reports_dir = project_root / "reports"
    videos_dir = project_root / "data" / "raw" / "抻面"
    processed_videos_dir = project_root / "data" / "processed_videos"
    scores_dir = project_root / "data" / "scores" / "抻面"
    reports_dir.mkdir(exist_ok=True)
    
    if web_dir.exists():
        app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")
    
    app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")
    
    # 评分数据服务
    if scores_dir.exists():
        hand_keypoints_dir = scores_dir / "hand_keypoints"
        if hand_keypoints_dir.exists():
            app.mount("/data/scores/抻面/hand_keypoints", StaticFiles(directory=str(hand_keypoints_dir)), name="hand_keypoints")
    
    # 视频文件服务
    if videos_dir.exists():
        app.mount("/data/videos/抻面", StaticFiles(directory=str(videos_dir)), name="videos")
    
    # 处理好的视频文件服务（带骨架线）
    if processed_videos_dir.exists():
        stretch_processed_dir = processed_videos_dir / "抻面"
        boiling_processed_dir = processed_videos_dir / "下面及捞面"
        if stretch_processed_dir.exists():
            print(f"[INFO] 注册静态文件服务: /data/processed_videos/抻面 -> {stretch_processed_dir}")
            app.mount("/data/processed_videos/抻面", StaticFiles(directory=str(stretch_processed_dir)), name="processed_videos_stretch")
        if boiling_processed_dir.exists():
            print(f"[INFO] 注册静态文件服务: /data/processed_videos/下面及捞面 -> {boiling_processed_dir}")
            app.mount("/data/processed_videos/下面及捞面", StaticFiles(directory=str(boiling_processed_dir)), name="processed_videos_boiling")
    
    # 拉面成品图片目录（用于成品评估标注）
    raw_lmcp_dir = project_root / "data" / "raw" / "拉面成品"
    product_scores_dir = project_root / "data" / "scores" / "拉面成品"
    
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

    # ivcam 目录：使用 ivcam 录制/拍摄的视频或图片（data/ivcam）
    ivcam_dir = project_root / "data" / "ivcam"
    _video_ext = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    _image_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    @app.get("/api/ivcam/list")
    async def api_ivcam_list():
        """列出 data/ivcam 下的视频与图片文件（仅文件名）。"""
        ivcam_dir.mkdir(parents=True, exist_ok=True)
        videos, images = [], []
        for f in ivcam_dir.iterdir():
            if not f.is_file():
                continue
            suf = f.suffix.lower()
            if suf in _video_ext:
                videos.append(f.name)
            elif suf in _image_ext:
                images.append(f.name)
        videos.sort()
        images.sort()
        return {"success": True, "videos": videos, "images": images}

    @app.get("/api/ivcam/file/{filename}")
    async def api_ivcam_file(filename: str):
        """返回 data/ivcam 下的文件内容，用于前端拉取后提交检测。仅允许单层文件名，禁止路径穿越。"""
        if not filename or ".." in filename or "/" in filename.replace("\\", "/"):
            raise HTTPException(status_code=400, detail="invalid filename")
        name = Path(filename).name
        path = ivcam_dir / name
        if not path.is_file() or path.resolve().parent != ivcam_dir.resolve():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(str(path), filename=name)

    # 实时流取消标记：前端点击「停止」时设置，生成器检查后退出并释放摄像头
    _realtime_stream_cancel = {}

    def _realtime_stream_generator(device_index: int, stage: str, conf: float, stream_id: str, use_mediapipe_hands: bool = True, use_mediapipe_face: bool = True, model_type: str = "cpu"):
        """生成 MJPEG 流：打开摄像头，逐帧检测并绘制；收到 stream_id 取消信号时退出并 release。model_type: cpu|gpu 预留。"""
        import cv2
        import sys
        cap = None
        try:
            # Windows 下优先用 DirectShow，兼容性更好，避免默认后端打不开摄像头
            if sys.platform == "win32":
                cap = cv2.VideoCapture(int(device_index), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(int(device_index))
            if not cap.isOpened():
                # Windows 下若 DirectShow 失败，再试一次默认后端（部分环境仅默认可用）
                if sys.platform == "win32":
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = cv2.VideoCapture(int(device_index))
                if not cap.isOpened():
                    print("[realtime-stream] 无法打开摄像头 device=%s，可能被占用或驱动异常" % device_index)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"\r\n")
                    return
            if stage == "boiling_scooping":
                from src.api.video_detection_api import get_boiling_scooping_detector
                detector = get_boiling_scooping_detector(model_type=model_type if model_type in ("cpu", "gpu") else "cpu")
            else:
                from src.api.video_detection_api import get_detector
                detector = get_detector(model_type=model_type if model_type in ("cpu", "gpu") else "cpu")
            while True:
                if _realtime_stream_cancel.get(stream_id):
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                # 在耗时推理前再次检查，避免点击「停止」后仍卡在 detect_frame 里不释放摄像头
                if _realtime_stream_cancel.get(stream_id):
                    break
                annotated, _ = detector.detect_frame(frame, conf_threshold=conf, draw_boxes=True, use_mediapipe_hands=use_mediapipe_hands, use_mediapipe_face=use_mediapipe_face, stage=stage)
                if _realtime_stream_cancel.get(stream_id):
                    break
                _, jpeg = cv2.imencode(".jpg", annotated)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg.tobytes() + b"\r\n")
        except GeneratorExit:
            pass
        except Exception as e:
            print(f"[realtime-stream] 错误: {e}")
        finally:
            _realtime_stream_cancel.pop(stream_id, None)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                print("[realtime-stream] 摄像头已释放 stream_id=%s" % stream_id)

    @app.get("/api/realtime-stream/stop")
    async def api_realtime_stream_stop(stream_id: str = ""):
        """通知指定实时流停止，便于服务端立即释放摄像头。"""
        if stream_id:
            _realtime_stream_cancel[stream_id] = True
        return {"ok": True}

    @app.get("/api/cameras/list")
    async def api_cameras_list():
        """检测 0～4 号设备哪些可打开，便于用户确认 ivCam 等虚拟摄像头对应「外接几」。"""
        import cv2
        import sys
        available = []
        for i in range(5):
            cap = None
            try:
                if sys.platform == "win32":
                    cap = cv2.VideoCapture(int(i), cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(int(i))
                if cap is not None and cap.isOpened():
                    available.append(i)
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        return {"available": available}

    @app.get("/api/realtime-stream")
    async def api_realtime_stream(device: int = 0, stage: str = "stretch", conf: float = 0.4, stream_id: str = "", use_mediapipe_hands: bool = True, use_mediapipe_face: bool = True, model_type: str = "cpu"):
        """
        实时视频流（MJPEG）。use_mediapipe_face: 头部检测；use_mediapipe_hands: 手部外源模型。model_type: cpu|gpu 预留。
        Windows 下 DirectShow 枚举常为 0=虚拟/外接、1=本机摄像头，故对 0/1 做映射：选「默认」用设备 1，选「外接1」用设备 0。
        """
        if not stream_id:
            import uuid
            stream_id = str(uuid.uuid4())
        device_index = int(device)
        if device_index in (0, 1):
            import sys
            if sys.platform == "win32":
                device_index = 1 if device_index == 0 else 0
        return StreamingResponse(
            _realtime_stream_generator(device_index, stage, conf, stream_id, use_mediapipe_hands, use_mediapipe_face, model_type if model_type in ("cpu", "gpu") else "cpu"),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/check_processed_video/{video_name}")
    async def check_processed_video(video_name: str, stage: str = "stretch"):
        """检查处理好的视频是否存在"""
        if stage == "boiling_scooping":
            video_file = project_root / "data" / "processed_videos" / "下面及捞面" / f"{video_name}_with_skeleton.mp4"
            stage_path = "下面及捞面"
        else:
            video_file = project_root / "data" / "processed_videos" / "抻面" / f"{video_name}_with_skeleton.mp4"
            stage_path = "抻面"
        
        exists = video_file.exists()
        
        if exists:
            # 返回API路径（避免静态文件路径编码问题）
            return {
                "success": True,
                "exists": True,
                "path": f"/api/get_processed_video/{video_name}?stage={stage}",
                "path_alt": f"/data/processed_videos/{stage_path}/{video_name}_with_skeleton.mp4"
            }
        else:
            return {
                "success": False,
                "exists": False,
                "message": f"视频文件不存在: {video_file}",
                "file_path": str(video_file),
                "stage_path": stage_path
            }
    
    @app.get("/api/get_processed_video/{video_name}")
    async def get_processed_video(video_name: str, stage: str = "stretch"):
        """直接返回处理好的视频文件（避免路径编码问题）"""
        if stage == "boiling_scooping":
            video_file = project_root / "data" / "processed_videos" / "下面及捞面" / f"{video_name}_with_skeleton.mp4"
        else:
            video_file = project_root / "data" / "processed_videos" / "抻面" / f"{video_name}_with_skeleton.mp4"
        
        if video_file.exists():
            return FileResponse(
                str(video_file),
                media_type="video/mp4",
                headers={"Accept-Ranges": "bytes"}
            )
        else:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"视频文件不存在: {video_file}")
    
    # 检测API（主要功能）：支持 async_mode=1 返回 job_id，前端轮询进度与结果；model_type=cpu|gpu 预留
    @app.post("/api/detect_video")
    async def detect_video(file: UploadFile = File(...), async_mode: bool = Query(False), model_type: str = Query("cpu", description="检测使用模型：cpu=CPU训练模型，gpu=GPU训练模型（预留）")):
        import tempfile
        import time
        import os
        from pathlib import Path
        
        from src.api.video_detection_api import get_detector
        detector = get_detector(model_type=model_type if model_type in ("cpu", "gpu") else "cpu")
        if detector.model is None:
            detail = getattr(detector, "_load_error", None) or "Model not loaded"
            return {
                "success": False,
                "message": f"检测模型未加载。请先运行: python src/training/train_detection_model.py（若已训练，请查看控制台具体错误：{detail}）",
                "error": detail
            }
        
        suffix = Path(file.filename).suffix if file.filename else '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        def run_detect():
            nonlocal tmp_path
            job_id = None
            try:
                with _jobs_lock:
                    job_id = str(uuid.uuid4())
                    _detect_jobs[job_id] = {"phase": "detect", "percent": 0, "current": 0, "total": 1, "message": "准备中", "result": None, "error": None}
                def progress_cb(current, total):
                    with _jobs_lock:
                        if job_id in _detect_jobs:
                            pct = (current / total * 100) if total else 0
                            _detect_jobs[job_id].update(phase="detect", percent=round(pct, 1), current=current, total=total, message=f"检测中 {current}/{total}")
                result = detector.detect_video(tmp_path, conf_threshold=0.20, progress_callback=progress_cb)
                time.sleep(0.5)
                with _jobs_lock:
                    if job_id in _detect_jobs:
                        _detect_jobs[job_id].update(phase="done", percent=100, result=result, message="检测完成")
            except Exception as e:
                import traceback
                traceback.print_exc()
                with _jobs_lock:
                    if job_id and job_id in _detect_jobs:
                        _detect_jobs[job_id].update(phase="error", error=str(e), message="检测失败")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    def delayed_delete(path):
                        time.sleep(3)
                        try:
                            if os.path.exists(path):
                                os.unlink(path)
                        except Exception:
                            pass
                    threading.Thread(target=delayed_delete, args=(tmp_path,), daemon=True).start()
        
        if async_mode:
            threading.Thread(target=run_detect, daemon=True).start()
            # 返回刚创建的那个 job_id（run_detect 里会先写一条记录，key 在 run_detect 内生成）
            # 我们必须在 run_detect 里把 job_id 传出来，否则无法返回。所以改为：先生成 job_id，再写入初始状态，再启动线程传入 job_id
            job_id = str(uuid.uuid4())
            with _jobs_lock:
                for j in _detect_jobs.values():
                    j["cancelled"] = True
                _detect_jobs[job_id] = {"phase": "detect", "percent": 0, "current": 0, "total": 1, "message": "准备中", "result": None, "error": None, "cancelled": False}
            def run_with_jid():
                jid = job_id
                try:
                    def progress_cb(current, total):
                        with _jobs_lock:
                            if _detect_jobs.get(jid, {}).get("cancelled"):
                                raise RuntimeError("检测已取消")
                            if jid in _detect_jobs:
                                pct = (current / total * 100) if total else 0
                                _detect_jobs[jid].update(phase="detect", percent=round(pct, 1), current=current, total=total, message=f"检测中 {current}/{total}")
                    result = detector.detect_video(tmp_path, conf_threshold=0.20, progress_callback=progress_cb)
                    time.sleep(0.5)
                    with _jobs_lock:
                        if jid in _detect_jobs:
                            _detect_jobs[jid].update(phase="done", percent=100, result=result, message="检测完成")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    with _jobs_lock:
                        if jid in _detect_jobs:
                            _detect_jobs[jid].update(phase="error", error=str(e), message="检测失败")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        def delayed_delete(path):
                            time.sleep(3)
                            try:
                                if os.path.exists(path):
                                    os.unlink(path)
                            except Exception:
                                pass
                        threading.Thread(target=delayed_delete, args=(tmp_path,), daemon=True).start()
            threading.Thread(target=run_with_jid, daemon=True).start()
            return JSONResponse(status_code=202, content={"success": True, "job_id": job_id, "message": "检测已开始，请轮询进度"})
        
        # 同步模式
        try:
            def progress_cb(current, total):
                pass
            result = detector.detect_video(tmp_path, conf_threshold=0.20, progress_callback=progress_cb)
            time.sleep(0.5)
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "message": f"检测失败: {e}"}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                def delayed_delete(path):
                    time.sleep(3)
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                    except Exception:
                        pass
                threading.Thread(target=delayed_delete, args=(tmp_path,), daemon=True).start()
    
    @app.post("/api/detect_cancel")
    async def cancel_detect(job_id: str = Query(..., description="要取消的检测任务 job_id")):
        """前端点击「停止检测」时调用，将所有检测任务标记为已取消，确保所有检测线程都会退出。"""
        with _jobs_lock:
            for j in _detect_jobs.values():
                j["cancelled"] = True
        return {"success": True, "message": "已发送取消请求，检测将尽快停止"}
    
    @app.get("/api/detect_progress")
    async def get_detect_progress(job_id: str = Query(...)):
        with _jobs_lock:
            job = _detect_jobs.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "job not found"})
        return {
            "job_id": job_id,
            "phase": job.get("phase"),
            "percent": job.get("percent", 0),
            "current": job.get("current", 0),
            "total": job.get("total", 1),
            "message": job.get("message", ""),
            "error": job.get("error"),
        }
    
    @app.get("/api/detect_result")
    async def get_detect_result(job_id: str = Query(...)):
        with _jobs_lock:
            job = _detect_jobs.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "job not found"})
        if job.get("phase") == "error":
            return JSONResponse(content={"success": False, "error": job.get("error", "unknown")})
        if job.get("phase") != "done":
            return JSONResponse(status_code=202, content={"message": "任务未完成", "phase": job.get("phase")})
        result = job.get("result")
        if result is None:
            return JSONResponse(status_code=404, content={"error": "no result"})
        return result
    
    # 下面及捞面检测API（同样支持 async_mode=1）；model_type=cpu|gpu 预留
    @app.post("/api/detect_boiling_scooping")
    async def detect_boiling_scooping(file: UploadFile = File(...), async_mode: bool = Query(False), model_type: str = Query("cpu", description="检测使用模型：cpu=CPU训练模型，gpu=GPU训练模型（预留）")):
        import tempfile
        import time
        import os
        from pathlib import Path
        
        from src.api.video_detection_api import get_boiling_scooping_detector
        detector = get_boiling_scooping_detector(model_type=model_type if model_type in ("cpu", "gpu") else "cpu")
        if detector.model is None:
            detail = getattr(detector, "_load_error", None) or "Model not loaded"
            return {
                "success": False,
                "message": f"检测模型未加载。请先运行: python src/training/train_boiling_scooping_model.py（若已训练，请查看控制台：{detail}）",
                "error": detail
            }
        suffix = Path(file.filename).suffix if file.filename else '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        if async_mode:
            job_id = str(uuid.uuid4())
            with _jobs_lock:
                for j in _detect_jobs.values():
                    j["cancelled"] = True
                _detect_jobs[job_id] = {"phase": "detect", "percent": 0, "current": 0, "total": 1, "message": "准备中", "result": None, "error": None, "cancelled": False}
            def run_boiling():
                jid = job_id
                try:
                    def progress_cb(current, total):
                        with _jobs_lock:
                            if _detect_jobs.get(jid, {}).get("cancelled"):
                                raise RuntimeError("检测已取消")
                            if jid in _detect_jobs:
                                pct = (current / total * 100) if total else 0
                                _detect_jobs[jid].update(phase="detect", percent=round(pct, 1), current=current, total=total, message=f"检测中 {current}/{total}")
                    result = detector.detect_video(tmp_path, conf_threshold=0.20, progress_callback=progress_cb)
                    time.sleep(0.5)
                    with _jobs_lock:
                        if jid in _detect_jobs:
                            _detect_jobs[jid].update(phase="done", percent=100, result=result, message="检测完成")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    with _jobs_lock:
                        if jid in _detect_jobs:
                            _detect_jobs[jid].update(phase="error", error=str(e), message="检测失败")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        def delayed_delete(path):
                            time.sleep(3)
                            try:
                                if os.path.exists(path):
                                    os.unlink(path)
                            except Exception:
                                pass
                        threading.Thread(target=delayed_delete, args=(tmp_path,), daemon=True).start()
            threading.Thread(target=run_boiling, daemon=True).start()
            return JSONResponse(status_code=202, content={"success": True, "job_id": job_id, "message": "检测已开始，请轮询进度"})
        
        try:
            result = detector.detect_video(tmp_path, conf_threshold=0.20, progress_callback=lambda a, b: None)
            time.sleep(0.5)
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "message": f"检测失败: {e}"}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                def delayed_delete(path):
                    time.sleep(3)
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                    except Exception:
                        pass
                threading.Thread(target=delayed_delete, args=(tmp_path,), daemon=True).start()
    
    # 评分API（本地视频上传 → 评分JSON）；支持 async_mode=1 返回 job_id，前端轮询进度后跳转综合评分页
    @app.post("/api/score_video")
    async def score_video(file: UploadFile = File(...), stage: str = "stretch", async_mode: bool = Query(False), model_type: str = Query("cpu", description="cpu=CPU训练模型，gpu=GPU训练模型（预留）")):
        """
        评分骨架：
        - 仅支持本地视频上传
        - 默认使用抻面模型（stage="stretch"），若传 stage="boiling_scooping" 则用下面及捞面模型
        - 评分逻辑为占位实现：基于检测结果统计占比，后续可替换为正式规则/模型
        - async_mode=1 时返回 job_id，需轮询 /api/score_progress 与 /api/score_result，完成后可跳转综合评分页并请求 /api/upload_score_result
        """
        import tempfile
        import time
        import os
        from pathlib import Path

        if model_type not in ("cpu", "gpu"):
            model_type = "cpu"

        if stage == "boiling_scooping":
            from src.api.video_detection_api import get_boiling_scooping_detector
            detector = get_boiling_scooping_detector(model_type=model_type)
            # 与 datasets/boiling_scooping_detection/data.yaml 一致：0 noodle_rope, 1 hand, 2 tools_noodle, 3 soup_noodle(汤中面条)
            classes = ["noodle_rope", "hand", "tools_noodle", "soup_noodle"]
        else:
            from src.api.video_detection_api import get_detector
            detector = get_detector(model_type=model_type)
            classes = ["hand", "noodle_rope", "noodle_bundle"]

        if detector.model is None:
            detail = getattr(detector, "_load_error", None) or "Model not loaded"
            return {
                "success": False,
                "message": f"检测模型未加载，请先运行对应训练脚本（详情：{detail}）",
                "error": detail
            }

        suffix = Path(file.filename).suffix if file.filename else '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        def _validate_stretch_content(detections, total_frames, classes):
            """高置信度面条目标过少则视为非抻面视频，拒绝评分，避免手部/背景误检导致无意义高分。"""
            if total_frames == 0:
                return False, "无有效帧"
            noodle_names = {"noodle_rope", "noodle_bundle"}
            min_conf = 0.55
            frames_with_noodle = 0
            for frame_data in detections:
                for det in frame_data.get("detections", []):
                    cls = det.get("class")
                    if isinstance(cls, (int, float)) and 0 <= int(cls) < len(classes):
                        cls = classes[int(cls)]
                    if cls in noodle_names and (det.get("conf") or 0) >= min_conf:
                        frames_with_noodle += 1
                        break
            if frames_with_noodle / total_frames < 0.12:
                return False, "未检测到有效的抻面内容（手部与面条），可能为非抻面视频或背景误检较多，无法给出可靠评分。请上传包含手部与面条的抻面视频。"
            return True, ""

        def _validate_boiling_scooping_content(detections, total_frames, classes):
            """下面及捞面：面条/工具/汤面过少则视为非相关视频，拒绝评分。置信度不宜过高以免正常视频被误拒。"""
            if total_frames == 0:
                return False, "无有效帧"
            relevant_names = {"noodle_rope", "tools_noodle", "soup_noodle"}
            min_conf = 0.50   # 0.5 以上即计入，兼顾正常视频中部分 0.5～0.6 的检测
            min_frame_ratio = 0.15  # 至少 15% 帧含有效目标即可视为相关视频
            frames_with_relevant = 0
            for frame_data in detections:
                for det in frame_data.get("detections", []):
                    cls = det.get("class")
                    if isinstance(cls, (int, float)) and 0 <= int(cls) < len(classes):
                        cls = classes[int(cls)]
                    if cls in relevant_names and (det.get("conf") or 0) >= min_conf:
                        frames_with_relevant += 1
                        break
            if frames_with_relevant / total_frames < min_frame_ratio:
                return False, "未检测到有效的下面及捞面内容（面条/工具/汤面等），可能为非相关视频或误检较多，无法给出可靠评分。请上传包含下面或捞面操作的视频。"
            return True, ""

        if async_mode:
            job_id = str(uuid.uuid4())
            with _jobs_lock:
                _score_jobs[job_id] = {"phase": "detect", "percent": 0, "current": 0, "total": 1, "message": "准备中", "result": None, "error": None}

            def run_score_job():
                jid = job_id
                try:
                    def progress_cb(current, total):
                        with _jobs_lock:
                            if jid in _score_jobs:
                                pct = (current / total * 100) if total else 0
                                _score_jobs[jid].update(phase="detect", percent=round(pct, 1), current=current, total=total, message=f"检测中 {current}/{total}")
                    result = detector.detect_video(tmp_path, conf_threshold=0.20, progress_callback=progress_cb)
                    if not result or not result.get("success", True):
                        with _jobs_lock:
                            if jid in _score_jobs:
                                _score_jobs[jid].update(phase="error", error="检测失败", message="检测失败，无法生成评分")
                        return
                    detections = result.get("detections", [])
                    total_frames = result.get("total_frames", len(detections))
                    if total_frames == 0:
                        with _jobs_lock:
                            if jid in _score_jobs:
                                _score_jobs[jid].update(phase="error", error="无有效帧", message="视频无有效帧，无法评分")
                        return
                    if stage == "stretch":
                        valid, msg = _validate_stretch_content(detections, total_frames, classes)
                        if not valid:
                            with _jobs_lock:
                                if jid in _score_jobs:
                                    _score_jobs[jid].update(phase="error", error=msg, message=msg)
                            return
                    if stage == "boiling_scooping":
                        valid, msg = _validate_boiling_scooping_content(detections, total_frames, classes)
                        if not valid:
                            with _jobs_lock:
                                if jid in _score_jobs:
                                    _score_jobs[jid].update(phase="error", error=msg, message=msg)
                            return
                    with _jobs_lock:
                        if jid in _score_jobs:
                            _score_jobs[jid].update(phase="score", percent=90, message="评分中…")
                    out = None
                    if stage == "stretch":
                        try:
                            from src.scoring.stretch_scorer import StretchScorer
                            scorer = StretchScorer()
                            video_detections = []
                            for frame_data in detections:
                                frame_detections = []
                                for det in frame_data.get("detections", []):
                                    cls_name = det.get("class")
                                    if isinstance(cls_name, (int, float)) and 0 <= int(cls_name) < len(classes):
                                        cls_name = classes[int(cls_name)]
                                    frame_detections.append({
                                        'class': cls_name, 'conf': det.get('conf', 0.5),
                                        'xyxy': det.get('xyxy', [0, 0, 0, 0]), 'width': det.get('width', 0), 'height': det.get('height', 0)
                                    })
                                video_detections.append({'frame_index': frame_data.get('frame_index', 0), 'detections': frame_detections})
                            video_score_result = scorer.score_video(video_detections, video_path=str(tmp_path))
                            model_path = getattr(detector, "model_path", None) or ""
                            out = {
                                "success": True, "stage": stage, "total_frames": total_frames,
                                "scored_frames": video_score_result.get('scored_frames', 0),
                                "average_overall_score": round(video_score_result.get('average_overall_score', 0), 2),
                                "class_average_scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                                "scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                                "total_score": round(video_score_result.get('average_overall_score', 0), 2),
                                "details": video_score_result.get('class_average_scores', {}),
                                "rules_used": "基于标准数据集的评分规则和阈值",
                                "frame_scores_sample": video_score_result.get('frame_scores', [])[:5],
                                "model_source": model_path or "当前最佳抻面模型(latest best.pt)",
                                "score_basis": "最佳抻面模型检测 + 规则/图像特征评分"
                            }
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                    if stage == "boiling_scooping" and out is None:
                        try:
                            from src.scoring.boiling_scooping_scorer import BoilingScoopingScorer
                            scorer = BoilingScoopingScorer()
                            video_detections = []
                            for frame_data in detections:
                                frame_detections = []
                                for det in frame_data.get("detections", []):
                                    cls_name = det.get("class")
                                    if isinstance(cls_name, (int, float)) and 0 <= int(cls_name) < len(classes):
                                        cls_name = classes[int(cls_name)]
                                    frame_detections.append({
                                        'class': cls_name, 'conf': det.get('conf', 0.5),
                                        'xyxy': det.get('xyxy', [0, 0, 0, 0]), 'width': det.get('width', 0), 'height': det.get('height', 0)
                                    })
                                video_detections.append({'frame_index': frame_data.get('frame_index', 0), 'detections': frame_detections})
                            video_score_result = scorer.score_video(video_detections, video_path=str(tmp_path))
                            out = {
                                "success": True, "stage": stage, "total_frames": total_frames,
                                "scored_frames": video_score_result.get('scored_frames', 0),
                                "average_overall_score": round(video_score_result.get('average_overall_score', 0), 2),
                                "class_average_scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                                "scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                                "total_score": round(video_score_result.get('average_overall_score', 0), 2),
                                "details": video_score_result.get('class_average_scores', {}),
                                "rules_used": "下面及捞面规则（scoring_rules.json）",
                                "frame_scores_sample": video_score_result.get('frame_scores', [])[:5],
                                "model_source": getattr(detector, "model_path", None) or "",
                                "score_basis": "检测 + 图像特征 + 规则（与抻面一致）"
                            }
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                    if out is None:
                        class_frame_presence = {c: 0 for c in classes}
                        class_object_counts = {c: 0 for c in classes}
                        for frame in detections:
                            present_in_frame = set()
                            for det in frame.get("detections", []):
                                cls_name = det.get("class")
                                if isinstance(cls_name, (int, float)) and 0 <= int(cls_name) < len(classes):
                                    cls_name = classes[int(cls_name)]
                                if cls_name in class_object_counts:
                                    class_object_counts[cls_name] += 1
                                    present_in_frame.add(cls_name)
                            for cls in present_in_frame:
                                class_frame_presence[cls] += 1
                        scores = {}
                        details = {}
                        for cls in classes:
                            coverage = class_frame_presence[cls] / total_frames if total_frames else 0
                            density = class_object_counts[cls] / max(total_frames, 1)
                            score = round(min(1.0, coverage * 0.7 + min(density, 1.0) * 0.3), 3)
                            scores[cls] = score
                            details[cls] = {"frame_coverage": round(coverage, 3), "avg_objects_per_frame": round(density, 3),
                                            "frames_with_class": class_frame_presence[cls], "total_objects": class_object_counts[cls]}
                        total_score_raw = round(sum(scores.values()) / len(scores), 3) if scores else 0.0
                        # 下面及捞面：占位规则为 0~1，统一换算为与抻面一致的 5 分制（1 + raw*4）
                        if stage == "boiling_scooping":
                            total_score = round(1.0 + total_score_raw * 4, 2)
                            for cls in list(details.keys()):
                                if not cls.startswith("_"):
                                    details[cls]["raw_score_0_1"] = scores.get(cls, 0)
                            details["_raw_total_0_1"] = total_score_raw
                            scores = {k: round(1.0 + v * 4, 2) for k, v in scores.items()}
                            rules_used = "占位规则（覆盖率+密度线性加权），已换算为 5 分制；正式评分建议后续接入专用规则或模型"
                        else:
                            total_score = total_score_raw
                            rules_used = "占位规则：覆盖率+目标密度线性加权"
                        out = {"success": True, "stage": stage, "total_frames": total_frames, "scores": scores,
                               "total_score": total_score, "details": details, "rules_used": rules_used}
                    global _upload_score_result, _upload_score_stage
                    _upload_score_result = out
                    _upload_score_stage = stage
                    with _jobs_lock:
                        if jid in _score_jobs:
                            _score_jobs[jid].update(phase="done", percent=100, result=out, message="评分完成")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    with _jobs_lock:
                        if jid in _score_jobs:
                            _score_jobs[jid].update(phase="error", error=str(e), message="评分失败")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        def delayed_delete(path):
                            time.sleep(3)
                            try:
                                if os.path.exists(path):
                                    os.unlink(path)
                            except Exception:
                                pass
                        threading.Thread(target=delayed_delete, args=(tmp_path,), daemon=True).start()
            threading.Thread(target=run_score_job, daemon=True).start()
            return JSONResponse(status_code=202, content={"success": True, "job_id": job_id, "message": "评分已开始，请轮询进度"})

        try:
            # 同步：复用检测流程
            result = detector.detect_video(tmp_path, conf_threshold=0.20)
            if not result or not result.get("success", True):
                return {
                    "success": False,
                    "message": "检测失败，无法生成评分",
                    "error": result.get("error") if isinstance(result, dict) else "detect failed"
                }

            detections = result.get("detections", [])
            total_frames = result.get("total_frames", len(detections))
            if total_frames == 0:
                return {
                    "success": False,
                    "message": "视频无有效帧，无法评分"
                }

            # 内容有效性校验：避免非相关视频（如仅手部晃动、背景误检）被当成正常评分
            if stage == "stretch":
                valid, msg = _validate_stretch_content(detections, total_frames, classes)
                if not valid:
                    return {"success": False, "message": msg}
            if stage == "boiling_scooping":
                valid, msg = _validate_boiling_scooping_content(detections, total_frames, classes)
                if not valid:
                    return {"success": False, "message": msg}

            # 使用评分规则进行自动评分（仅抻面阶段）
            if stage == "stretch":
                try:
                    from src.scoring.stretch_scorer import StretchScorer
                    scorer = StretchScorer()
                    
                    # 转换检测结果格式
                    video_detections = []
                    for frame_data in detections:
                        frame_detections = []
                        for det in frame_data.get("detections", []):
                            cls_name = det.get("class")
                            # 兼容数字类别
                            if isinstance(cls_name, (int, float)) and 0 <= int(cls_name) < len(classes):
                                cls_name = classes[int(cls_name)]
                            
                            frame_detections.append({
                                'class': cls_name,
                                'conf': det.get('conf', 0.5),
                                'xyxy': det.get('xyxy', [0, 0, 0, 0]),
                                'width': det.get('width', 0),
                                'height': det.get('height', 0)
                            })
                        
                        video_detections.append({
                            'frame_index': frame_data.get('frame_index', 0),
                            'detections': frame_detections
                        })
                    
                    # 进行评分（传入视频路径以提取图像特征）
                    video_score_result = scorer.score_video(video_detections, video_path=str(tmp_path))
                    model_path = getattr(detector, "model_path", None) or ""
                    return {
                        "success": True,
                        "stage": stage,
                        "total_frames": total_frames,
                        "scored_frames": video_score_result.get('scored_frames', 0),
                        "average_overall_score": round(video_score_result.get('average_overall_score', 0), 2),
                        "class_average_scores": {
                            k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()
                        },
                        "scores": {
                            k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()
                        },
                        "total_score": round(video_score_result.get('average_overall_score', 0), 2),
                        "details": video_score_result.get('class_average_scores', {}),
                        "rules_used": "基于标准数据集的评分规则和阈值",
                        "frame_scores_sample": video_score_result.get('frame_scores', [])[:5],
                        "model_source": model_path or "当前最佳抻面模型(latest best.pt)",
                        "score_basis": "最佳抻面模型检测 + 规则/图像特征评分"
                    }
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    error_msg = str(e)
                    print(f"[警告] 评分规则加载失败，使用占位逻辑: {error_msg}")

            if stage == "boiling_scooping":
                try:
                    from src.scoring.boiling_scooping_scorer import BoilingScoopingScorer
                    scorer = BoilingScoopingScorer()
                    video_detections = []
                    for frame_data in detections:
                        frame_detections = []
                        for det in frame_data.get("detections", []):
                            cls_name = det.get("class")
                            if isinstance(cls_name, (int, float)) and 0 <= int(cls_name) < len(classes):
                                cls_name = classes[int(cls_name)]
                            frame_detections.append({
                                'class': cls_name, 'conf': det.get('conf', 0.5),
                                'xyxy': det.get('xyxy', [0, 0, 0, 0]), 'width': det.get('width', 0), 'height': det.get('height', 0)
                            })
                        video_detections.append({'frame_index': frame_data.get('frame_index', 0), 'detections': frame_detections})
                    video_score_result = scorer.score_video(video_detections, video_path=str(tmp_path))
                    return {
                        "success": True, "stage": stage, "total_frames": total_frames,
                        "scored_frames": video_score_result.get('scored_frames', 0),
                        "average_overall_score": round(video_score_result.get('average_overall_score', 0), 2),
                        "class_average_scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                        "scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                        "total_score": round(video_score_result.get('average_overall_score', 0), 2),
                        "details": video_score_result.get('class_average_scores', {}),
                        "rules_used": "下面及捞面规则（scoring_rules.json）",
                        "frame_scores_sample": video_score_result.get('frame_scores', [])[:5],
                        "model_source": getattr(detector, "model_path", None) or "",
                        "score_basis": "检测 + 图像特征 + 规则（与抻面一致）"
                    }
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[警告] 下面及捞面评分器失败，使用占位逻辑: {e}")

            # 占位评分逻辑（用于下面及捞面阶段或评分规则加载失败时）
            class_frame_presence = {c: 0 for c in classes}
            class_object_counts = {c: 0 for c in classes}

            for frame in detections:
                present_in_frame = set()
                for det in frame.get("detections", []):
                    cls_name = det.get("class")
                    if isinstance(cls_name, (int, float)) and 0 <= int(cls_name) < len(classes):
                        cls_name = classes[int(cls_name)]
                    if cls_name in class_object_counts:
                        class_object_counts[cls_name] += 1
                        present_in_frame.add(cls_name)
                for cls in present_in_frame:
                    class_frame_presence[cls] += 1

            scores = {}
            details = {}
            for cls in classes:
                coverage = class_frame_presence[cls] / total_frames if total_frames else 0
                density = class_object_counts[cls] / max(total_frames, 1)
                score = round(min(1.0, coverage * 0.7 + min(density, 1.0) * 0.3), 3)
                scores[cls] = score
                details[cls] = {
                    "frame_coverage": round(coverage, 3),
                    "avg_objects_per_frame": round(density, 3),
                    "frames_with_class": class_frame_presence[cls],
                    "total_objects": class_object_counts[cls]
                }

            total_score_raw = round(sum(scores.values()) / len(scores), 3) if scores else 0.0
            # 下面及捞面：占位规则为 0~1，统一换算为与抻面一致的 5 分制
            if stage == "boiling_scooping":
                total_score = round(1.0 + total_score_raw * 4, 2)
                for cls in list(details.keys()):
                    if not cls.startswith("_"):
                        details[cls]["raw_score_0_1"] = scores.get(cls, 0)
                details["_raw_total_0_1"] = total_score_raw
                scores = {k: round(1.0 + v * 4, 2) for k, v in scores.items()}
                rules_used = "占位规则（覆盖率+密度线性加权），已换算为 5 分制；正式评分建议后续接入专用规则或模型"
            else:
                total_score = total_score_raw
                rules_used = "占位规则：覆盖率+目标密度线性加权"

            return {
                "success": True,
                "stage": stage,
                "total_frames": total_frames,
                "scores": scores,
                "total_score": total_score,
                "details": details,
                "rules_used": rules_used
            }
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "message": f"评分失败: {error_msg}"
            }
        finally:
            # 延迟删除临时文件（使用模块级 threading，勿在函数内 import 以免遮蔽导致 async 分支报错）
            if tmp_path and os.path.exists(tmp_path):
                def delayed_delete(path):
                    time.sleep(3)  # 等待3秒确保文件完全释放
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                            print(f"[OK] 已删除临时文件: {path}")
                    except Exception as e:
                        print(f"[WARN] 删除临时文件失败: {e}")

                thread = threading.Thread(target=delayed_delete, args=(tmp_path,))
                thread.daemon = True
                thread.start()

    @app.get("/api/score_progress")
    async def get_score_progress(job_id: str = Query(...)):
        with _jobs_lock:
            job = _score_jobs.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "job not found"})
        return {
            "job_id": job_id,
            "phase": job.get("phase"),
            "percent": job.get("percent", 0),
            "current": job.get("current", 0),
            "total": job.get("total", 1),
            "message": job.get("message", "")
        }

    @app.get("/api/score_result")
    async def get_score_result(job_id: str = Query(...)):
        with _jobs_lock:
            job = _score_jobs.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "job not found"})
        if job.get("phase") == "error":
            return JSONResponse(content={"success": False, "error": job.get("error", "unknown")})
        if job.get("phase") != "done":
            return JSONResponse(status_code=202, content={"message": "任务未完成", "phase": job.get("phase")})
        result = job.get("result")
        if result is None:
            return JSONResponse(status_code=404, content={"error": "no result"})
        return result

    @app.get("/api/upload_score_result")
    async def get_upload_score_result():
        """综合评分页 source=upload 时拉取本次上传视频的评分结果（与 stage 一致）"""
        global _upload_score_result, _upload_score_stage
        if _upload_score_result is None:
            return JSONResponse(status_code=404, content={"error": "暂无上传评分结果"})
        return {"success": True, "stage": _upload_score_stage, "result": _upload_score_result}

    # ========== MediaPipe单例（优化性能） ==========
    _mediapipe_landmarker = None
    _mediapipe_lock = threading.Lock()
    
    def get_mediapipe_landmarker():
        """获取MediaPipe HandLandmarker单例"""
        global _mediapipe_landmarker
        if _mediapipe_landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python import BaseOptions
            
            model_dir = project_root / "weights" / "mediapipe"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "hand_landmarker.task"
            
            if not model_path.exists():
                return None
            
            try:
                base_options = BaseOptions(model_asset_path=str(model_path))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2,
                    running_mode=vision.RunningMode.IMAGE,
                )
                _mediapipe_landmarker = vision.HandLandmarker.create_from_options(options)
                print("[OK] MediaPipe HandLandmarker单例已创建")
            except Exception as e:
                print(f"[错误] MediaPipe初始化失败: {e}")
                return None
        
        return _mediapipe_landmarker
    
    # 检测+姿态估计API（可选功能，预留接口）
    @app.post("/api/detect_with_pose")
    async def detect_with_pose(file: UploadFile = File(...)):
        """
        检测+姿态估计API（可选功能）
        在检测基础上添加姿态估计，两者独立运行
        """
        # TODO: 实现检测+姿态估计的联合API
        # 当前阶段主要关注检测功能，此接口为预留
        return {
            "success": False,
            "message": "此功能待实现，当前主要关注检测功能"
        }
    
    # ========== 评分标注工具API ==========
    
    @app.get("/scoring-tool")
    async def scoring_tool():
        """抻面评分标注工具页面"""
        web_file = web_dir / "stretch_scoring_tool.html"
        if web_file.exists():
            return FileResponse(str(web_file))
        return {"message": "评分标注工具未找到"}

    @app.get("/scoring-tool-boiling")
    async def scoring_tool_boiling():
        """下面及捞面评分标注工具页面（xl 标注视频，noodle_rope/hand/tools_noodle/soup_noodle）"""
        web_file = web_dir / "boiling_scoring_tool.html"
        if web_file.exists():
            return FileResponse(str(web_file))
        return {"message": "下面及捞面评分标注工具未找到"}

    @app.get("/scoring-visualization")
    async def scoring_visualization():
        """评分结果可视化页面（禁止缓存，始终加载最新）"""
        web_file = web_dir / "scoring_visualization.html"
        if web_file.exists():
            r = FileResponse(str(web_file))
            r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            r.headers["Pragma"] = "no-cache"
            r.headers["Expires"] = "0"
            return r
        return {"message": "评分可视化页面未找到"}

    @app.get("/scoring-visualization-boiling")
    async def scoring_visualization_boiling():
        """下面及捞面综合评分可视化页面"""
        web_file = web_dir / "scoring_visualization_boiling.html"
        if web_file.exists():
            r = FileResponse(str(web_file))
            r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            r.headers["Pragma"] = "no-cache"
            r.headers["Expires"] = "0"
            return r
        return {"message": "下面及捞面评分可视化页面未找到"}

    @app.get("/product-scoring-visualization")
    async def product_scoring_visualization():
        """兰州拉面成品综合评分可视化页面"""
        web_file = web_dir / "product_scoring_visualization.html"
        if web_file.exists():
            r = FileResponse(str(web_file))
            r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            r.headers["Pragma"] = "no-cache"
            r.headers["Expires"] = "0"
            return r
        return {"message": "成品综合评分可视化页面未找到"}

    # ========== 兰州拉面成品评估标注 ==========
    @app.get("/product-annotation")
    async def product_annotation_page():
        """兰州拉面成品分析评估 - 标注工具页面"""
        web_file = web_dir / "product_annotation.html"
        if web_file.exists():
            return FileResponse(str(web_file))
        return {"message": "成品标注工具页面未找到"}

    @app.get("/api/product-images/list")
    async def api_product_images_list():
        """拉面成品图片列表（仅 lmcp.*.jpg），按文件名中的数字编号从小到大排序"""
        if not raw_lmcp_dir.exists():
            return {"success": False, "images": [], "message": "拉面成品目录不存在"}
        import re
        pat = re.compile(r"^lmcp\.(\d+)\.(jpg|jpeg|png)$", re.I)
        def num_sort_key(name):
            m = pat.match(name)
            return (int(m.group(1)), name) if m else (999999, name)
        files = sorted(
            [f.name for f in raw_lmcp_dir.iterdir() if f.is_file() and pat.match(f.name)],
            key=num_sort_key
        )
        return {"success": True, "images": files}

    @app.get("/api/product-image/{filename}")
    async def api_product_image(filename: str):
        """按文件名返回拉面成品图片（避免中文路径在 URL 中的问题）"""
        if ".." in filename or "/" in filename or "\\" in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = raw_lmcp_dir / filename
        if not path.exists() or not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(path), media_type="image/jpeg")

    @app.get("/api/product-annotations")
    async def api_product_annotations_get():
        """读取拉面成品标注汇总"""
        ann_file = product_scores_dir / "annotations.json"
        if not ann_file.exists():
            return {"success": True, "items": [], "updated": None}
        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"success": True, "items": data.get("items", []), "updated": data.get("updated")}
        except Exception as e:
            return {"success": False, "items": [], "message": str(e)}

    @app.post("/api/product-annotations")
    async def api_product_annotations_save(request: Request):
        """保存拉面成品标注汇总。body: { \"items\": [ {...}, ... ], \"updated\": \"optional\" }"""
        product_scores_dir.mkdir(parents=True, exist_ok=True)
        ann_file = product_scores_dir / "annotations.json"
        try:
            body = await request.json()
            items = body.get("items", [])
            from datetime import datetime
            data = {
                "items": items,
                "updated": body.get("updated") or datetime.now().isoformat(),
                "_schema": "image, noodle_quality, soup_type, noodle_chili, beef, egg, scallion, notes"
            }
            with open(ann_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return {"success": True, "message": "已保存", "count": len(items)}
        except Exception as e:
            return {"success": False, "message": str(e)}

    # ========== 拉面成品评分 API（以面条质感为主体，汤型/辣椒不惩罚）==========
    @app.post("/api/product-score")
    async def api_product_score(request: Request):
        """成品得分。body: { \"annotation\": {...} } 或 { \"items\": [...] }。使用规则公式 S_texture + w*S_presentation。"""
        try:
            from src.scoring.product_scorer import ProductScorer
            body = await request.json()
            scorer = ProductScorer()
            if "annotation" in body:
                out = scorer.score_from_annotation(body["annotation"], use_presentation=body.get("use_presentation", True))
                return {"success": True, "score": out}
            if "items" in body:
                results = scorer.batch_score_from_annotations(body["items"], use_presentation=body.get("use_presentation", True))
                return {"success": True, "scores": results, "count": len(results)}
            return {"success": False, "message": "请提供 annotation 或 items"}
        except FileNotFoundError as e:
            return {"success": False, "message": str(e)}
        except Exception as e:
            return {"success": False, "message": str(e)}

    @app.get("/api/product-score/batch")
    async def api_product_score_batch():
        """对当前 annotations.json 中全部标注批量计算成品得分。"""
        try:
            from src.scoring.product_scorer import ProductScorer, load_annotations
            items = load_annotations()
            if not items:
                return {"success": True, "scores": [], "count": 0, "message": "无标注数据"}
            scorer = ProductScorer()
            results = scorer.batch_score_from_annotations(items, use_presentation=True)
            return {"success": True, "scores": results, "count": len(results)}
        except Exception as e:
            return {"success": False, "message": str(e), "scores": []}

    @app.get("/api/product-score/image/{filename}")
    async def api_product_score_image(filename: str):
        """用训练好的成品质感模型对单张图片预测并打分（需先运行 scripts 训练）。"""
        if ".." in filename or "/" in filename or "\\" in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = raw_lmcp_dir / filename
        if not path.exists() or not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        try:
            from src.scoring.product_predictor import ProductPredictor
            predictor = ProductPredictor()
            result = predictor.predict_and_score(path, presentation_bonus=0.0)
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "message": str(e)}

    @app.post("/api/product-score/upload")
    async def api_product_score_upload(file: UploadFile = File(...)):
        """上传成品图片进行评分。若有训练好的模型则用模型预测质感并打分，否则尝试从数据集匹配或返回提示。"""
        import tempfile
        import shutil
        suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
        if suffix.lower() not in (".jpg", ".jpeg", ".png"):
            suffix = ".jpg"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = Path(tmp.name)
            try:
                from src.scoring.product_predictor import ProductPredictor
                predictor = ProductPredictor()
                if predictor.is_loaded:
                    result = predictor.predict_and_score(tmp_path, presentation_bonus=0.0)
                    if result.get("error") == "not_ramen":
                        return {"success": False, "message": result.get("message", "无法识别为拉面成品，请上传成品图片。")}
                    return {"success": True, "image_filename": file.filename, **result}
                from src.scoring.product_scorer import ProductScorer
                scorer = ProductScorer()
                result = scorer.score_from_prediction("fair", 0.0)
                out = {"success": True, "image_filename": file.filename, "message": "模型未加载，返回默认规则分", **result}
                out.setdefault("soup_type", "")
                out.setdefault("noodle_chili", "")
                out.setdefault("beef", ""); out.setdefault("egg", ""); out.setdefault("scallion", "")
                return out
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        except Exception as e:
            return {"success": False, "message": str(e)}

    @app.get("/api/product-score/representative-samples")
    async def api_product_representative_samples():
        """从标注数据集中挑选约 10 个具有代表性且得分优秀的样本（红汤多肉有蛋、清汤多蒜苗等）及评分细节。"""
        try:
            from src.scoring.product_scorer import ProductScorer, load_annotations
            items = load_annotations()
            if not items:
                return {"success": True, "samples": []}
            scorer = ProductScorer()
            scored = []
            for it in items:
                s = scorer.score_from_annotation(it, use_presentation=True)
                scored.append({"annotation": it, "score": s})
            scored.sort(key=lambda x: x["score"]["total_score"], reverse=True)
            def label_zh(a):
                parts = []
                if a.get("soup_type") == "red":
                    parts.append("红汤")
                else:
                    parts.append("清汤")
                if a.get("beef") == "more":
                    parts.append("多肉")
                elif a.get("beef") == "little":
                    parts.append("少量肉")
                if a.get("egg") and str(a.get("egg")).lower() in ("yes", "little", "normal", "more"):
                    parts.append("有蛋")
                if a.get("scallion") == "more":
                    parts.append("多蒜苗")
                elif a.get("scallion") == "little":
                    parts.append("少蒜苗")
                return " · ".join(parts) if parts else "成品"
            samples = []
            seen_labels = set()
            for x in scored[:30]:
                lab = label_zh(x["annotation"])
                if lab not in seen_labels or len(samples) < 10:
                    seen_labels.add(lab)
                    samples.append({
                        "image": x["annotation"].get("image"),
                        "label_zh": lab,
                        "annotation": x["annotation"],
                        "score": x["score"],
                    })
                    if len(samples) >= 10:
                        break
            if len(samples) < 10:
                for x in scored:
                    if len(samples) >= 10:
                        break
                    if not any(s["image"] == x["annotation"].get("image") for s in samples):
                        samples.append({
                            "image": x["annotation"].get("image"),
                            "label_zh": label_zh(x["annotation"]),
                            "annotation": x["annotation"],
                            "score": x["score"],
                        })
            return {"success": True, "samples": samples[:10]}
        except Exception as e:
            return {"success": False, "message": str(e), "samples": []}

    # ========== AI分析API ==========
    # 配置与集成说明见 docs/AI接口配置与集成说明.md（一次配置持久可用，其他界面可复用 /api/ai-analyze-scoring 与 /api/ai-chat）
    
    @app.get("/api/ai-config-status")
    async def ai_config_status():
        """查看 AI 配置是否已加载（排查用，不返回密钥内容）"""
        return {
            "api_key_loaded": bool(_ai_api_key_from_file and str(_ai_api_key_from_file).strip()),
            "config_path": str(_secret_path),
            "config_exists": _secret_path.exists(),
            "load_error": _ai_config_load_error,
        }

    @app.post("/api/ai-analyze-scoring")
    async def ai_analyze_scoring(request: Request):
        """AI分析评分数据。body 可带 stage: stretch（抻面）/ boiling（下面及捞面）/ product（成品），默认 stretch。"""
        try:
            body = await request.json()
            video = body.get('video', '')
            data = body.get('data', {})
            stage = (body.get('stage') or 'stretch').strip().lower()
            is_boiling = stage == 'boiling'
            is_product = stage == 'product'

            if is_product:
                analysis_prompt = f"""
请分析以下兰州拉面成品评分数据，并给出专业的评估和建议：

图片/样本: {video}
综合得分（0-100）: {data.get('total_score', 0):.2f}
面条质感得分 S_texture: {data.get('s_texture', 0):.2f}
呈现加分 S_presentation: {data.get('s_presentation', 0):.2f}
面条质感等级: {data.get('noodle_quality', '')}
汤型: {data.get('soup_type', '')} | 面条辣椒: {data.get('noodle_chili', '')}
辅料: 牛肉 {data.get('beef', '')} 鸡蛋 {data.get('egg', '')} 蒜苗 {data.get('scallion', '')}

请从以下几个方面进行分析（注意：评分以面条质感为主体，清汤/红汤、辣椒有无不参与扣分）：
1. 整体成品质量评估
2. 面条质感维度分析
3. 呈现与辅料搭配建议
4. 可改进点（如有）

请用中文回答，语言专业简洁，控制篇幅。
"""
                analysis = await call_ai_service(analysis_prompt)
                return JSONResponse({"analysis": analysis})
            if is_boiling:
                analysis_prompt = f"""
请分析以下下面及捞面视频的评分数据，并给出专业的评估和建议：

视频名称: {video}
总分: {data.get('total_score', 0):.2f}
手部得分: {data.get('hand_score', 0):.2f}
面条/汤面得分: {data.get('noodle_rope_score', 0):.2f}
操作规范得分: {data.get('noodle_bundle_score', 0):.2f}
检测质量: {data.get('detection_quality', {}).get('quality_score', 0) * 100:.1f}%

请从以下几个方面进行分析：
1. 整体表现评估
2. 手部、面条/汤面、操作规范各维度得分分析
3. 存在的问题和改进建议

请用中文回答，语言专业简洁，控制篇幅。
"""
            else:
                analysis_prompt = f"""
请分析以下抻面视频的评分数据，并给出专业的评估和建议：

视频名称: {video}
总分: {data.get('total_score', 0):.2f}
手部得分: {data.get('hand_score', 0):.2f}
面条得分: {data.get('noodle_rope_score', 0):.2f}
面条束得分: {data.get('noodle_bundle_score', 0):.2f}
DTW评分: {data.get('dtw_result', {}).get('score', 0):.2f}
检测质量: {data.get('detection_quality', {}).get('quality_score', 0) * 100:.1f}%

请从以下几个方面进行分析：
1. 整体表现评估
2. 各维度得分分析
3. 存在的问题和改进建议
4. 与标准动作的对比分析

请用中文回答，语言专业简洁，控制篇幅。
"""

            analysis = await call_ai_service(analysis_prompt)
            return JSONResponse({"analysis": analysis})
        except Exception as e:
            return JSONResponse({"error": str(e), "analysis": "分析服务暂时不可用，请稍后重试。"})
    
    @app.post("/api/ai-chat")
    async def ai_chat(request: Request):
        """AI对话接口。body 可带 stage: stretch / boiling / product，默认 stretch。"""
        try:
            body = await request.json()
            video = body.get('video', '')
            data = body.get('data', {})
            question = body.get('question', '')
            history = body.get('history', [])
            stage = (body.get('stage') or 'stretch').strip().lower()
            is_boiling = stage == 'boiling'
            is_product = stage == 'product'

            if is_product:
                context = f"""
当前分析的是兰州拉面成品: {video}
评分数据摘要:
- 综合得分（0-100）: {data.get('total_score', 0):.2f}
- 面条质感得分: {data.get('s_texture', 0):.2f}
- 呈现加分: {data.get('s_presentation', 0):.2f}
- 面条质感等级: {data.get('noodle_quality', '')}
- 汤型/辣椒/辅料: {data.get('soup_type', '')} {data.get('noodle_chili', '')} 牛肉{data.get('beef', '')} 蛋{data.get('egg', '')} 蒜苗{data.get('scallion', '')}

用户问题: {question}

请基于以上成品评分数据回答用户的问题，用中文回答，语言要专业但易懂。评分以面条质感为主体，清汤/红汤与辣椒有无不参与扣分。
"""
                answer = await call_ai_service(context, history)
                return JSONResponse({"answer": answer})
            if is_boiling:
                context = f"""
当前分析的是下面及捞面视频: {video}
评分数据摘要:
- 总分: {data.get('total_score', 0):.2f}
- 手部得分: {data.get('hand_score', 0):.2f}
- 面条/汤面得分: {data.get('noodle_rope_score', 0):.2f}
- 操作规范得分: {data.get('noodle_bundle_score', 0):.2f}

用户问题: {question}

请基于以上评分数据回答用户的问题，用中文回答，语言要专业但易懂。
"""
            else:
                context = f"""
当前分析的视频: {video}
评分数据摘要:
- 总分: {data.get('total_score', 0):.2f}
- 手部得分: {data.get('hand_score', 0):.2f}
- 面条得分: {data.get('noodle_rope_score', 0):.2f}
- DTW评分: {data.get('dtw_result', {}).get('score', 0):.2f}

用户问题: {question}

请基于以上评分数据回答用户的问题，用中文回答，语言要专业但易懂。
"""

            answer = await call_ai_service(context, history)
            return JSONResponse({"answer": answer})
        except Exception as e:
            return JSONResponse({"error": str(e), "answer": "抱歉，处理您的请求时出错，请稍后重试。"})
    
    def _qiniu_exchange_ak_for_api_key(ak: str) -> str:
        """七牛云：用控制台 AK 兑换 AI 推理用的 API 密钥（sk- 开头）。文档：developer.qiniu.com/aitokenapi/12884"""
        ak = (ak or "").strip()
        if not ak:
            return ""
        for auth_header in (ak, f"Bearer {ak}"):
            try:
                import urllib.request
                req = urllib.request.Request(
                    "https://api.qnaigc.com/api/llmapikey",
                    headers={"Authorization": auth_header}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get("status") and data.get("api_key"):
                        return (data["api_key"] or "").strip()
            except Exception as e:
                print(f"[AI] 七牛 AK 兑换 API 密钥失败: {e}")
        return ""

    def _load_ai_api_config():
        """加载 AI API 配置（bysj2026 / qnaigc 兼容 OpenAI 接口）"""
        import os
        # 优先使用启动时从 configs/ai_api_secret.json 加载的密钥（与 start_web.py 同路径）
        api_key = (_ai_api_key_from_file or "").strip() or os.getenv('AI_API_KEY') or os.getenv('OPENAI_API_KEY')
        base_url = "https://api.qnaigc.com/v1"
        model = "deepseek-v3"
        if not api_key:
            secret_file = (project_root / "configs" / "ai_api_secret.json").resolve()
            if secret_file.exists():
                try:
                    with open(secret_file, 'r', encoding='utf-8') as f:
                        secret = json.load(f)
                        raw = (secret.get('api_key') or secret.get('secret_key') or "").strip()
                        if raw:
                            api_key = raw
                except Exception as e:
                    print(f"[AI] 读取 configs/ai_api_secret.json 失败: {e}")
        yaml_file = project_root / "configs" / "ai_api.yaml"
        if yaml_file.exists():
            try:
                import yaml
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                    if cfg:
                        base_url = cfg.get('base_url', base_url)
                        model = cfg.get('model', model)
            except Exception:
                pass
        # 七牛云：若当前是 AK（非 sk- 开头），尝试兑换为 AI 推理用的 API 密钥
        if api_key and "qnaigc.com" in base_url and not str(api_key).strip().lower().startswith("sk-"):
            exchanged = _qiniu_exchange_ak_for_api_key(str(api_key))
            if exchanged:
                api_key = exchanged
        return api_key, base_url, model

    def _do_openai_call(api_key: str, base_url: str, model: str, messages: list):
        """同步调用 OpenAI 兼容接口（在线程中执行，避免阻塞事件循环）"""
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6,
            max_tokens=800
        )

    async def call_ai_service(prompt: str, history: list = None):
        """调用七牛云 AI（OpenAI BaseURL: https://api.qnaigc.com/v1）"""
        import asyncio
        try:
            api_key, base_url, model = _load_ai_api_config()
            unconfigured_msg = (
                "AI 未配置或不可用。\n\n"
                "请将「七牛云 AI API KEY」（与文档中 OPENAI_API_KEY 一致）"
                "填入 configs/ai_api_secret.json 的 api_key 字段，保存后重启本服务。"
            )
            if not api_key:
                print("[AI] 未配置 api_key，跳过调用")
                return unconfigured_msg

            messages = [{"role": "system", "content": "你是一个专业的抻面动作分析专家，擅长分析动作评分数据并提供改进建议。请严格基于用户给出的评分数据和问题作答，不要泛泛而谈。"}]
            if history:
                for h in history[-6:]:
                    messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": prompt})

            print(f"[AI] 正在调用七牛 API: base_url={base_url}, model={model}")
            try:
                # 在线程池中执行同步请求，避免阻塞
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: _do_openai_call(api_key, base_url, model, messages)
                )
                text = (response.choices[0].message.content or "").strip()
                print(f"[AI] 调用成功，返回 {len(text)} 字")
                return text
            except ImportError:
                print("请安装 openai: pip install openai")
                return unconfigured_msg
            except Exception as e:
                err_msg = str(e).strip() or "未知错误"
                print(f"[AI] 七牛 API 调用失败: {e}")
                if "401" in err_msg or "Unauthorized" in err_msg or "authentication" in err_msg.lower():
                    tip = "请检查 configs/ai_api_secret.json 中的 api_key 是否与七牛控制台 → AI大模型推理 → API Key 一致（sk- 开头）。"
                elif "404" in err_msg or "model" in err_msg.lower() or "not found" in err_msg.lower():
                    tip = "当前模型 ID 可能已变更。请到 qiniu.com/ai 控制台查看「模型广场」中的可用 model，并修改 configs/ai_api.yaml 的 model 字段（当前为 deepseek-v3）。"
                else:
                    tip = "请确认 OpenAI BaseURL 为 https://api.qnaigc.com/v1、api_key 正确，且网络可访问七牛云。"
                return f"AI 接口调用失败：{err_msg}\n\n{tip}"
        except Exception as e:
            print(f"[AI] 服务异常: {e}")
            return f"AI 服务异常：{e}\n\n请检查配置与网络后重试。"

    def _is_chat_prompt(prompt: str) -> bool:
        """判断是否为「发送」对话请求（包含用户问题），而非「开始分析」"""
        return "用户问题:" in prompt

    def generate_mock_analysis(prompt: str, is_chat: bool = None) -> str:
        """生成模拟分析（仅当 AI 服务调用失败时使用）"""
        if is_chat is None:
            is_chat = _is_chat_prompt(prompt)
        if is_chat:
            question = ""
            for line in prompt.splitlines():
                if line.strip().startswith("用户问题:") or "用户问题:" in line:
                    question = line.split("用户问题:", 1)[-1].strip()
                    break
            if question:
                return f"（当前为模拟回复，AI 未接通）您问的是：「{question}」。请配置 configs/ai_api_secret.json 中的 api_key 后重启服务，即可获得针对该问题的真实分析。"
            return "请配置 configs/ai_api_secret.json 中的 api_key 后，可获得针对您问题的真实回答。"
        # 「开始分析」的固定报告
        return """## 评分数据分析报告

### 整体表现评估
根据评分数据显示，该视频的综合表现处于中等水平。各维度得分分布较为均匀，但仍有改进空间。

### 各维度得分分析
- **手部动作**: 得分反映了手部动作的规范性，建议关注动作的协调性和角度控制。
- **面条质量**: 面条得分显示了面条的拉伸质量，建议关注粗细均匀度和弹性。
- **DTW匹配**: DTW评分反映了与标准动作序列的相似度，得分越高说明动作越标准。

### 改进建议
1. 加强手部协调性训练，提高双手配合的默契度
2. 注意控制拉伸力度，保持面条粗细均匀
3. 参考标准动作，优化动作轨迹和节奏

### 与标准动作对比
建议观看标准视频（cm1, cm2, cm3）进行对比学习，重点关注动作的流畅性和规范性。"""
    
    def _sort_video_names_by_number(names: list, prefix: str) -> list:
        """按名称中的数字排序，使 xl10 在 xl9 后、cm10 在 cm9 后。"""
        import re
        def key(name):
            if not name.startswith(prefix):
                return (0, name)
            m = re.search(r"\d+", name)
            return (int(m.group()) if m else 0, name)
        return sorted(names, key=key)

    @app.get("/api/get_scoring_videos")
    async def get_scoring_videos(stage: str = "stretch"):
        """获取可用于评分的视频列表。stage=stretch 为抻面（cm*），stage=boiling_scooping 为下面及捞面（xl*）"""
        if stage == "boiling_scooping":
            labels_dir = project_root / "data" / "labels" / "下面及捞面"
            prefix = "xl"
        else:
            labels_dir = project_root / "data" / "labels" / "抻面"
            prefix = "cm"
        videos = []
        if labels_dir.exists():
            for item in labels_dir.iterdir():
                if item.is_dir() and item.name.startswith(prefix):
                    txt_files = list(item.glob("*.txt"))
                    if len(txt_files) > 1:
                        videos.append(item.name)
        videos = _sort_video_names_by_number(videos, prefix)
        return {"videos": videos, "stage": stage}
    
    @app.get("/api/get_video_frames")
    async def get_video_frames(video: str, keyframes_only: str = "false", include_additional: str = "false", stage: str = "stretch", only_with_labels: str = "true"):
        """获取视频的帧列表。only_with_labels=true 时只返回有对应标注文件的帧，便于评分时必有检测框"""
        import json
        if stage == "boiling_scooping":
            images_dir = project_root / "data" / "processed" / "下面及捞面" / video
            labels_dir = project_root / "data" / "labels" / "下面及捞面" / video
            scores_prefix = project_root / "data" / "scores" / "下面及捞面" / video
        else:
            images_dir = project_root / "data" / "processed" / "抻面" / video
            labels_dir = project_root / "data" / "labels" / "抻面" / video
            scores_prefix = project_root / "data" / "scores" / "抻面" / video

        def has_label(fname: str) -> bool:
            if not labels_dir.exists():
                return True
            base = fname.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
            return (labels_dir / (base + ".txt")).exists()

        if keyframes_only.lower() == "true":
            frames = []
            is_keyframes = True
            keyframes_file = scores_prefix / "key_frames.json"
            if keyframes_file.exists():
                try:
                    with open(keyframes_file, 'r', encoding='utf-8') as f:
                        keyframes_data = json.load(f)
                        frames.extend([item['frame'] for item in keyframes_data if 'frame' in item])
                except Exception as e:
                    print(f"[警告] 读取关键帧文件失败: {e}")
            if include_additional.lower() == "true" and stage != "boiling_scooping":
                additional_file = scores_prefix / "additional_key_frames.json"
                if additional_file.exists():
                    try:
                        with open(additional_file, 'r', encoding='utf-8') as f:
                            additional_data = json.load(f)
                            additional_frames = [item['frame'] for item in additional_data if 'frame' in item]
                            frames.extend(additional_frames)
                            frames = sorted(list(set(frames)))
                    except Exception as e:
                        print(f"[警告] 读取额外关键帧文件失败: {e}")
            if frames:
                if only_with_labels.lower() == "true":
                    frames = [f for f in frames if has_label(f)]
                return {"frames": frames, "is_keyframes": True, "includes_additional": include_additional.lower() == "true"}

        frames = []
        if images_dir.exists():
            frames = sorted([f.name for f in images_dir.glob("*.jpg")])
        if only_with_labels.lower() == "true":
            frames = [f for f in frames if has_label(f)]
        return {"frames": frames, "is_keyframes": False, "includes_additional": False}
    
    @app.get("/api/get_frame_image")
    async def get_frame_image(video: str, frame: str, stage: str = "stretch"):
        """获取帧图片。stage=stretch 抻面，stage=boiling_scooping 下面及捞面"""
        if stage == "boiling_scooping":
            image_path = project_root / "data" / "processed" / "下面及捞面" / video / frame
        else:
            image_path = project_root / "data" / "processed" / "抻面" / video / frame
        if image_path.exists():
            return FileResponse(str(image_path))
        return {"error": "图片不存在"}
    
    @app.get("/api/get_frame_detections")
    async def get_frame_detections(video: str, frame: str, stage: str = "stretch"):
        """获取帧的检测标注数据。stage=stretch 抻面，stage=boiling_scooping 下面及捞面"""
        import cv2
        frame = (frame or "").strip()
        if not frame or not video:
            return {"detections": []}
        if stage == "boiling_scooping":
            labels_dir = project_root / "data" / "labels" / "下面及捞面" / video
            images_dir = project_root / "data" / "processed" / "下面及捞面" / video
        else:
            labels_dir = project_root / "data" / "labels" / "抻面" / video
            images_dir = project_root / "data" / "processed" / "抻面" / video
        classes_file = labels_dir / "classes.txt"
        if not classes_file.exists():
            classes_file = labels_dir.parent / "classes.txt"
        if not classes_file.exists():
            return {"detections": []}
        with open(classes_file, 'r', encoding='utf-8-sig') as f:
            class_names = [line.strip() for line in f if line.strip()]
        base = frame.replace('.jpg', '').replace('.jpeg', '').replace('.png', '').strip()
        label_file = labels_dir / (base + ".txt")
        if not label_file.exists():
            label_file = None
        detections = []
        image_path = images_dir / frame
        if not image_path.exists():
            for ext in (".jpg", ".jpeg", ".png"):
                alt = images_dir / (base + ext)
                if alt.exists():
                    image_path = alt
                    break
        if label_file and label_file.exists() and image_path.exists():
            # Windows 下路径含中文时 cv2.imread 会失败，改为读字节再解码
            try:
                with open(image_path, 'rb') as f:
                    buf = f.read()
                import numpy as np
                arr = np.frombuffer(buf, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                with open(label_file, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                x_center = float(parts[1]) * w
                                y_center = float(parts[2]) * h
                                width = float(parts[3]) * w
                                height = float(parts[4]) * h
                                x1 = x_center - width / 2
                                y1 = y_center - height / 2
                                x2 = x_center + width / 2
                                y2 = y_center + height / 2
                                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                                detections.append({
                                    "class": cls_name,
                                    "class_id": cls_id,
                                    "xyxy": [x1, y1, x2, y2],
                                    "conf": 1.0
                                })
                            except (ValueError, IndexError):
                                pass
        return {"detections": detections}

    @app.get("/api/debug_frame_detection")
    async def debug_frame_detection(video: str, frame: str, stage: str = "stretch"):
        """诊断：检查标注/图片路径是否存在，便于排查“无检测结果”"""
        frame = (frame or "").strip()
        if stage == "boiling_scooping":
            labels_dir = project_root / "data" / "labels" / "下面及捞面" / video
            images_dir = project_root / "data" / "processed" / "下面及捞面" / video
        else:
            labels_dir = project_root / "data" / "labels" / "抻面" / video
            images_dir = project_root / "data" / "processed" / "抻面" / video
        base = frame.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").strip()
        label_file = labels_dir / (base + ".txt")
        image_path = images_dir / frame
        classes_file = labels_dir / "classes.txt"
        if not classes_file.exists():
            classes_file = labels_dir.parent / "classes.txt"
        out = {
            "video": video,
            "frame": frame,
            "stage": stage,
            "labels_dir": str(labels_dir),
            "labels_dir_exists": labels_dir.exists(),
            "label_file": str(label_file),
            "label_file_exists": label_file.exists(),
            "image_path": str(image_path),
            "image_exists": image_path.exists(),
            "classes_file_exists": classes_file.exists(),
        }
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8-sig") as f:
                lines = [l.strip() for l in f if l.strip() and len(l.split()) >= 5]
            out["label_line_count"] = len(lines)
        return out

    @app.post("/api/save_frame_scores")
    async def save_frame_scores(data: dict):
        """保存帧的评分数据。data 中可带 stage（stretch/boiling_scooping），默认抻面"""
        import json
        import time
        from pathlib import Path

        video = data.get("video")
        frame = data.get("frame")
        scores = data.get("scores", {})
        stage = data.get("stage", "stretch")

        if not video or not frame:
            return {"success": False, "message": "缺少必要参数"}

        if stage == "boiling_scooping":
            scores_dir = project_root / "data" / "scores" / "下面及捞面" / video
        else:
            scores_dir = project_root / "data" / "scores" / "抻面" / video
        scores_dir.mkdir(parents=True, exist_ok=True)
        score_file = scores_dir / f"{frame.replace('.jpg', '')}_scores.json"
        score_data = {
            "video": video,
            "frame": frame,
            "scores": scores,
            "stage": stage,
            "timestamp": time.time()
        }
        try:
            with open(score_file, 'w', encoding='utf-8') as f:
                json.dump(score_data, f, ensure_ascii=False, indent=2)
            return {"success": True, "message": "评分已保存"}
        except Exception as e:
            return {"success": False, "message": f"保存失败: {str(e)}"}
    
    @app.get("/api/hand_keypoints/{video_name}")
    async def get_hand_keypoints(video_name: str, stage: str = "stretch"):
        """获取指定视频的手部关键点JSON数据（预处理方案，无延迟）"""
        import json
        
        # 根据stage确定目录
        if stage == "boiling_scooping":
            keypoints_file = project_root / "data" / "scores" / "下面及捞面" / "hand_keypoints" / f"hand_keypoints_{video_name}.json"
        else:
            keypoints_file = project_root / "data" / "scores" / "抻面" / "hand_keypoints" / f"hand_keypoints_{video_name}.json"
        
        if not keypoints_file.exists():
            return {
                "success": False,
                "message": f"未找到视频 {video_name} 的关键点数据。请先运行预处理脚本：python scripts/extract_hand_keypoints_from_video.py",
                "error": "File not found"
            }
        
        try:
            with open(keypoints_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                "success": True,
                "data": data
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"读取关键点数据失败: {str(e)}",
                "error": str(e)
            }
    
    @app.post("/api/detect_hand_pose")
    async def detect_hand_pose(file: UploadFile = File(...)):
        """实时检测手部姿态关键点（接收视频帧图像）"""
        import cv2
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions
        import base64
        from io import BytesIO
        
        try:
            # 读取上传的图像
            image_data = await file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {
                    "success": False,
                    "message": "无法解码图像",
                    "error": "Invalid image"
                }
            
            H, W = frame.shape[:2]
            
            # 加载MediaPipe HandLandmarker
            model_dir = project_root / "weights" / "mediapipe"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "hand_landmarker.task"
            
            if not model_path.exists():
                return {
                    "success": False,
                    "message": "手部姿态模型未找到，请先运行提取脚本",
                    "error": "Model not found"
                }
            
            base_options = BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                running_mode=vision.RunningMode.IMAGE,
            )
            landmarker = vision.HandLandmarker.create_from_options(options)
            
            # 转换图像格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 检测手部关键点
            result = landmarker.detect(mp_image)
            
            # 转换结果
            hands_data = []
            if result.hand_landmarks:
                for hand_idx, hand in enumerate(result.hand_landmarks):
                    keypoints = []
                    for kp in hand:
                        visibility = getattr(kp, "visibility", None)
                        keypoints.append({
                            "x": float(kp.x * W),
                            "y": float(kp.y * H),
                            "z": float(kp.z),
                            "confidence": float(visibility if visibility is not None else 1.0)
                        })
                    hands_data.append({
                        "id": hand_idx,
                        "keypoints": keypoints
                    })
            
            return {
                "success": True,
                "hands": hands_data,
                "image_width": W,
                "image_height": H
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"检测失败: {str(e)}",
                "error": str(e)
            }
    
    @app.post("/api/process_frame_for_pose")
    async def process_frame_for_pose(
        video_name: str,
        frame_index: int,
        file: UploadFile = File(...)
    ):
        """
        实时处理视频帧进行手部姿态估计
        接收视频名称、帧索引和图像数据，基于hand标签进行ROI检测
        优化：使用MediaPipe单例，提高性能
        """
        import cv2
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions
        import threading
        
        try:
            # 读取上传的图像（使用高质量解码）
            image_data = await file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            # 使用IMREAD_COLOR保持原始质量
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {
                    "success": False,
                    "message": "无法解码图像",
                    "error": "Invalid image"
                }
            
            H, W = frame.shape[:2]
            
            # 使用单例获取MediaPipe HandLandmarker（避免重复创建）
            with _mediapipe_lock:
                landmarker = get_mediapipe_landmarker()
            
            if landmarker is None:
                return {
                    "success": False,
                    "message": "手部姿态模型未找到或初始化失败",
                    "error": "Model not found"
                }
            
            vision_module = vision
            
            # 读取hand标签（可选，用于ROI检测）
            label_dir = project_root / "data" / "labels" / "抻面" / video_name
            label_path = label_dir / f"{video_name}_{frame_index+1:05d}.txt"
            
            hand_boxes = []
            has_label_file = False
            if label_path.exists():
                has_label_file = True
                try:
                    for line in label_path.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls = int(parts[0])
                        if cls == 0:  # class 0 is hand
                            cx, cy, w, h = map(float, parts[1:5])
                            # 转换为xyxy格式
                            bw = w * W
                            bh = h * H
                            x = cx * W
                            y = cy * H
                            x1 = max(0, int(x - bw / 2))
                            y1 = max(0, int(y - bh / 2))
                            x2 = min(W - 1, int(x + bw / 2))
                            y2 = min(H - 1, int(y + bh / 2))
                            hand_boxes.append((x1, y1, x2, y2))
                except Exception as e:
                    print(f"[WARN] 读取标签文件失败: {e}")
            
            # 辅助函数：ROI检测
            def run_hand_on_roi(frame, bbox, landmarker, vision_module):
                import mediapipe as mp
                H, W = frame.shape[:2]
                x1, y1, x2, y2 = bbox
                scales = [1.0, 1.2, 1.4, 1.6]
                for s in scales:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    bw = (x2 - x1) * s
                    bh = (y2 - y1) * s
                    nx1 = int(max(0, cx - bw / 2))
                    ny1 = int(max(0, cy - bh / 2))
                    nx2 = int(min(W - 1, cx + bw / 2))
                    ny2 = int(min(H - 1, cy + bh / 2))
                    roi = frame[ny1:ny2, nx1:nx2]
                    if roi.size == 0:
                        continue
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect(mp_image)
                    if result.hand_landmarks:
                        hand = result.hand_landmarks[0]
                        h, w = roi.shape[:2]
                        kps = []
                        for kp in hand:
                            visibility = getattr(kp, "visibility", None)
                            kps.append({
                                "x": float(nx1 + kp.x * w),
                                "y": float(ny1 + kp.y * h),
                                "z": float(kp.z),
                                "confidence": float(visibility if visibility is not None else 1.0),
                            })
                        return kps
                return None
            
            # 辅助函数：全图检测
            def detect_full_frame(frame, landmarker, vision_module):
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)
                if not result.hand_landmarks:
                    return []
                h, w = frame.shape[:2]
                hands = []
                for hand in result.hand_landmarks:
                    kps = []
                    for kp in hand:
                        visibility = getattr(kp, "visibility", None)
                        kps.append({
                            "x": float(kp.x * w),
                            "y": float(kp.y * h),
                            "z": float(kp.z),
                            "confidence": float(visibility if visibility is not None else 1.0),
                        })
                    hands.append(kps)
                return hands
            
            # 检测手部关键点
            hands_detected = []
            
            # 先进行全图检测（无论是否有标签文件都进行）
            hands_ff = detect_full_frame(frame, landmarker, vision_module)
            
            if hand_boxes:
                # 如果有标签文件，优先使用ROI检测
                for bbox_idx, bbox in enumerate(hand_boxes):
                    kps = run_hand_on_roi(frame, bbox, landmarker, vision_module)
                    
                    # 如果ROI失败，尝试全图匹配最近手
                    if kps is None and hands_ff:
                        bx1, by1, bx2, by2 = bbox
                        bc = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
                        best = None
                        best_dist = 1e9
                        for hand_kps_full in hands_ff:
                            cx = np.mean([p["x"] for p in hand_kps_full])
                            cy = np.mean([p["y"] for p in hand_kps_full])
                            dist = np.linalg.norm(bc - np.array([cx, cy]))
                            in_box = (bx1 <= cx <= bx2) and (by1 <= cy <= by2)
                            if in_box and dist < best_dist:
                                best = hand_kps_full
                                best_dist = dist
                        if best:
                            kps = best
                    
                    if kps is not None:
                        hands_detected.append({
                            "id": bbox_idx,
                            "keypoints": kps
                        })
            else:
                # 如果没有标签文件，直接使用全图检测结果
                for hand_idx, hand_kps in enumerate(hands_ff):
                    hands_detected.append({
                        "id": hand_idx,
                        "keypoints": hand_kps
                    })
            
            return {
                "success": True,
                "frame_index": frame_index,
                "hands": hands_detected,
                "image_width": W,
                "image_height": H
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"处理失败: {str(e)}",
                "error": str(e)
            }
    
    @app.on_event("startup")
    async def _log_ai_config():
        try:
            api_key, base_url, model = _load_ai_api_config()
            secret_file = (project_root / "configs" / "ai_api_secret.json").resolve()
            loaded = bool(api_key and str(api_key).strip())
            print(f"[AI] 配置: {secret_file} | api_key 已加载: {'是' if loaded else '否'}")
        except Exception as e:
            print(f"[AI] 启动时检查配置异常: {e}")

    # 检查可用端口（connect_ex 成功=端口已被占用，失败=端口可用）
    import socket
    port = None
    for p in [8000, 8001, 8002, 8003, 8004, 8005]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.5)
            result = sock.connect_ex(('127.0.0.1', p))
        finally:
            sock.close()
        if result != 0:  # 连接失败说明端口未被占用，可用
            port = p
            break
    if port is None:
        print("\n[错误] 端口 8000~8005 均已被占用，请先关闭其他 Web 服务后再启动。")
        sys.exit(1)
    
    print(f"\n服务器地址: http://127.0.0.1:{port}")
    print(f"Web界面: http://127.0.0.1:{port}/")
    print(f"评分可视化: http://127.0.0.1:{port}/scoring-visualization")
    print(f"API文档: http://127.0.0.1:{port}/docs")
    print("\n按 Ctrl+C 停止服务器\n")
    print("="*60)
    
    # 启动时预加载检测模型，避免首次上传视频时才加载或报错
    print("[启动] 正在预加载检测模型...")
    try:
        from src.api.video_detection_api import get_detector, get_boiling_scooping_detector
        d_stretch = get_detector()
        if d_stretch.model is not None:
            print("[启动] 抻面检测模型已就绪")
        else:
            print("[启动] 抻面检测模型未找到或加载失败，请运行: python src/training/train_detection_model.py")
        d_boiling = get_boiling_scooping_detector()
        if d_boiling.model is not None:
            print("[启动] 下面及捞面检测模型已就绪")
        else:
            print("[启动] 下面及捞面检测模型未找到或加载失败，请运行: python src/training/train_boiling_scooping_model.py")
    except Exception as e:
        import traceback
        print("[启动] 预加载模型时出错（服务仍会启动，但检测可能不可用）:")
        traceback.print_exc()
    print("="*60)
    
    # 启动服务器
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("\n请安装依赖:")
    print("  pip install fastapi uvicorn python-multipart")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

