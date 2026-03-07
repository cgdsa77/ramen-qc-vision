#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""启动Web服务器 - 简化版"""
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional, List

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
    from fastapi import FastAPI, UploadFile, File, Form, Request, Query, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
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

    # 主页面背景图（data/web_background_images）
    web_bg_dir = project_root / "data" / "web_background_images"
    if web_bg_dir.exists():
        app.mount("/data/web_background_images", StaticFiles(directory=str(web_bg_dir)), name="web_background_images")
    
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

    # ---------- 毕设：用户权限与 Session（SQLite 或 MySQL）----------
    try:
        from src import auth_db
        auth_db.init_db()
        mode = auth_db._resolve_db_mode()
        if mode == "mysql":
            print("[认证] 用户库已初始化（MySQL），默认管理员 admin / admin123")
        else:
            print("[认证] 用户库已初始化（SQLite），默认管理员 admin / admin123")
    except Exception as e:
        print(f"[认证] 初始化失败: {e}，登录/用户管理将不可用")

    def _auth_session_id(request: Request) -> Optional[str]:
        return request.cookies.get("ramen_session")

    def _auth_current_user(request: Request):
        uid = _auth_session_id(request)
        return auth_db.get_session(uid)

    # 登录
    @app.post("/api/auth/login")
    async def api_auth_login(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"success": False, "error": "请求体无效"})
        username = (body.get("username") or "").strip()
        password = (body.get("password") or "")
        if not username or password is None:
            return JSONResponse(content={"success": False, "error": "请输入账号和密码"})
        user, session_id = auth_db.login(username, password)
        if user is None:
            return JSONResponse(content={"success": False, "error": "账号或密码错误"})
        if user.get("_disabled"):
            return JSONResponse(content={"success": False, "error": "账号已禁用，请联系管理员"})
        res = JSONResponse(content={
            "success": True,
            "user": {"id": user["id"], "username": user["username"], "role": user["role"], "name": user["name"]},
        })
        res.set_cookie(key="ramen_session", value=session_id, max_age=24 * 3600, httponly=True, samesite="lax")
        return res

    @app.get("/api/auth/session")
    async def api_auth_session(request: Request):
        u = _auth_current_user(request)
        if not u:
            return JSONResponse(status_code=401, content={"success": False, "error": "未登录"})
        return JSONResponse(content={"success": True, "user": u})

    @app.post("/api/auth/logout")
    async def api_auth_logout(request: Request):
        sid = _auth_session_id(request)
        if sid:
            auth_db.logout(sid)
        res = JSONResponse(content={"success": True})
        res.delete_cookie(key="ramen_session")
        return res

    # 用户管理（仅管理员 role=0）
    @app.get("/api/users")
    async def api_users_list(request: Request):
        u = _auth_current_user(request)
        if not u:
            return JSONResponse(status_code=401, content={"success": False, "error": "未登录"})
        if u.get("role") != 0:
            return JSONResponse(status_code=403, content={"success": False, "error": "无权限"})
        users = auth_db.list_users()
        return JSONResponse(content={"success": True, "users": users})

    @app.post("/api/users")
    async def api_users_create(request: Request):
        u = _auth_current_user(request)
        if not u or u.get("role") != 0:
            return JSONResponse(status_code=403, content={"success": False, "error": "无权限"})
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"success": False, "error": "请求体无效"})
        tid = body.get("assigned_trainer_id")
        if tid is not None and tid != "":
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                tid = None
        ok, msg, row = auth_db.create_user(
            body.get("username", ""),
            body.get("password", ""),
            int(body.get("role", 2)),
            body.get("name", ""),
            assigned_trainer_id=tid,
        )
        if not ok:
            return JSONResponse(content={"success": False, "error": msg})
        return JSONResponse(content={"success": True, "user": row, "message": msg})

    @app.put("/api/users/{uid}")
    async def api_users_update(uid: int, request: Request):
        u = _auth_current_user(request)
        if not u or u.get("role") != 0:
            return JSONResponse(status_code=403, content={"success": False, "error": "无权限"})
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"success": False, "error": "请求体无效"})
        kw = {}
        if "role" in body:
            kw["role"] = int(body["role"])
        if "name" in body:
            kw["name"] = body.get("name")
        if "status" in body:
            kw["status"] = int(body["status"])
        if "assigned_trainer_id" in body:
            tid = body["assigned_trainer_id"]
            if tid is not None and tid != "":
                try:
                    kw["assigned_trainer_id"] = int(tid)
                except (TypeError, ValueError):
                    kw["assigned_trainer_id"] = None
            else:
                kw["assigned_trainer_id"] = None
        ok, msg = auth_db.update_user(uid, **kw)
        if not ok:
            return JSONResponse(content={"success": False, "error": msg})
        return JSONResponse(content={"success": True, "message": msg})

    @app.delete("/api/users/{uid}")
    async def api_users_delete(uid: int, request: Request):
        u = _auth_current_user(request)
        if not u or u.get("role") != 0:
            return JSONResponse(status_code=403, content={"success": False, "error": "无权限"})
        ok, msg = auth_db.delete_user(uid)
        if not ok:
            return JSONResponse(content={"success": False, "error": msg})
        return JSONResponse(content={"success": True, "message": msg})

    @app.post("/api/users/{uid}/reset-password")
    async def api_users_reset_password(uid: int, request: Request):
        u = _auth_current_user(request)
        if not u or u.get("role") != 0:
            return JSONResponse(status_code=403, content={"success": False, "error": "无权限"})
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"success": False, "error": "请求体无效"})
        ok, msg = auth_db.reset_password(uid, body.get("new_password", ""))
        if not ok:
            return JSONResponse(content={"success": False, "error": msg})
        return JSONResponse(content={"success": True, "message": msg})

    # 培训师：获取当前培训师名下的学员列表（role=1 可调）
    @app.get("/api/trainer/students")
    async def api_trainer_students(request: Request):
        u = _auth_current_user(request)
        if not u:
            return JSONResponse(status_code=401, content={"success": False, "error": "未登录"})
        if u.get("role") != 1:
            return JSONResponse(status_code=403, content={"success": False, "error": "仅培训师可查看"})
        students = auth_db.get_students_by_trainer(int(u["id"]))
        return JSONResponse(content={"success": True, "students": students})

    # 培训师建议与学员回复（存 data/trainer_suggestions.json）；图片存 data/suggestion_images/
    _suggestions_file = project_root / "data" / "trainer_suggestions.json"
    _suggestion_images_dir = project_root / "data" / "suggestion_images"
    def _load_suggestions():
        if not _suggestions_file.exists():
            return {"threads": []}
        try:
            with open(_suggestions_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"threads": []}
    def _save_suggestions(data):
        _suggestions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_suggestions_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    def _find_thread(threads, trainer_id, student_id):
        for t in threads:
            if t.get("trainer_id") == trainer_id and t.get("student_id") == student_id:
                return t
        return None

    @app.get("/api/suggestions")
    async def api_suggestions_get(request: Request, student_id: Optional[int] = None):
        """培训师传 student_id 获取与该学员的对话；学员不传则获取自己的对话。"""
        u = _auth_current_user(request)
        if not u:
            return JSONResponse(status_code=401, content={"success": False, "error": "未登录"})
        data = _load_suggestions()
        threads = data.get("threads", [])
        if u.get("role") == 1 and student_id is not None:
            tid = int(u["id"])
            t = _find_thread(threads, tid, student_id)
            student = next((s for s in auth_db.get_students_by_trainer(tid) if s.get("id") == student_id), None)
            return JSONResponse(content={"success": True, "thread": t or {"trainer_id": tid, "student_id": student_id, "messages": []}, "student_name": (student and (student.get("name") or student.get("username"))) or ""})
        if u.get("role") == 2:
            full_user = next((x for x in auth_db.list_users() if x.get("id") == u["id"]), None)
            tid = (full_user or {}).get("assigned_trainer_id")
            if not tid:
                return JSONResponse(content={"success": True, "thread": {"messages": []}, "trainer_name": ""})
            t = _find_thread(threads, int(tid), int(u["id"]))
            trainer = next((x for x in auth_db.list_users() if x.get("id") == tid), None)
            return JSONResponse(content={"success": True, "thread": t or {"trainer_id": tid, "student_id": u["id"], "messages": []}, "trainer_name": (trainer and (trainer.get("name") or trainer.get("username"))) or ""})
        return JSONResponse(status_code=403, content={"success": False, "error": "无权限"})

    @app.post("/api/suggestions")
    async def api_suggestions_post(request: Request):
        """培训师给学员发建议（文本）。"""
        u = _auth_current_user(request)
        if not u or u.get("role") != 1:
            return JSONResponse(status_code=403, content={"success": False, "error": "仅培训师可发建议"})
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"success": False, "error": "请求体无效"})
        student_id = body.get("student_id")
        content = (body.get("content") or "").strip()
        if not content:
            return JSONResponse(content={"success": False, "error": "建议内容不能为空"})
        try:
            student_id = int(student_id)
        except (TypeError, ValueError):
            return JSONResponse(content={"success": False, "error": "学员无效"})
        students = auth_db.get_students_by_trainer(int(u["id"]))
        if not any(s.get("id") == student_id for s in students):
            return JSONResponse(status_code=403, content={"success": False, "error": "只能给所负责学员发建议"})
        data = _load_suggestions()
        threads = data.get("threads", [])
        t = _find_thread(threads, int(u["id"]), student_id)
        if not t:
            t = {"trainer_id": int(u["id"]), "student_id": student_id, "messages": []}
            threads.append(t)
        from datetime import datetime
        t["messages"].append({"role": "trainer", "content": content, "content_type": "text", "at": datetime.now().isoformat()})
        _save_suggestions(data)
        return JSONResponse(content={"success": True, "message": "已发送"})

    @app.post("/api/suggestions/upload")
    async def api_suggestions_upload(request: Request, student_id: str = Form(...), content: str = Form(""), image: UploadFile = File(None)):
        """培训师给学员发建议（图片，可选附文字）。支持从本地选择或拖拽上传。"""
        u = _auth_current_user(request)
        if not u or u.get("role") != 1:
            return JSONResponse(status_code=403, content={"success": False, "error": "仅培训师可发建议"})
        try:
            sid = int(student_id)
        except (TypeError, ValueError):
            return JSONResponse(content={"success": False, "error": "学员无效"})
        students = auth_db.get_students_by_trainer(int(u["id"]))
        if not any(s.get("id") == sid for s in students):
            return JSONResponse(status_code=403, content={"success": False, "error": "只能给所负责学员发建议"})
        if not image or not image.filename:
            return JSONResponse(content={"success": False, "error": "请选择或拖拽上传一张图片"})
        ext = (Path(image.filename).suffix or ".jpg").lower()
        if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
            return JSONResponse(content={"success": False, "error": "仅支持图片格式（jpg/png/gif等）"})
        _suggestion_images_dir.mkdir(parents=True, exist_ok=True)
        safe_name = str(uuid.uuid4()) + ext
        path = _suggestion_images_dir / safe_name
        try:
            content_bytes = await image.read()
            path.write_bytes(content_bytes)
        except Exception as e:
            return JSONResponse(content={"success": False, "error": "保存图片失败: " + str(e)})
        data = _load_suggestions()
        threads = data.get("threads", [])
        t = _find_thread(threads, int(u["id"]), sid)
        if not t:
            t = {"trainer_id": int(u["id"]), "student_id": sid, "messages": []}
            threads.append(t)
        from datetime import datetime
        text_part = (content or "").strip()
        if text_part:
            t["messages"].append({"role": "trainer", "content": text_part, "content_type": "text", "at": datetime.now().isoformat()})
        t["messages"].append({"role": "trainer", "content": safe_name, "content_type": "image", "at": datetime.now().isoformat()})
        _save_suggestions(data)
        return JSONResponse(content={"success": True, "message": "已发送"})

    def _suggestion_image_media_type(name: str):
        n = name.lower()
        if n.endswith((".png",)): return "image/png"
        if n.endswith((".gif",)): return "image/gif"
        if n.endswith((".webp",)): return "image/webp"
        if n.endswith((".bmp",)): return "image/bmp"
        return "image/jpeg"

    @app.get("/api/suggestions/image/{filename:path}")
    async def api_suggestions_image(filename: str):
        """返回建议中上传的图片，供前端展示。"""
        if not filename or ".." in filename:
            raise HTTPException(status_code=400, detail="invalid")
        name = Path(filename.replace("\\", "/").strip("/")).name
        if not name:
            raise HTTPException(status_code=400, detail="invalid")
        path = _suggestion_images_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        try:
            path.resolve().relative_to(_suggestion_images_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(str(path), media_type=_suggestion_image_media_type(name))

    @app.post("/api/suggestions/reply")
    async def api_suggestions_reply(request: Request):
        """学员回复培训师。"""
        u = _auth_current_user(request)
        if not u or u.get("role") != 2:
            return JSONResponse(status_code=403, content={"success": False, "error": "仅学员可回复"})
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"success": False, "error": "请求体无效"})
        content = (body.get("content") or "").strip()
        if not content:
            return JSONResponse(content={"success": False, "error": "回复内容不能为空"})
        full_user = next((x for x in auth_db.list_users() if x.get("id") == u["id"]), None)
        tid = (full_user or {}).get("assigned_trainer_id")
        if not tid:
            return JSONResponse(content={"success": False, "error": "未分配培训师"})
        data = _load_suggestions()
        threads = data.get("threads", [])
        t = _find_thread(threads, int(tid), int(u["id"]))
        if not t:
            t = {"trainer_id": int(tid), "student_id": int(u["id"]), "messages": []}
            threads.append(t)
        from datetime import datetime
        t["messages"].append({"role": "student", "content": content, "content_type": "text", "at": datetime.now().isoformat()})
        _save_suggestions(data)
        return JSONResponse(content={"success": True, "message": "已回复"})

    # 培训师列表（管理员/培训师用于下拉：所属培训师）
    @app.get("/api/trainers")
    async def api_trainers_list(request: Request):
        u = _auth_current_user(request)
        if not u:
            return JSONResponse(status_code=401, content={"success": False, "error": "未登录"})
        if u.get("role") not in (0, 1):
            return JSONResponse(status_code=403, content={"success": False, "error": "无权限"})
        trainers = auth_db.list_trainers()
        return JSONResponse(content={"success": True, "trainers": trainers})

    # 评分标准（培训师/管理员可读可改，权重总和须为 1.0）
    _scoring_rules_paths = {
        "stretch": project_root / "data" / "scores" / "抻面" / "scoring_rules.json",
        "boiling": project_root / "data" / "scores" / "下面及捞面" / "scoring_rules.json",
    }
    _overall_weight_labels = {
        "stretch": {"noodle_rope": "面绳", "hand": "手部", "noodle_bundle": "面束"},
        "boiling": {"noodle_rope": "面绳", "hand": "手部", "tools_noodle": "工具面", "soup_noodle": "汤面"},
    }

    @app.get("/api/scoring-rules")
    async def api_scoring_rules_get(request: Request, stage: str = Query("stretch", description="stretch|boiling")):
        u = _auth_current_user(request)
        if not u or u.get("role") != 1:
            return JSONResponse(status_code=403, content={"success": False, "error": "仅培训师可查看评分标准"})
        if stage not in _scoring_rules_paths:
            return JSONResponse(status_code=400, content={"success": False, "error": "stage 须为 stretch 或 boiling"})
        path = _scoring_rules_paths[stage]
        if not path.exists():
            return JSONResponse(content={"success": False, "error": "规则文件不存在"})
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            weights = data.get("overall_weights") or {}
            labels = _overall_weight_labels.get(stage) or {}
            return JSONResponse(content={
                "success": True,
                "stage": stage,
                "overall_weights": weights,
                "labels": labels,
                "min_confidence": data.get("min_confidence", 0.3),
                "min_frame_ratio": data.get("min_frame_ratio", 0.1),
                "pass_threshold": data.get("pass_threshold", 60),
                "excellent_threshold": data.get("excellent_threshold", 85),
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

    def _score_100_and_grade(stage: str, average_overall_score_1_5: float) -> tuple:
        """根据规则中的及格线/优秀线，将 1-5 分制转为百分制并得到等级。"""
        path = _scoring_rules_paths.get(stage)
        if not path or not path.exists():
            return round((average_overall_score_1_5 - 1) / 4 * 100, 1), None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return round((average_overall_score_1_5 - 1) / 4 * 100, 1), None
        pass_t = float(data.get("pass_threshold", 60))
        excellent_t = float(data.get("excellent_threshold", 85))
        score_100 = (average_overall_score_1_5 - 1) / 4 * 100
        score_100 = round(max(0, min(100, score_100)), 1)
        if score_100 >= excellent_t:
            grade = "优秀"
        elif score_100 >= pass_t:
            grade = "良好"
        else:
            grade = "不及格"
        return score_100, grade

    @app.put("/api/scoring-rules")
    async def api_scoring_rules_put(request: Request):
        u = _auth_current_user(request)
        if not u or u.get("role") != 1:
            return JSONResponse(status_code=403, content={"success": False, "error": "仅培训师可修改评分标准"})
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"success": False, "error": "请求体无效"})
        stage = body.get("stage")
        if stage not in _scoring_rules_paths:
            return JSONResponse(content={"success": False, "error": "stage 须为 stretch 或 boiling"})
        new_weights = body.get("overall_weights")
        if not isinstance(new_weights, dict):
            return JSONResponse(content={"success": False, "error": "overall_weights 须为对象"})
        try:
            total = sum(float(v) for v in new_weights.values())
        except (TypeError, ValueError):
            return JSONResponse(content={"success": False, "error": "权重须为数字"})
        if abs(total - 1.0) > 1e-6:
            return JSONResponse(content={"success": False, "error": "权重总和须为 100%（即 1.0），当前为 " + str(round(total * 100, 1)) + "%"})
        # 可选：评分行为参数
        min_confidence = body.get("min_confidence")
        min_frame_ratio = body.get("min_frame_ratio")
        pass_threshold = body.get("pass_threshold")
        excellent_threshold = body.get("excellent_threshold")
        if min_confidence is not None:
            v = float(min_confidence)
            if v < 0 or v > 1:
                return JSONResponse(content={"success": False, "error": "最低参与置信度须在 0～1 之间"})
        if min_frame_ratio is not None:
            v = float(min_frame_ratio)
            if v < 0 or v > 1:
                return JSONResponse(content={"success": False, "error": "视频有效帧比例须在 0～1 之间"})
        if pass_threshold is not None:
            v = float(pass_threshold)
            if v < 0 or v > 100:
                return JSONResponse(content={"success": False, "error": "及格线须在 0～100 之间"})
        if excellent_threshold is not None:
            v = float(excellent_threshold)
            if v < 0 or v > 100:
                return JSONResponse(content={"success": False, "error": "优秀线须在 0～100 之间"})
        if pass_threshold is not None and excellent_threshold is not None and float(excellent_threshold) < float(pass_threshold):
            return JSONResponse(content={"success": False, "error": "优秀线不得低于及格线"})
        path = _scoring_rules_paths[stage]
        if not path.exists():
            return JSONResponse(content={"success": False, "error": "规则文件不存在"})
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["overall_weights"] = {k: float(v) for k, v in new_weights.items()}
            if min_confidence is not None:
                data["min_confidence"] = float(min_confidence)
            if min_frame_ratio is not None:
                data["min_frame_ratio"] = float(min_frame_ratio)
            if pass_threshold is not None:
                data["pass_threshold"] = float(pass_threshold)
            if excellent_threshold is not None:
                data["excellent_threshold"] = float(excellent_threshold)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return JSONResponse(content={"success": True, "message": "已保存"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

    # ---------- 毕设用户权限结束 ----------
    
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
        """实时检测骨架线页面（摄像头 + 手部骨架线叠加）"""
        web_file = web_dir / "realtime_skeleton.html"
        if web_file.exists():
            return FileResponse(str(web_file))
        return {"message": "实时骨架线页面未找到"}

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

    # 实时流取消标记：前端点击「停止」时设置，生成器检查后退出并释放摄像头
    _realtime_stream_cancel = {}
    # 检测框 EMA 平滑：stream_id -> {"hand": [x1,y1,x2,y2], "head": [x1,y1,x2,y2]}，使框更跟手、更稳
    _realtime_det_prev_boxes = {}
    # 缺检时沿用上一帧框的帧数：stream_id -> {"hand": n, "head": n}，最多延续 2 帧，减少闪烁、提升跟随感
    _realtime_det_hold_count = {}

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

    def _realtime_combined_stream_generator(device_index: int, stage: str, conf: float, stream_id: str, use_mediapipe_hands: bool = True, use_mediapipe_face: bool = True):
        """单摄像头一次打开，每帧同时做检测与骨架叠加，左右拼接为一帧输出，避免同一摄像头开两次。"""
        import cv2
        import sys
        import mediapipe as mp
        cap = None
        try:
            if sys.platform == "win32":
                cap = cv2.VideoCapture(int(device_index), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(int(device_index))
            if not cap.isOpened():
                if sys.platform == "win32":
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = cv2.VideoCapture(int(device_index))
                if not cap.isOpened():
                    print("[realtime-combined] 无法打开摄像头 device=%s" % device_index)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"\r\n")
                    return
            if stage == "boiling_scooping":
                from src.api.video_detection_api import get_boiling_scooping_detector
                detector = get_boiling_scooping_detector(model_type="cpu")
            else:
                from src.api.video_detection_api import get_detector
                detector = get_detector(model_type="cpu")
            hand_landmarker = get_mediapipe_landmarker()
            pose_landmarker = get_mediapipe_pose_landmarker()
            target_h = 360
            while True:
                if _realtime_stream_cancel.get(stream_id):
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if _realtime_stream_cancel.get(stream_id):
                    break
                H, W = frame.shape[:2]
                frame_det = frame.copy()
                try:
                    _, detections = detector.detect_frame(frame_det, conf_threshold=conf, draw_boxes=False, use_mediapipe_hands=use_mediapipe_hands, use_mediapipe_face=use_mediapipe_face, stage=stage)
                    prev = _realtime_det_prev_boxes.setdefault(stream_id, {})
                    hold_count = _realtime_det_hold_count.setdefault(stream_id, {})
                    ema_alpha = 0.62
                    smoothed_list = []
                    seen_hand_head = set()
                    for d in (detections or []):
                        cls = (d.get("class") or "").strip().lower()
                        xyxy = list(d.get("xyxy", [0, 0, 0, 0]))
                        if cls in ("hand", "head"):
                            seen_hand_head.add(cls)
                            hold_count[cls] = 0
                            if len(xyxy) == 4 and prev.get(cls) is not None:
                                p = prev[cls]
                                xyxy = [ema_alpha * xyxy[i] + (1.0 - ema_alpha) * p[i] for i in range(4)]
                            prev[cls] = xyxy
                        smoothed_list.append({**d, "xyxy": xyxy})
                    for cls in ("hand", "head"):
                        if cls not in seen_hand_head and prev.get(cls) is not None:
                            n = hold_count.get(cls, 0)
                            if n < 2:
                                smoothed_list.append({"class": cls, "class_id": -1 if cls == "hand" else -2, "conf": 0.5, "xyxy": list(prev[cls])})
                                hold_count[cls] = n + 1
                    frame_det = detector._draw_detections(frame_det, smoothed_list)
                except Exception:
                    try:
                        annotated, _ = detector.detect_frame(frame_det, conf_threshold=conf, draw_boxes=True, use_mediapipe_hands=use_mediapipe_hands, use_mediapipe_face=use_mediapipe_face, stage=stage)
                        frame_det = annotated
                    except Exception:
                        pass
                frame_skel = frame.copy()
                try:
                    import time
                    now = time.time()
                    te = 1.0 / 30.0
                    if stream_id in _realtime_skeleton_last_te:
                        dt = now - _realtime_skeleton_last_te[stream_id]
                        if 0.005 <= dt <= 0.2:
                            te = dt
                    _realtime_skeleton_last_te[stream_id] = now
                    rgb = cv2.cvtColor(frame_skel, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    hand_result = None
                    if hand_landmarker is not None:
                        try:
                            hand_result = hand_landmarker.detect(mp_image)
                        except Exception:
                            hand_result = None
                    smoothed_hands = _smooth_hand_landmarks_one_euro(stream_id, hand_result, te) if hand_result else []
                    left_wrist_xy = right_wrist_xy = None
                    hands_xy_for_draw = []
                    for hand_side, points_21 in smoothed_hands:
                        hands_xy_for_draw.append(points_21)
                        if len(points_21) > 0:
                            w = points_21[0]
                            if "LEFT" in hand_side:
                                left_wrist_xy = w
                            elif "RIGHT" in hand_side:
                                right_wrist_xy = w
                    if pose_landmarker is not None:
                        try:
                            pose_result = pose_landmarker.detect(mp_image)
                            if getattr(pose_result, "pose_landmarks", None):
                                smoothed = _smooth_pose_one_euro(stream_id, pose_result.pose_landmarks, te)
                                if smoothed:
                                    persons_xy = [list(p) for p in smoothed]
                                    if persons_xy and len(persons_xy[0]) >= 17:
                                        if left_wrist_xy is not None:
                                            persons_xy[0][15] = left_wrist_xy
                                        if right_wrist_xy is not None:
                                            persons_xy[0][16] = right_wrist_xy
                                    frame_skel = _draw_pose_skeleton_frame_from_normalized(frame_skel, persons_xy, H, W)
                        except Exception:
                            pass
                    if hands_xy_for_draw:
                        try:
                            frame_skel = _draw_hand_skeleton_frame_from_normalized(frame_skel, hands_xy_for_draw, H, W)
                        except Exception:
                            pass
                except Exception:
                    pass
                if frame_det.shape[0] != target_h or frame_det.shape[1] != int(W * target_h / H):
                    frame_det = cv2.resize(frame_det, (int(W * target_h / H), target_h))
                if frame_skel.shape[0] != target_h or frame_skel.shape[1] != int(W * target_h / H):
                    frame_skel = cv2.resize(frame_skel, (int(W * target_h / H), target_h))
                combined = cv2.hconcat([frame_det, frame_skel])
                if _realtime_stream_cancel.get(stream_id):
                    break
                _, jpeg = cv2.imencode(".jpg", combined)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg.tobytes() + b"\r\n")
        except GeneratorExit:
            pass
        except Exception as e:
            print(f"[realtime-combined] 错误: {e}")
        finally:
            _realtime_stream_cancel.pop(stream_id, None)
            _realtime_skeleton_pose_prev.pop(stream_id, None)
            _realtime_skeleton_1euro_pose.pop(stream_id, None)
            _realtime_skeleton_1euro_hand.pop(stream_id, None)
            _realtime_skeleton_last_te.pop(stream_id, None)
            _realtime_det_prev_boxes.pop(stream_id, None)
            _realtime_det_hold_count.pop(stream_id, None)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                print("[realtime-combined] 摄像头已释放 stream_id=%s" % stream_id)

    # MediaPipe Pose 33 点连接关系（身体骨架）。不含手腕到手指的连线，手部仅由 HandLandmarker 绿色骨架展示。
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 躯干与手臂（到手腕为止）
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
        (27, 29), (29, 31), (28, 30), (30, 32),
        # 不绘制 (15,17),(15,19),(15,21),(16,18),(16,20),(16,22)，避免与绿色手部骨架重复
    ]
    # 身体模型中手部关键点索引（15~22 为手腕与手指），不绘制蓝点，仅保留绿色手部骨架
    POSE_HAND_INDICES = {15, 16, 17, 18, 19, 20, 21, 22}

    # 各 stream 上一帧平滑后的身体关键点，用于时序平滑（list of list of (x,y) per person）
    _realtime_skeleton_pose_prev = {}
    # One Euro Filter 状态：身体 33 点 * 2 坐标；手部 2 手 * 21 点 * 2 坐标
    _realtime_skeleton_1euro_pose = {}
    _realtime_skeleton_1euro_hand = {}
    _realtime_skeleton_last_te = {}  # 用于估算 dt（秒）

    class _OneEuroFilter:
        """One Euro Filter：根据速度自适应平滑，快速运动时减少滞后、慢速时减少抖动。"""
        def __init__(self, min_cutoff=1.2, beta=0.5, d_cutoff=1.0):
            self.min_cutoff = min_cutoff
            self.beta = beta
            self.d_cutoff = d_cutoff
            self._x_prev = None
            self._d_prev = 0.0

        def filter(self, x, te=1.0 / 30.0):
            if self._x_prev is None:
                self._x_prev = x
                return x
            import math
            d = (x - self._x_prev) / te if te > 0 else 0.0
            d_filtered = self._smooth_alpha(te, self.d_cutoff)(d, self._d_prev)
            cutoff = self.min_cutoff + self.beta * abs(d_filtered)
            x_filtered = self._smooth_alpha(te, cutoff)(x, self._x_prev)
            self._d_prev = d_filtered
            self._x_prev = x_filtered
            return x_filtered

        def _smooth_alpha(self, te, cutoff):
            tau = 1.0 / (2.0 * 3.14159265359 * cutoff) if cutoff > 0 else 0.0
            alpha = 1.0 / (1.0 + tau / te) if te > 0 else 1.0
            def f(x, x_prev):
                return alpha * x + (1.0 - alpha) * x_prev
            return f

    def _ensure_1euro_pose(stream_id):
        """确保当前 stream 的身体 One Euro Filter 已初始化（33 点 * x,y）。参数偏跟手、略减滞后。"""
        if stream_id not in _realtime_skeleton_1euro_pose:
            # min_cutoff 略高、beta 略高：快速运动时更跟手，位置更贴合
            _realtime_skeleton_1euro_pose[stream_id] = [
                [_OneEuroFilter(min_cutoff=1.4, beta=0.75, d_cutoff=1.0), _OneEuroFilter(min_cutoff=1.4, beta=0.75, d_cutoff=1.0)]
                for _ in range(33)
            ]
        return _realtime_skeleton_1euro_pose[stream_id]

    def _smooth_pose_one_euro(stream_id, pose_landmarks_list, te=1.0 / 30.0):
        """用 One Euro Filter 平滑身体关键点，快速运动时更跟手、慢速时更稳。"""
        if not pose_landmarks_list:
            return None
        filters = _ensure_1euro_pose(stream_id)
        out = []
        for person in pose_landmarks_list:
            try:
                landmarks = list(person) if hasattr(person, "__iter__") else list(getattr(person, "landmark", []))
            except Exception:
                continue
            if len(landmarks) < 33:
                continue
            smoothed = []
            for i in range(33):
                x = getattr(landmarks[i], "x", 0)
                y = getattr(landmarks[i], "y", 0)
                sx = filters[i][0].filter(x, te)
                sy = filters[i][1].filter(y, te)
                smoothed.append((sx, sy, getattr(landmarks[i], "z", 0)))
            out.append(smoothed)
        return out

    def _ensure_1euro_hand(stream_id):
        if stream_id not in _realtime_skeleton_1euro_hand:
            # 手部同样提高跟随性，减少滞后
            hand_filters = []
            for _ in range(2):
                hand_filters.append([
                    [_OneEuroFilter(min_cutoff=1.5, beta=0.8, d_cutoff=1.0), _OneEuroFilter(min_cutoff=1.5, beta=0.8, d_cutoff=1.0)]
                    for _ in range(21)
                ])
            _realtime_skeleton_1euro_hand[stream_id] = hand_filters
        return _realtime_skeleton_1euro_hand[stream_id]

    def _smooth_hand_landmarks_one_euro(stream_id, hand_result, te=1.0 / 30.0):
        """用 One Euro Filter 平滑手部 21 点，返回 [(hand_side, list of 21 (x,y)), ...]。手与身体共用同一平滑手腕，贴合度更好。"""
        if not hand_result or not getattr(hand_result, "hand_landmarks", None):
            return []
        filters_list = _ensure_1euro_hand(stream_id)
        out = []
        for hi, hand_pts in enumerate(hand_result.hand_landmarks):
            if len(hand_pts) < 21:
                continue
            hand_side = ""
            try:
                if getattr(hand_result, "handedness", None) and hi < len(hand_result.handedness):
                    cat = hand_result.handedness[hi]
                    hand_side = (getattr(cat, "category_name", None) or getattr(cat, "display_name", None) or "").strip().upper()
            except Exception:
                pass
            filters = filters_list[hi % 2]
            smoothed = []
            for i in range(21):
                x = getattr(hand_pts[i], "x", 0)
                y = getattr(hand_pts[i], "y", 0)
                sx = filters[i][0].filter(x, te)
                sy = filters[i][1].filter(y, te)
                smoothed.append((sx, sy))
            out.append((hand_side, smoothed))
        return out

    def _smooth_pose_landmarks(stream_id, pose_landmarks_list, alpha=0.45):
        """对身体关键点做指数移动平均，减少抖动、更贴合姿态变化。alpha 越大越平滑、延迟略增。"""
        if not pose_landmarks_list:
            return None
        prev_all = _realtime_skeleton_pose_prev.get(stream_id)
        out = []
        for pi, person in enumerate(pose_landmarks_list):
            try:
                landmarks = list(person) if hasattr(person, "__iter__") else list(getattr(person, "landmark", []))
            except Exception:
                continue
            if len(landmarks) < 33:
                continue
            curr = [(getattr(lm, "x", 0), getattr(lm, "y", 0), getattr(lm, "z", 0)) for lm in landmarks]
            prev_one = prev_all[pi] if prev_all and pi < len(prev_all) and len(prev_all[pi]) == len(curr) else None
            if prev_one is not None:
                curr = [
                    (alpha * p[0] + (1 - alpha) * c[0], alpha * p[1] + (1 - alpha) * c[1], alpha * p[2] + (1 - alpha) * c[2])
                    for p, c in zip(prev_one, curr)
                ]
            out.append(curr)
        if out:
            _realtime_skeleton_pose_prev[stream_id] = out
        return out

    def _draw_pose_skeleton_frame_from_normalized(frame_bgr, persons_xy, H, W):
        """根据已归一化的身体关键点绘制骨架（persons_xy: 每人 33 个 (x,y)，0~1）。不绘手部蓝点。"""
        import cv2
        for points_norm in (persons_xy or []):
            if len(points_norm) < 33:
                continue
            points = [(int(p[0] * W), int(p[1] * H)) for p in points_norm]
            for (i, j) in POSE_CONNECTIONS:
                if i < len(points) and j < len(points):
                    cv2.line(frame_bgr, points[i], points[j], (255, 165, 0), 2)
            for idx, pt in enumerate(points):
                if idx not in POSE_HAND_INDICES:
                    cv2.circle(frame_bgr, pt, 3, (255, 0, 0), -1)
        return frame_bgr

    def _draw_pose_skeleton_frame(frame_bgr, pose_landmarks_list, H, W):
        """在 BGR 帧上绘制身体骨架线。不绘制手腕与手指的蓝点/连线，手部由 HandLandmarker 绿色展示。"""
        import cv2
        for pose_landmarks in (pose_landmarks_list or []):
            try:
                landmarks = list(pose_landmarks) if hasattr(pose_landmarks, "__iter__") else list(getattr(pose_landmarks, "landmark", []))
            except Exception:
                landmarks = []
            if len(landmarks) < 33:
                continue
            points = []
            for lm in landmarks:
                x = int(getattr(lm, "x", 0) * W)
                y = int(getattr(lm, "y", 0) * H)
                points.append((x, y))
            for (i, j) in POSE_CONNECTIONS:
                if i < len(points) and j < len(points):
                    cv2.line(frame_bgr, points[i], points[j], (255, 165, 0), 2)  # 身体用橙色
            for idx, pt in enumerate(points):
                if idx not in POSE_HAND_INDICES:  # 手部关键点不画蓝点，避免与绿色手部骨架重叠
                    cv2.circle(frame_bgr, pt, 3, (255, 0, 0), -1)  # 身体关键点蓝色
        return frame_bgr

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    def _draw_hand_skeleton_frame_from_normalized(frame_bgr, hands_xy_list, H, W):
        """根据已归一化的手部关键点绘制骨架。hands_xy_list: 每手 21 个 (x,y)，0~1。用于平滑后的手部绘制。"""
        import cv2
        for hand_xy in (hands_xy_list or []):
            if len(hand_xy) < 21:
                continue
            points = [(int(p[0] * W), int(p[1] * H)) for p in hand_xy]
            for (i, j) in HAND_CONNECTIONS:
                if i < len(points) and j < len(points):
                    cv2.line(frame_bgr, points[i], points[j], (0, 255, 0), 2)
            for pt in points:
                cv2.circle(frame_bgr, pt, 3, (0, 0, 255), -1)
        return frame_bgr

    def _skeleton_overlay_single_frame(frame_bgr):
        """对单帧做手部+身体骨架叠加（不依赖 stream 状态，用于本地视频逐帧分析）。返回绘制后的 BGR 帧。"""
        import cv2
        import mediapipe as mp
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        out = frame_bgr.copy()
        pose_landmarker = get_mediapipe_pose_landmarker()
        hand_landmarker = get_mediapipe_landmarker()
        persons_xy = None
        if pose_landmarker:
            try:
                pose_result = pose_landmarker.detect(mp_image)
                if getattr(pose_result, "pose_landmarks", None):
                    pl = pose_result.pose_landmarks
                    persons_xy = []
                    for person in pl:
                        try:
                            lm = list(person) if hasattr(person, "__iter__") else list(getattr(person, "landmark", []))
                        except Exception:
                            continue
                        if len(lm) < 33:
                            continue
                        persons_xy.append([(getattr(lm[i], "x", 0), getattr(lm[i], "y", 0), getattr(lm[i], "z", 0)) for i in range(33)])
            except Exception:
                pass
        left_wrist_xy = right_wrist_xy = None
        hands_xy_for_draw = []
        if hand_landmarker:
            try:
                hand_result = hand_landmarker.detect(mp_image)
                if getattr(hand_result, "hand_landmarks", None):
                    for hi, hand_pts in enumerate(hand_result.hand_landmarks):
                        if len(hand_pts) < 21:
                            continue
                        hand_side = ""
                        if getattr(hand_result, "handedness", None) and hi < len(hand_result.handedness):
                            cat = hand_result.handedness[hi]
                            hand_side = (getattr(cat, "category_name", None) or getattr(cat, "display_name", None) or "").strip().upper()
                        pts = [(getattr(hand_pts[i], "x", 0), getattr(hand_pts[i], "y", 0)) for i in range(21)]
                        hands_xy_for_draw.append(pts)
                        if pts and "LEFT" in hand_side:
                            left_wrist_xy = pts[0]
                        elif pts and "RIGHT" in hand_side:
                            right_wrist_xy = pts[0]
            except Exception:
                pass
        if persons_xy and len(persons_xy[0]) >= 17:
            if left_wrist_xy is not None:
                persons_xy[0][15] = left_wrist_xy
            if right_wrist_xy is not None:
                persons_xy[0][16] = right_wrist_xy
            out = _draw_pose_skeleton_frame_from_normalized(out, persons_xy, H, W)
        if hands_xy_for_draw:
            out = _draw_hand_skeleton_frame_from_normalized(out, hands_xy_for_draw, H, W)
        return out

    def _draw_hand_skeleton_frame(frame_bgr, hands_landmarks_list, H, W):
        """在 BGR 帧上绘制手部骨架线（关键点+连线）。hands_landmarks_list 为 MediaPipe 返回的 hand_landmarks 列表，坐标为归一化。"""
        import cv2
        for hand_landmarks in (hands_landmarks_list or []):
            points = []
            for lm in hand_landmarks:
                x = int(lm.x * W)
                y = int(lm.y * H)
                points.append((x, y))
            if len(points) < 21:
                continue
            for (i, j) in HAND_CONNECTIONS:
                if i < len(points) and j < len(points):
                    cv2.line(frame_bgr, points[i], points[j], (0, 255, 0), 2)
            for pt in points:
                cv2.circle(frame_bgr, pt, 3, (0, 0, 255), -1)
        return frame_bgr

    def _realtime_skeleton_stream_generator(device_index: int, stream_id: str):
        """生成手部+身体骨架线的 MJPEG 流，支持 ivcam 等外接摄像头。先绘身体再绘手部。"""
        import cv2
        import sys
        import mediapipe as mp
        cap = None
        try:
            if sys.platform == "win32":
                cap = cv2.VideoCapture(int(device_index), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(int(device_index))
            if not cap.isOpened():
                if sys.platform == "win32":
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = cv2.VideoCapture(int(device_index))
                if not cap.isOpened():
                    print("[realtime-skeleton-stream] 无法打开摄像头 device=%s" % device_index)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"\r\n")
                    return
            hand_landmarker = get_mediapipe_landmarker()
            pose_landmarker = get_mediapipe_pose_landmarker()
            if hand_landmarker is None and pose_landmarker is None:
                print("[realtime-skeleton-stream] 未找到骨架模型，请放置 weights/mediapipe/ 下 hand_landmarker.task 或 pose_landmarker_lite.task")
                ret, frame = cap.read()
                if ret and frame is not None:
                    cv2.putText(frame, "Place hand_landmarker.task or pose_landmarker_lite.task", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    _, jpeg = cv2.imencode(".jpg", frame)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg.tobytes() + b"\r\n")
                return
            while True:
                if _realtime_stream_cancel.get(stream_id):
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if _realtime_stream_cancel.get(stream_id):
                    break
                H, W = frame.shape[:2]
                import time
                now = time.time()
                te = 1.0 / 30.0
                if stream_id in _realtime_skeleton_last_te:
                    dt = now - _realtime_skeleton_last_te[stream_id]
                    if 0.005 <= dt <= 0.2:
                        te = dt
                _realtime_skeleton_last_te[stream_id] = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hand_result = None
                if hand_landmarker is not None:
                    try:
                        hand_result = hand_landmarker.detect(mp_image)
                    except Exception:
                        hand_result = None
                smoothed_hands = _smooth_hand_landmarks_one_euro(stream_id, hand_result, te) if hand_result else []
                left_wrist_xy = right_wrist_xy = None
                hands_xy_for_draw = []
                for hand_side, points_21 in smoothed_hands:
                    hands_xy_for_draw.append(points_21)
                    if len(points_21) > 0:
                        w = points_21[0]
                        if "LEFT" in hand_side:
                            left_wrist_xy = w
                        elif "RIGHT" in hand_side:
                            right_wrist_xy = w
                if pose_landmarker is not None:
                    try:
                        pose_result = pose_landmarker.detect(mp_image)
                        if getattr(pose_result, "pose_landmarks", None):
                            smoothed = _smooth_pose_one_euro(stream_id, pose_result.pose_landmarks, te)
                            if smoothed:
                                persons_xy = [list(p) for p in smoothed]
                                if persons_xy and len(persons_xy[0]) >= 17:
                                    if left_wrist_xy is not None:
                                        persons_xy[0][15] = left_wrist_xy
                                    if right_wrist_xy is not None:
                                        persons_xy[0][16] = right_wrist_xy
                                frame = _draw_pose_skeleton_frame_from_normalized(frame.copy(), persons_xy, H, W)
                            else:
                                frame = _draw_pose_skeleton_frame(frame.copy(), pose_result.pose_landmarks, H, W)
                    except Exception:
                        pass
                if hands_xy_for_draw:
                    try:
                        frame = _draw_hand_skeleton_frame_from_normalized(frame, hands_xy_for_draw, H, W)
                    except Exception:
                        if hand_result and getattr(hand_result, "hand_landmarks", None):
                            frame = _draw_hand_skeleton_frame(frame, hand_result.hand_landmarks, H, W)
                elif hand_result and getattr(hand_result, "hand_landmarks", None):
                    try:
                        frame = _draw_hand_skeleton_frame(frame, hand_result.hand_landmarks, H, W)
                    except Exception:
                        pass
                if _realtime_stream_cancel.get(stream_id):
                    break
                _, jpeg = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg.tobytes() + b"\r\n")
        except GeneratorExit:
            pass
        except Exception as e:
            print(f"[realtime-skeleton-stream] 错误: {e}")
        finally:
            _realtime_stream_cancel.pop(stream_id, None)
            _realtime_skeleton_pose_prev.pop(stream_id, None)
            _realtime_skeleton_1euro_pose.pop(stream_id, None)
            _realtime_skeleton_1euro_hand.pop(stream_id, None)
            _realtime_skeleton_last_te.pop(stream_id, None)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                print("[realtime-skeleton-stream] 摄像头已释放 stream_id=%s" % stream_id)

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

    @app.get("/api/realtime-skeleton-stream")
    async def api_realtime_skeleton_stream(device: int = 0, stream_id: str = ""):
        """实时骨架线 MJPEG 流：仅绘制手部骨架，支持 ivcam/外接摄像头。与实时监测共用 stop 接口。"""
        if not stream_id:
            import uuid
            stream_id = str(uuid.uuid4())
        device_index = int(device)
        if device_index in (0, 1):
            import sys
            if sys.platform == "win32":
                device_index = 1 if device_index == 0 else 0
        return StreamingResponse(
            _realtime_skeleton_stream_generator(device_index, stream_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/realtime-combined-stream")
    async def api_realtime_combined_stream(device: int = 0, stage: str = "stretch", conf: float = 0.4, stream_id: str = "", use_mediapipe_hands: bool = True, use_mediapipe_face: bool = True):
        """单路合并流：一次打开摄像头，每帧左半检测、右半骨架线，避免摄像头被占只显一个画面。"""
        if not stream_id:
            import uuid
            stream_id = str(uuid.uuid4())
        device_index = int(device)
        if device_index in (0, 1):
            import sys
            if sys.platform == "win32":
                device_index = 1 if device_index == 0 else 0
        return StreamingResponse(
            _realtime_combined_stream_generator(device_index, stage, conf, stream_id, use_mediapipe_hands, use_mediapipe_face),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/skeleton_overlay_frame")
    async def api_skeleton_overlay_frame(file: UploadFile = File(...)):
        """接收一帧图像，返回叠加手部+身体骨架后的 JPEG。用于本地视频骨架线逐帧分析。使用本地 hand/pose 模型。"""
        import cv2
        import numpy as np
        try:
            data = await file.read()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return Response(status_code=400, content=b"Invalid image")
            out = _skeleton_overlay_single_frame(frame)
            _, jpeg = cv2.imencode(".jpg", out)
            return Response(content=jpeg.tobytes(), media_type="image/jpeg")
        except Exception as e:
            print(f"[skeleton_overlay_frame] {e}")
            return Response(status_code=500, content=b"Server error")

    @app.get("/video-skeleton-ivcam")
    async def video_skeleton_ivcam():
        """本地视频骨架线分析页（从 ivcam 选择视频，逐帧叠加骨架后播放）。"""
        web_file = web_dir / "video_skeleton_ivcam.html"
        if web_file.exists():
            return FileResponse(str(web_file))
        return {"message": "页面未找到"}

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
    
    def _normalize_conv_bn_error(err_text: str) -> str:
        """将 Conv.bn 相关原始异常转为对用户友好的说明（检测/评分过程中若推理报错会触发）。"""
        if err_text and "Conv" in err_text and "bn" in err_text:
            return (
                "模型与当前 ultralytics 版本不兼容（Conv.bn 结构变更）。"
                "请尝试：pip install ultralytics==8.0.200 后重启服务；"
                "或使用当前环境重新训练得到新的 best.pt。详见 docs/模型与ultralytics版本兼容说明.md"
            )
        return err_text or ""

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
                        _detect_jobs[job_id].update(phase="error", error=_normalize_conv_bn_error(str(e)), message="检测失败")
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
                            _detect_jobs[jid].update(phase="error", error=_normalize_conv_bn_error(str(e)), message="检测失败")
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
            err = _normalize_conv_bn_error(str(e))
            return {"success": False, "error": err, "message": f"检测失败: {e}"}
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
                            _detect_jobs[jid].update(phase="error", error=_normalize_conv_bn_error(str(e)), message="检测失败")
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
            err = _normalize_conv_bn_error(str(e))
            return {"success": False, "error": err, "message": f"检测失败: {e}"}
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
                            avg = video_score_result.get('average_overall_score', 0)
                            score_100, grade = _score_100_and_grade("stretch", avg)
                            out = {
                                "success": True, "stage": stage, "total_frames": total_frames,
                                "scored_frames": video_score_result.get('scored_frames', 0),
                                "average_overall_score": round(avg, 2),
                                "score_100": score_100, "grade": grade,
                                "class_average_scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                                "scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                                "total_score": round(avg, 2),
                                "details": video_score_result.get('class_average_scores', {}),
                                "rules_used": "基于标准数据集的评分规则和阈值",
                                "frame_scores_sample": video_score_result.get('frame_scores', [])[:5],
                                "model_source": model_path or "当前最佳抻面模型(latest best.pt)",
                                "score_basis": "最佳抻面模型检测 + 规则/图像特征评分",
                                "normalization_detail": video_score_result.get('normalization_detail'),
                                "warning": video_score_result.get('warning'),
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
                            avg = video_score_result.get('average_overall_score', 0)
                            score_100, grade = _score_100_and_grade("boiling", avg)
                            out = {
                                "success": True, "stage": stage, "total_frames": total_frames,
                                "scored_frames": video_score_result.get('scored_frames', 0),
                                "average_overall_score": round(avg, 2),
                                "score_100": score_100, "grade": grade,
                                "class_average_scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                                "scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                                "total_score": round(avg, 2),
                                "details": video_score_result.get('class_average_scores', {}),
                                "rules_used": "下面及捞面规则（scoring_rules.json）",
                                "frame_scores_sample": video_score_result.get('frame_scores', [])[:5],
                                "model_source": getattr(detector, "model_path", None) or "",
                                "score_basis": "检测 + 图像特征 + 规则（与抻面一致）",
                                "normalization_detail": video_score_result.get('normalization_detail'),
                                "warning": video_score_result.get('warning'),
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
                        # 仅用有检测到的类别求平均，不因缺失类别惩罚
                        present = [c for c in classes if scores.get(c, 0) > 0]
                        total_score_raw = round(sum(scores[c] for c in present) / len(present), 3) if present else 0.0
                        # 下面及捞面：占位规则为 0~1，统一换算为与抻面一致的 5 分制（1 + raw*4）
                        if stage == "boiling_scooping":
                            total_score = round(1.0 + total_score_raw * 4, 2)
                            for cls in list(details.keys()):
                                if not cls.startswith("_"):
                                    details[cls]["raw_score_0_1"] = scores.get(cls, 0)
                            details["_raw_total_0_1"] = total_score_raw
                            scores = {k: round(1.0 + v * 4, 2) for k, v in scores.items()}
                            # 操作规范：仅基于有检测到的工具/汤面；若只一类有则用该类
                            tn, sn = scores.get("tools_noodle", 0), scores.get("soup_noodle", 0)
                            scores["noodle_bundle"] = round((tn + sn) / 2.0, 2) if (tn and sn) else round(tn or sn, 2)
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
                            _score_jobs[jid].update(phase="error", error=_normalize_conv_bn_error(str(e)), message="评分失败")
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
                    avg = video_score_result.get('average_overall_score', 0)
                    score_100, grade = _score_100_and_grade("stretch", avg)
                    return {
                        "success": True,
                        "stage": stage,
                        "total_frames": total_frames,
                        "scored_frames": video_score_result.get('scored_frames', 0),
                        "average_overall_score": round(avg, 2),
                        "score_100": score_100,
                        "grade": grade,
                        "class_average_scores": {
                            k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()
                        },
                        "scores": {
                            k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()
                        },
                        "total_score": round(avg, 2),
                        "details": video_score_result.get('class_average_scores', {}),
                        "rules_used": "基于标准数据集的评分规则和阈值",
                        "frame_scores_sample": video_score_result.get('frame_scores', [])[:5],
                        "model_source": model_path or "当前最佳抻面模型(latest best.pt)",
                        "score_basis": "最佳抻面模型检测 + 规则/图像特征评分",
                        "normalization_detail": video_score_result.get('normalization_detail'),
                        "warning": video_score_result.get('warning'),
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
                    avg = video_score_result.get('average_overall_score', 0)
                    score_100, grade = _score_100_and_grade("boiling", avg)
                    return {
                        "success": True, "stage": stage, "total_frames": total_frames,
                        "scored_frames": video_score_result.get('scored_frames', 0),
                        "average_overall_score": round(avg, 2),
                        "score_100": score_100, "grade": grade,
                        "class_average_scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                        "scores": {k: round(v, 2) for k, v in video_score_result.get('class_average_scores', {}).items()},
                        "total_score": round(avg, 2),
                        "details": video_score_result.get('class_average_scores', {}),
                        "rules_used": "下面及捞面规则（scoring_rules.json）",
                        "frame_scores_sample": video_score_result.get('frame_scores', [])[:5],
                        "model_source": getattr(detector, "model_path", None) or "",
                        "score_basis": "检测 + 图像特征 + 规则（与抻面一致）",
                        "normalization_detail": video_score_result.get('normalization_detail'),
                        "warning": video_score_result.get('warning'),
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

            # 仅用有检测到的类别求平均，不因缺失类别惩罚
            present = [c for c in classes if scores.get(c, 0) > 0]
            total_score_raw = round(sum(scores[c] for c in present) / len(present), 3) if present else 0.0
            # 下面及捞面：占位规则为 0~1，统一换算为与抻面一致的 5 分制
            if stage == "boiling_scooping":
                total_score = round(1.0 + total_score_raw * 4, 2)
                for cls in list(details.keys()):
                    if not cls.startswith("_"):
                        details[cls]["raw_score_0_1"] = scores.get(cls, 0)
                details["_raw_total_0_1"] = total_score_raw
                scores = {k: round(1.0 + v * 4, 2) for k, v in scores.items()}
                # 操作规范：仅基于有检测到的工具/汤面；若只一类有则用该类
                tn, sn = scores.get("tools_noodle", 0), scores.get("soup_noodle", 0)
                scores["noodle_bundle"] = round((tn + sn) / 2.0, 2) if (tn and sn) else round(tn or sn, 2)
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

    def _make_score_result_html(stage: str, video_name: str, total: float, hand: float, rope: float, bundle: float, ai_analysis: str, save_time: str) -> str:
        """生成与界面一致的 HTML（文字+表格+柱状图），用于保存与在 Web 内展示。"""
        stage_label = "抻面" if stage == "stretch" else "下面及捞面"
        dim1 = "面条/汤面" if stage == "boiling" else "面条"
        dim2 = "操作规范" if stage == "boiling" else "面条束"
        labels = ["总分", "手部", dim1, dim2]
        values = [total, hand, rope, bundle]
        max_val = 5.0
        bar_colors = ["rgba(102,126,234,0.8)", "rgba(255,99,132,0.8)", "rgba(54,162,235,0.8)", "rgba(75,192,192,0.8)"]
        if stage == "boiling":
            bar_colors = ["rgba(13,148,136,0.8)", "rgba(255,99,132,0.8)", "rgba(54,162,235,0.8)", "rgba(245,87,108,0.8)"]
        bar_rows = "".join(
            f'<div class="bar-row"><span class="bar-label">{labels[i]}</span><div class="bar-track"><div class="bar-fill" style="width:{100 * values[i] / max_val}%;background:{bar_colors[i]}"></div></div><span class="bar-value">{values[i]:.2f}</span></div>'
            for i in range(4)
        )
        ai_escaped = (ai_analysis or "（无分析内容）").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>评分结果 - {stage_label}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Microsoft YaHei', sans-serif; padding: 20px; background: #f5f5f5; color: #333; }}
.card {{ background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
h1 {{ font-size: 1.2em; color: #667eea; margin-bottom: 12px; }}
.meta {{ font-size: 13px; color: #666; margin-bottom: 16px; line-height: 1.8; }}
table {{ border-collapse: collapse; width: 100%; max-width: 320px; margin-bottom: 16px; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #f8f9fa; }}
.section-title {{ font-size: 12px; color: #666; margin-bottom: 8px; }}
.ai-text {{ background: #f8f9fa; padding: 14px; border-radius: 8px; font-size: 14px; line-height: 1.6; white-space: pre-wrap; min-height: 60px; }}
.chart-title {{ font-size: 12px; color: #666; margin: 16px 0 8px; }}
.bar-row {{ display: flex; align-items: center; margin-bottom: 10px; gap: 10px; }}
.bar-label {{ width: 90px; font-size: 13px; }}
.bar-track {{ flex: 1; height: 24px; background: #eee; border-radius: 4px; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
.bar-value {{ width: 44px; font-size: 13px; font-weight: bold; }}
</style></head><body>
<div class="card"><h1>评分结果</h1>
<div class="meta">视频/样本：{video_name}<br>阶段：{stage_label}<br>保存时间：{save_time}</div>
<div class="section-title">评分汇总</div>
<table><tr><th>维度</th><th>得分</th></tr>
<tr><td>总分</td><td>{total:.2f}</td></tr>
<tr><td>手部</td><td>{hand:.2f}</td></tr>
<tr><td>{dim1}</td><td>{rope:.2f}</td></tr>
<tr><td>{dim2}</td><td>{bundle:.2f}</td></tr>
</table>
<div class="section-title">AI 分析</div>
<div class="ai-text">{ai_escaped}</div>
<div class="chart-title">各维度得分（精简）</div>
<div class="chart-bars">{bar_rows}</div>
</div></body></html>"""

    @app.post("/api/save_score_result")
    async def save_score_result(request: Request):
        """将当前评分结果与 AI 分析保存为 HTML（按用户分目录，便于学员/培训师查看与删除）"""
        import re
        from datetime import datetime
        u = _auth_current_user(request)
        user_id = int(u["id"]) if u else 0
        try:
            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse({"success": False, "error": "请求体不是有效 JSON: " + str(e)}, status_code=400)
            if not isinstance(body, dict):
                body = {}
            stage = (body.get("stage") or "stretch").strip().lower()
            video_name = (body.get("video_name") or body.get("videoName") or "未命名").strip()
            score_data = body.get("score_data") or body.get("scoreData") or {}
            if not isinstance(score_data, dict):
                score_data = {}
            ai_analysis = (body.get("ai_analysis") or body.get("aiAnalysis") or "").strip()
            if not isinstance(ai_analysis, str):
                ai_analysis = str(ai_analysis) if ai_analysis is not None else ""
            subdir = "cm_scoring_ras" if stage == "stretch" else "xl_scoring_ras"
            save_dir = project_root / "data" / "Scoring_results_and_suggestions" / subdir / str(user_id)
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return JSONResponse({"success": False, "error": "创建目录失败: " + str(e)}, status_code=500)
            safe_name = re.sub(r"[^\w\u4e00-\u9fa5\-.]", "_", video_name)[:80]
            if not safe_name:
                safe_name = "未命名"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = f"{stage}_{ts}_{safe_name}.html"
            filepath = save_dir / filename
            try:
                total = float(score_data.get("total_score") or 0)
                hand = float(score_data.get("hand_score") or 0)
                rope = float(score_data.get("noodle_rope_score") or 0)
                bundle = float(score_data.get("noodle_bundle_score") or 0)
            except (TypeError, ValueError):
                total = hand = rope = bundle = 0.0
            html_content = _make_score_result_html(stage, video_name, total, hand, rope, bundle, ai_analysis, save_time)
            try:
                filepath.write_text(html_content, encoding="utf-8")
            except Exception as e:
                return JSONResponse({"success": False, "error": "写入文件失败: " + str(e)}, status_code=500)
            rel_path = f"data/Scoring_results_and_suggestions/{subdir}/{user_id}/{filename}"
            return JSONResponse({"success": True, "path": rel_path, "filename": filename, "user_id": user_id})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    def _score_result_target_user(request: Request, student_id: Optional[int] = None):
        """确定要查看/操作的评分记录所属用户：学员看自己，培训师可传 student_id 看学员，管理员可看任意。"""
        u = _auth_current_user(request)
        if not u:
            return None, "未登录"
        role = u.get("role")
        uid = u.get("id")
        if student_id is not None:
            if role == 0:
                return student_id, None
            if role == 1:
                students = auth_db.get_students_by_trainer(int(uid))
                if not any(s.get("id") == student_id for s in students):
                    return None, "只能查看所负责学员的记录"
                return student_id, None
            if role == 2:
                # 学员/厨师查看自己的记录时，前端会传 user_id（列表返回的记录所属人），仅允许与当前用户一致
                if int(student_id) == int(uid):
                    return student_id, None
                return None, "无权限查看他人记录"
            return None, "无权限查看他人记录"
        return uid, None

    @app.get("/api/list_saved_score_results")
    async def list_saved_score_results(request: Request, stage: str = "stretch", student_id: Optional[int] = None):
        """列出已保存的评分结果（按用户）：学员看自己，培训师可传 student_id 看学员。"""
        target_uid, err = _score_result_target_user(request, student_id)
        if err:
            return JSONResponse({"success": False, "error": err}, status_code=403 if "未登录" in err else 403)
        subdir = "cm_scoring_ras" if stage.strip().lower() == "stretch" else "xl_scoring_ras"
        folder = project_root / "data" / "Scoring_results_and_suggestions" / subdir / str(target_uid)
        if not folder.exists():
            return JSONResponse({"success": True, "files": [], "user_id": target_uid})
        files = []
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() == ".html":
                try:
                    mtime = f.stat().st_mtime
                    files.append({"filename": f.name, "mtime": mtime, "user_id": target_uid})
                except Exception:
                    pass
        files.sort(key=lambda x: x["mtime"], reverse=True)
        return JSONResponse({"success": True, "files": files, "user_id": target_uid})

    @app.get("/api/saved_score_result_content")
    async def saved_score_result_content(request: Request, stage: str = "stretch", filename: str = "", user_id: Optional[int] = None):
        """获取已保存的评分结果 HTML 内容。user_id 为记录所属用户（列表返回）；不传则用当前用户。"""
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            return JSONResponse({"success": False, "error": "无效文件名"}, status_code=400)
        u = _auth_current_user(request)
        if not u:
            return JSONResponse({"success": False, "error": "未登录"}, status_code=401)
        target_uid, err = _score_result_target_user(request, user_id)
        if err:
            return JSONResponse({"success": False, "error": err}, status_code=403)
        subdir = "cm_scoring_ras" if stage.strip().lower() == "stretch" else "xl_scoring_ras"
        folder = project_root / "data" / "Scoring_results_and_suggestions" / subdir / str(target_uid)
        filepath = folder / filename
        if not filepath.is_file() or filepath.suffix.lower() != ".html":
            return JSONResponse({"success": False, "error": "文件不存在"}, status_code=404)
        try:
            content = filepath.read_text(encoding="utf-8")
            return JSONResponse({"success": True, "content": content})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.delete("/api/saved_score_result")
    async def delete_saved_score_result(request: Request, stage: str = "stretch", filename: str = "", user_id: Optional[int] = None):
        """删除一条已保存的评分记录。仅本人、其培训师或管理员可删。"""
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            return JSONResponse({"success": False, "error": "无效文件名"}, status_code=400)
        target_uid, err = _score_result_target_user(request, user_id)
        if err:
            return JSONResponse({"success": False, "error": err}, status_code=403)
        subdir = "cm_scoring_ras" if stage.strip().lower() == "stretch" else "xl_scoring_ras"
        folder = project_root / "data" / "Scoring_results_and_suggestions" / subdir / str(target_uid)
        filepath = folder / filename
        if not filepath.is_file() or filepath.suffix.lower() != ".html":
            return JSONResponse({"success": False, "error": "文件不存在"}, status_code=404)
        try:
            filepath.unlink()
            return JSONResponse({"success": True, "message": "已删除"})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.get("/api/open_score_result_folder")
    async def open_score_result_folder(stage: str = "stretch"):
        """在服务器本机打开已保存评分结果所在文件夹。stage=stretch 打开 cm_scoring_ras，stage=boiling 打开 xl_scoring_ras。"""
        import subprocess
        import sys
        subdir = "cm_scoring_ras" if stage.strip().lower() == "stretch" else "xl_scoring_ras"
        folder = project_root / "data" / "Scoring_results_and_suggestions" / subdir
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path_str = str(folder.resolve())
            if sys.platform == "win32":
                os.startfile(path_str)
            elif sys.platform == "darwin":
                subprocess.run(["open", path_str], check=False)
            else:
                subprocess.run(["xdg-open", path_str], check=False)
            return JSONResponse({"success": True, "path": path_str})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    # ========== MediaPipe单例（优化性能） ==========
    _mediapipe_landmarker = None
    _mediapipe_pose_landmarker = None
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

    def get_mediapipe_pose_landmarker():
        """获取MediaPipe PoseLandmarker单例（身体 33 点），用于实时骨架线身体部分。"""
        global _mediapipe_pose_landmarker
        if _mediapipe_pose_landmarker is None:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python import BaseOptions
            model_dir = project_root / "weights" / "mediapipe"
            model_dir.mkdir(parents=True, exist_ok=True)
            # 优先使用 heavy（更准），其次 lite（更快），最后 pose_landmarker.task
            for name in ("pose_landmarker_heavy.task", "pose_landmarker_lite.task", "pose_landmarker.task"):
                model_path = model_dir / name
                if model_path.exists():
                    break
            else:
                return None
            try:
                base_options = BaseOptions(model_asset_path=str(model_path))
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    num_poses=2,
                    running_mode=vision.RunningMode.IMAGE,
                )
                _mediapipe_pose_landmarker = vision.PoseLandmarker.create_from_options(options)
                print("[OK] MediaPipe PoseLandmarker 已创建（模型: %s，身体骨架更准请使用 pose_landmarker_heavy.task）" % model_path.name)
            except Exception as e:
                print(f"[提示] PoseLandmarker 未加载（仅显示手部骨架）: {e}")
                return None
        return _mediapipe_pose_landmarker
    
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

    @app.get("/score-history")
    async def score_history_page():
        """我的评分记录：按抻面/下面捞面/成品分页查看与删除，培训师可查学员并给建议。"""
        web_file = web_dir / "score_history.html"
        if web_file.exists():
            r = FileResponse(str(web_file))
            r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return r
        return {"message": "评分记录页面未找到"}

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

