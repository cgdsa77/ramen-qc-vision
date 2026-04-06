"""
认证、用户、培训师建议、评分标准 API（由 start_web 注册）。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse

from src.webapp.auth_deps import auth_current_user, auth_session_id
from src.webapp.scoring_rules_helpers import OVERALL_WEIGHT_LABELS, scoring_rules_paths


def register_auth_suggestions_routes(app: FastAPI, project_root: Path) -> None:
    """注册登录、用户、建议、评分标准等路由。"""
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
        u = auth_current_user(request)
        if not u:
            return JSONResponse(status_code=401, content={"success": False, "error": "未登录"})
        return JSONResponse(content={"success": True, "user": u})
    
    @app.post("/api/auth/logout")
    async def api_auth_logout(request: Request):
        sid = auth_session_id(request)
        if sid:
            auth_db.logout(sid)
        res = JSONResponse(content={"success": True})
        res.delete_cookie(key="ramen_session")
        return res
    
    # 用户管理（仅管理员 role=0）
    @app.get("/api/users")
    async def api_users_list(request: Request):
        u = auth_current_user(request)
        if not u:
            return JSONResponse(status_code=401, content={"success": False, "error": "未登录"})
        if u.get("role") != 0:
            return JSONResponse(status_code=403, content={"success": False, "error": "无权限"})
        users = auth_db.list_users()
        return JSONResponse(content={"success": True, "users": users})
    
    @app.post("/api/users")
    async def api_users_create(request: Request):
        u = auth_current_user(request)
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
        u = auth_current_user(request)
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
        u = auth_current_user(request)
        if not u or u.get("role") != 0:
            return JSONResponse(status_code=403, content={"success": False, "error": "无权限"})
        ok, msg = auth_db.delete_user(uid)
        if not ok:
            return JSONResponse(content={"success": False, "error": msg})
        return JSONResponse(content={"success": True, "message": msg})
    
    @app.post("/api/users/{uid}/reset-password")
    async def api_users_reset_password(uid: int, request: Request):
        u = auth_current_user(request)
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
        u = auth_current_user(request)
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
        u = auth_current_user(request)
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
        u = auth_current_user(request)
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
        u = auth_current_user(request)
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
        u = auth_current_user(request)
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
        u = auth_current_user(request)
        if not u:
            return JSONResponse(status_code=401, content={"success": False, "error": "未登录"})
        if u.get("role") not in (0, 1):
            return JSONResponse(status_code=403, content={"success": False, "error": "无权限"})
        trainers = auth_db.list_trainers()
        return JSONResponse(content={"success": True, "trainers": trainers})
    
    # 评分标准（培训师/管理员可读可改，权重总和须为 1.0）
    _scoring_rules_paths = scoring_rules_paths(project_root)
    _overall_weight_labels = OVERALL_WEIGHT_LABELS
    
    @app.get("/api/scoring-rules")
    async def api_scoring_rules_get(request: Request, stage: str = Query("stretch", description="stretch|boiling")):
        u = auth_current_user(request)
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
    
    @app.put("/api/scoring-rules")
    async def api_scoring_rules_put(request: Request):
        u = auth_current_user(request)
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
