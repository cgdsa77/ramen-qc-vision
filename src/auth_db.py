# -*- coding: utf-8 -*-
"""
毕设用户权限：SQLite（默认）或 MySQL 用户表 + 内存 Session。
用户表字段：id, username, password(MD5), role(0管理员/1培训师/2厨师), name, create_time, status(0禁用/1正常)
配置 MySQL：复制 configs/db_config.example.json 为 configs/db_config.json，填写连接信息；未配置则使用 SQLite。
"""
import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

_UNSET = object()  # 表示「未传该字段」

# 项目根目录
_project_root = Path(__file__).resolve().parents[1]
_db_path = _project_root / "data" / "ramen_auth.db"
_db_path.parent.mkdir(parents=True, exist_ok=True)

# 内存 Session：session_id -> { user_id, username, role, name, expire_at }
_sessions: Dict[str, Dict[str, Any]] = {}
_SESSION_TTL = 24 * 3600  # 24 小时

ROLE_ADMIN = 0
ROLE_TRAINER = 1
ROLE_CHEF = 2

# 数据库模式：None=未初始化，'sqlite' | 'mysql'
_db_mode: Optional[str] = None
_mysql_config: Optional[Dict[str, Any]] = None


def _load_mysql_config() -> Optional[Dict[str, Any]]:
    """若存在 configs/db_config.json 且含 mysql 配置则返回，否则 None。"""
    global _mysql_config
    if _mysql_config is not None:
        return _mysql_config
    cfg_path = _project_root / "configs" / "db_config.json"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("mysql"), dict) and data["mysql"].get("database"):
            _mysql_config = data["mysql"]
            return _mysql_config
    except Exception:
        pass
    return None


def _resolve_db_mode() -> str:
    global _db_mode
    if _db_mode is not None:
        return _db_mode
    _db_mode = "mysql" if _load_mysql_config() else "sqlite"
    return _db_mode


def _md5(s: str) -> str:
    return hashlib.md5((s or "").strip().encode("utf-8")).hexdigest().lower()


# ---------- SQLite ----------
def _get_conn_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    return conn


# ---------- MySQL ----------
def _get_conn_mysql():
    import pymysql
    from pymysql.cursors import DictCursor
    cfg = _load_mysql_config()
    if not cfg:
        raise RuntimeError("MySQL 未配置：请创建 configs/db_config.json 并填写 mysql 连接信息")
    return pymysql.connect(
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 3306)),
        user=cfg.get("user", "root"),
        password=cfg.get("password", ""),
        database=cfg.get("database", "ramen_qc"),
        charset=cfg.get("charset", "utf8mb4"),
        cursorclass=DictCursor,
    )


def _get_conn():
    """统一入口：返回 SQLite 或 MySQL 连接。MySQL 使用 DictCursor，行为与 SQLite Row 一致（按列名访问）。"""
    if _resolve_db_mode() == "mysql":
        return _get_conn_mysql()
    return _get_conn_sqlite()


def _param_style() -> str:
    """占位符：SQLite 用 ?，MySQL 用 %s。"""
    return "?" if _resolve_db_mode() == "sqlite" else "%s"


def _row_to_user(r: Any) -> Dict[str, Any]:
    """从 SQLite Row 或 MySQL Dict 转为统一 dict。"""
    return {
        "id": r["id"],
        "username": r["username"],
        "role": int(r["role"]),
        "name": (r["name"] or ""),
        "status": int(r["status"]),
    }


def init_db() -> None:
    """创建用户表并插入默认管理员（若表为空）。SQLite 与 MySQL 分别执行对应 DDL。"""
    mode = _resolve_db_mode()
    conn = _get_conn()
    try:
        if mode == "sqlite":
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL,
                    role INTEGER NOT NULL DEFAULT 2,
                    name TEXT NOT NULL DEFAULT '',
                    create_time TEXT NOT NULL,
                    status INTEGER NOT NULL DEFAULT 1,
                    assigned_trainer_id INTEGER NULL
                )
            """)
            conn.commit()
            cur = conn.execute("SELECT COUNT(*) FROM users")
            count = cur.fetchone()[0]
        else:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(64) NOT NULL UNIQUE,
                        password VARCHAR(64) NOT NULL,
                        role TINYINT NOT NULL DEFAULT 2,
                        name VARCHAR(64) NOT NULL DEFAULT '',
                        create_time VARCHAR(32) NOT NULL,
                        status TINYINT NOT NULL DEFAULT 1,
                        assigned_trainer_id INT NULL COMMENT '所属培训师用户id'
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                cur.execute("SELECT COUNT(*) AS c FROM users")
                count = cur.fetchone()["c"]
            conn.commit()
        if count == 0:
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            p = _param_style()
            if mode == "sqlite":
                conn.execute(
                    "INSERT INTO users (username, password, role, name, create_time, status) VALUES (" + ", ".join([p] * 6) + ")",
                    ("admin", _md5("admin123"), ROLE_ADMIN, "管理员", now, 1),
                )
                conn.commit()
            else:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (username, password, role, name, create_time, status) VALUES (" + ", ".join([p] * 6) + ")",
                        ("admin", _md5("admin123"), ROLE_ADMIN, "管理员", now, 1),
                    )
                conn.commit()
    finally:
        conn.close()
    _ensure_assigned_trainer_id()


def _ensure_assigned_trainer_id() -> None:
    """为 users 表增加 assigned_trainer_id 列（若不存在），用于培训师–学员关系。"""
    mode = _resolve_db_mode()
    conn = _get_conn()
    try:
        if mode == "sqlite":
            cur = conn.execute("PRAGMA table_info(users)")
            cols = [row[1] for row in cur.fetchall()]
            if "assigned_trainer_id" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN assigned_trainer_id INTEGER NULL")
                conn.commit()
        else:
            with conn.cursor() as cur:
                cur.execute("SHOW COLUMNS FROM users LIKE 'assigned_trainer_id'")
                if cur.fetchone() is None:
                    cur.execute("ALTER TABLE users ADD COLUMN assigned_trainer_id INT NULL COMMENT '所属培训师用户id'")
                    conn.commit()
    finally:
        conn.close()


def login(username: str, password: str) -> Tuple[Optional[Dict], Optional[str]]:
    pw_hash = _md5(password)
    conn = _get_conn()
    mode = _resolve_db_mode()
    p = _param_style()
    try:
        if mode == "sqlite":
            row = conn.execute(
                "SELECT id, username, role, name, status FROM users WHERE username = " + p + " AND password = " + p,
                (username.strip(), pw_hash),
            ).fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, role, name, status FROM users WHERE username = " + p + " AND password = " + p,
                    (username.strip(), pw_hash),
                )
                row = cur.fetchone()
        if not row:
            return (None, None)
        if row["status"] != 1:
            return ({"_disabled": True}, None)
        user = _row_to_user(row)
        user["name"] = (row.get("name") or "").strip()
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "name": user["name"],
            "expire_at": time.time() + _SESSION_TTL,
        }
        return (user, session_id)
    finally:
        conn.close()


def get_session(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None
    s = _sessions.get(session_id)
    if not s or s["expire_at"] < time.time():
        if s:
            del _sessions[session_id]
        return None
    return {
        "id": s["user_id"],
        "username": s["username"],
        "role": s["role"],
        "name": s["name"],
    }


def logout(session_id: Optional[str]) -> None:
    if session_id and session_id in _sessions:
        del _sessions[session_id]


def list_users() -> List[Dict[str, Any]]:
    conn = _get_conn()
    mode = _resolve_db_mode()
    try:
        if mode == "sqlite":
            rows = conn.execute(
                "SELECT id, username, role, name, create_time, status, assigned_trainer_id FROM users ORDER BY id"
            ).fetchall()
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, role, name, create_time, status, assigned_trainer_id FROM users ORDER BY id"
                )
                rows = cur.fetchall()
        out = []
        for r in rows:
            row = {
                "id": r["id"],
                "username": r["username"],
                "role": int(r["role"]),
                "name": r["name"] or "",
                "create_time": r["create_time"] or "",
                "status": int(r["status"]),
            }
            if "assigned_trainer_id" in r and r["assigned_trainer_id"] is not None:
                row["assigned_trainer_id"] = int(r["assigned_trainer_id"])
            else:
                row["assigned_trainer_id"] = None
            out.append(row)
        return out
    finally:
        conn.close()


def list_trainers() -> List[Dict[str, Any]]:
    """返回所有培训师（role=1），用于下拉选择所属培训师。"""
    return [u for u in list_users() if u["role"] == ROLE_TRAINER]


def get_students_by_trainer(trainer_id: int) -> List[Dict[str, Any]]:
    """返回指定培训师名下的学员（role=2 且 assigned_trainer_id=trainer_id）。"""
    conn = _get_conn()
    mode = _resolve_db_mode()
    p = _param_style()
    try:
        if mode == "sqlite":
            rows = conn.execute(
                "SELECT id, username, role, name, create_time, status FROM users WHERE role = 2 AND assigned_trainer_id = " + p,
                (trainer_id,),
            ).fetchall()
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, role, name, create_time, status FROM users WHERE role = 2 AND assigned_trainer_id = " + p,
                    (trainer_id,),
                )
                rows = cur.fetchall()
        return [
            {
                "id": r["id"],
                "username": r["username"],
                "role": int(r["role"]),
                "name": r["name"] or "",
                "create_time": r["create_time"] or "",
                "status": int(r["status"]),
            }
            for r in rows
        ]
    finally:
        conn.close()


def create_user(
    username: str,
    password: str,
    role: int,
    name: str,
    assigned_trainer_id: Optional[int] = None,
) -> Tuple[bool, str, Optional[Dict]]:
    username = (username or "").strip()
    name = (name or "").strip()
    if not username:
        return (False, "账号不能为空", None)
    if not password:
        return (False, "初始密码不能为空", None)
    if role not in (0, 1, 2):
        return (False, "角色无效", None)
    conn = _get_conn()
    mode = _resolve_db_mode()
    p = _param_style()
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        if mode == "sqlite":
            conn.execute(
                "INSERT INTO users (username, password, role, name, create_time, status, assigned_trainer_id) VALUES ("
                + ", ".join([p] * 7)
                + ")",
                (username, _md5(password), role, name, now, 1, assigned_trainer_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id, username, role, name, create_time, status, assigned_trainer_id FROM users WHERE username = " + p,
                (username,),
            ).fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, password, role, name, create_time, status, assigned_trainer_id) VALUES ("
                    + ", ".join([p] * 7)
                    + ")",
                    (username, _md5(password), role, name, now, 1, assigned_trainer_id),
                )
                conn.commit()
                cur.execute(
                    "SELECT id, username, role, name, create_time, status, assigned_trainer_id FROM users WHERE username = " + p,
                    (username,),
                )
                row = cur.fetchone()
        out = {
            "id": row["id"],
            "username": row["username"],
            "role": int(row["role"]),
            "name": row["name"] or "",
            "create_time": row["create_time"] or "",
            "status": int(row["status"]),
        }
        out["assigned_trainer_id"] = int(row["assigned_trainer_id"]) if row.get("assigned_trainer_id") is not None else None
        return (True, "创建成功", out)
    except Exception as e:
        err_msg = str(e).lower()
        if "unique" in err_msg or "duplicate" in err_msg or "1062" in err_msg:
            return (False, "账号已存在", None)
        raise
    finally:
        conn.close()


def update_user(
    uid: int,
    role: Optional[int] = None,
    name: Optional[str] = None,
    status: Optional[int] = None,
    assigned_trainer_id: Any = _UNSET,
) -> Tuple[bool, str]:
    conn = _get_conn()
    mode = _resolve_db_mode()
    p = _param_style()
    updates = []
    args = []
    if role is not None:
        if role not in (0, 1, 2):
            return (False, "角色无效")
        updates.append("role = " + p)
        args.append(role)
    if name is not None:
        updates.append("name = " + p)
        args.append((name or "").strip())
    if status is not None:
        if status not in (0, 1):
            return (False, "状态无效")
        updates.append("status = " + p)
        args.append(status)
    if assigned_trainer_id is not _UNSET:
        updates.append("assigned_trainer_id = " + p)
        args.append(int(assigned_trainer_id) if assigned_trainer_id else None)
    if not updates:
        return (True, "无变更")
    args.append(uid)
    try:
        if mode == "sqlite":
            conn.execute("UPDATE users SET " + ", ".join(updates) + " WHERE id = " + p, args)
            conn.commit()
        else:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET " + ", ".join(updates) + " WHERE id = " + p, args)
            conn.commit()
        return (True, "更新成功")
    finally:
        conn.close()


def delete_user(uid: int) -> Tuple[bool, str]:
    conn = _get_conn()
    mode = _resolve_db_mode()
    p = _param_style()
    try:
        if mode == "sqlite":
            cur = conn.execute("SELECT role FROM users WHERE id = " + p, (uid,))
            row = cur.fetchone()
            if not row:
                return (False, "用户不存在")
            if row["role"] == ROLE_ADMIN:
                admins = conn.execute("SELECT COUNT(*) FROM users WHERE role = " + p, (ROLE_ADMIN,)).fetchone()[0]
                if admins <= 1:
                    return (False, "不能删除唯一管理员")
            conn.execute("DELETE FROM users WHERE id = " + p, (uid,))
            conn.commit()
        else:
            with conn.cursor() as cur:
                cur.execute("SELECT role FROM users WHERE id = " + p, (uid,))
                row = cur.fetchone()
                if not row:
                    return (False, "用户不存在")
                if row["role"] == ROLE_ADMIN:
                    cur.execute("SELECT COUNT(*) AS c FROM users WHERE role = " + p, (ROLE_ADMIN,))
                    if cur.fetchone()["c"] <= 1:
                        return (False, "不能删除唯一管理员")
                cur.execute("DELETE FROM users WHERE id = " + p, (uid,))
            conn.commit()
        return (True, "删除成功")
    finally:
        conn.close()


def reset_password(uid: int, new_password: str) -> Tuple[bool, str]:
    new_password = (new_password or "").strip()
    if not new_password:
        return (False, "新密码不能为空")
    conn = _get_conn()
    mode = _resolve_db_mode()
    p = _param_style()
    try:
        if mode == "sqlite":
            cur = conn.execute("UPDATE users SET password = " + p + " WHERE id = " + p, (_md5(new_password), uid))
            conn.commit()
            rowcount = cur.rowcount
        else:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET password = " + p + " WHERE id = " + p, (_md5(new_password), uid))
                rowcount = cur.rowcount
            conn.commit()
        if rowcount == 0:
            return (False, "用户不存在")
        return (True, "密码已重置")
    finally:
        conn.close()
