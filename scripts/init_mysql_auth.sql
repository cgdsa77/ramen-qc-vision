-- 毕设用户权限：MySQL 建库与用户表
-- 使用方式：安装 MySQL 后，在命令行或 MySQL 客户端执行本文件，例如：
--   mysql -u root -p < scripts/init_mysql_auth.sql
-- 或登录 MySQL 后： source 本文件路径;

CREATE DATABASE IF NOT EXISTS ramen_qc DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE ramen_qc;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE COMMENT '登录账号',
    password VARCHAR(64) NOT NULL COMMENT 'MD5 加密',
    role TINYINT NOT NULL DEFAULT 2 COMMENT '0=管理员 1=培训师 2=厨师/学员',
    name VARCHAR(64) NOT NULL DEFAULT '' COMMENT '真实姓名',
    create_time VARCHAR(32) NOT NULL COMMENT '创建时间',
    status TINYINT NOT NULL DEFAULT 1 COMMENT '0=禁用 1=正常',
    assigned_trainer_id INT NULL COMMENT '所属培训师用户id'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';

-- 若表为空，插入默认管理员（密码为 admin123 的 MD5）
INSERT INTO users (username, password, role, name, create_time, status)
SELECT 'admin', '0192023a7bbd73250516f069df18b500', 0, '管理员', DATE_FORMAT(NOW(), '%Y-%m-%d %H:%M:%S'), 1
FROM DUAL
WHERE NOT EXISTS (SELECT 1 FROM users LIMIT 1);
