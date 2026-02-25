# 将项目备份到 GitHub 指南

按下面步骤操作，即可把项目实时保存到 GitHub，防止丢失。

---

## 一、安装 Git（若尚未安装）

1. 打开 **https://git-scm.com/download/win** 下载 Windows 版 Git。
2. 运行安装程序，按默认选项下一步即可（可勾选 “Add Git to PATH”）。
3. 安装完成后**关闭并重新打开**终端（或 Cursor），再执行下面的命令。

验证是否安装成功（在项目目录下打开终端执行）：

```bash
git --version
```

若显示版本号（如 `git version 2.43.0`）即表示成功。

---

## 二、在项目目录初始化 Git 并首次提交

在 **PowerShell** 或 **命令提示符** 中执行（请把路径改成你的实际项目路径）：

```powershell
cd C:\Users\86187\Desktop\ramen-qc-vision

# 1. 初始化仓库
git init

# 2. 添加所有文件（.gitignore 里排除的不会加入）
git add .

# 3. 第一次提交
git commit -m "初始提交：拉面质检视觉项目"
```

---

## 三、在 GitHub 上创建仓库

1. 登录 **https://github.com**。
2. 右上角点击 **“+”** → **“New repository”**。
3. 填写：
   - **Repository name**：例如 `ramen-qc-vision`
   - **Description**（可选）：例如 “拉面质检视觉评分系统”
   - 选择 **Private**（私有，仅自己可见）或 **Public**（公开）。
   - **不要**勾选 “Add a README file”等（仓库保持空）。
4. 点击 **“Create repository”**。
5. 创建完成后，页面上会显示仓库地址，形如：  
   `https://github.com/你的用户名/ramen-qc-vision.git`  
   记下这个地址，下面会用到。

---

## 四、把本地项目推送到 GitHub

在**同一终端、项目目录**下执行（把 `你的用户名/ramen-qc-vision` 换成你的仓库地址）：

```powershell
# 添加远程仓库（替换成你的 GitHub 仓库地址）
git remote add origin https://github.com/你的用户名/ramen-qc-vision.git

# 推送（首次推送并设置上游分支）
git branch -M main
git push -u origin main
```

首次推送时，浏览器可能会弹出 GitHub 登录或授权，按提示完成即可。  
若提示输入用户名/密码，建议在 GitHub 网站 → Settings → Developer settings → Personal access tokens 中生成一个 **Token**，用 Token 代替密码。

---

## 五、之后的“实时保存”（日常使用）

每次改完代码或配置，在项目目录执行：

```powershell
git add .
git commit -m "简短说明你改了什么"
git push
```

也可以只在重要节点执行（例如完成一个功能再 commit + push），GitHub 上会保留所有历史版本。

---

## 六、说明：哪些文件会被提交、哪些不会

- **会提交**：代码（`src/`、`web/`、`scripts/` 等）、配置（如 `configs/` 下非密钥文件）、文档、`data/scores` 下规则与标注、`data/labels` 等。
- **不会提交**（已写在 `.gitignore`）：  
  虚拟环境 `.venv/`、模型权重 `models/*.pt`、大体积数据（如 `data/processed_videos/`、`data/ivcam/`、`datasets/` 等）。

若你希望把某类大文件也备份，可编辑项目根目录的 **`.gitignore`**，注释掉或删除对应规则后再 `git add` / `git commit` / `git push`。  
若数据量特别大，可考虑用 **Git LFS** 或网盘单独备份数据。

---

## 常见问题

- **提示 “git 不是内部或外部命令”**：说明 Git 未安装或未加入 PATH，请重新安装并勾选 “Add Git to PATH”，然后重启终端。
- **推送时要求登录**：使用 GitHub 用户名 + Personal access token（在 GitHub 设置里生成）作为密码。
- **想换一台电脑继续开发**：在新电脑安装 Git 后执行 `git clone https://github.com/你的用户名/ramen-qc-vision.git`，即可拉取完整项目与历史。

按上述步骤做完后，项目就会在 GitHub 上有一份完整备份，之后用 `git add` + `git commit` + `git push` 即可持续“实时保存”。
