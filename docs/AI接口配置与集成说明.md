# AI 接口配置与集成说明

本文档说明如何让 AI 分析功能**一次配置、持续可用**，以及如何在其他界面中复用同一套 AI 接口。

---

## 一、一次配置、持续可用

AI 功能**不依赖每次手动配置**，只要按下面做一次即可长期使用。

### 1. 配置文件（唯一需要配置的地方）

- **密钥文件**：`configs/ai_api_secret.json`  
  - 格式：`{ "api_key": "您的七牛云 AI API Key（sk- 开头）" }`  
  - 可从 `configs/ai_api_secret.example.json` 复制后改名并填入真实 key。  
- **可选**：`configs/ai_api.yaml` 中可修改 `base_url`、`model`（默认七牛云 `deepseek-v3`）。  
- **可选**：环境变量 `AI_API_KEY` 或 `OPENAI_API_KEY` 也可作为 api_key 来源（优先级在密钥文件之后）。

服务启动时会从上述位置加载配置；**只要该文件存在且 api_key 有效，之后每次打开页面都无需再配置**。

### 2. 启动方式

- 在项目根目录执行：`python start_web.py`  
- 启动日志中若出现「已从 … 加载 api_key，AI 分析可用」，即表示配置成功，刷新或新开页面即可使用 AI。

### 3. 状态排查

- 接口：`GET /api/ai-config-status`  
- 返回示例：`{ "api_key_loaded": true, "config_path": "...", "config_exists": true }`  
- 用于确认当前服务是否已正确加载密钥（不返回密钥内容）。

---

## 二、与其他界面结合：API 约定

若要把「调用 AI 的界面」与其它页面（如其他评分页、报表页）结合，只需让前端调用下面两个接口，**无需再为 AI 单独做配置**（仍只需保证 `configs/ai_api_secret.json` 正确且服务由 `start_web.py` 启动）。

### 1. 开始分析（一键生成完整报告）

- **接口**：`POST /api/ai-analyze-scoring`  
- **请求体**：`{ "video": "视频名称", "data": { 当前视频的评分摘要 } }`  
- **说明**：不依赖用户输入问题，后端会用固定分析提示词生成「整体评估 + 各维度分析 + 改进建议」等完整报告。  
- **前端用法**：对应界面上「开始分析」按钮：用户无需输入任何文字，点击即请求此接口并展示返回的 `analysis` 文本。

### 2. 发送（针对用户问题的对话）

- **接口**：`POST /api/ai-chat`  
- **请求体**：`{ "video": "视频名称", "data": { 评分摘要 }, "question": "用户输入的问题", "history": [ 之前的对话 { "role": "user"|"assistant", "content": "..." } ] }`  
- **说明**：根据用户当前问题 + 当前视频的评分数据 + 可选历史对话，返回针对该问题的回答。  
- **前端用法**：对应界面上「发送」按钮：用户输入问题后点击，把 `question` 和 `history` 传给此接口，展示返回的 `answer`。

### 3. 两个按钮的区别（小结）

| 能力       | 开始分析                     | 发送                           |
|------------|------------------------------|--------------------------------|
| 是否必须输入问题 | 否，可不输入直接点           | 是，需在输入框输入问题后点击   |
| 后端接口   | `POST /api/ai-analyze-scoring` | `POST /api/ai-chat`            |
| 典型用途   | 一键生成当前视频的完整分析报告 | 针对具体问题多轮问答           |

---

## 三、前端集成示例（供其他界面参考）

```javascript
// 开始分析（无需用户输入问题）
const res1 = await fetch('/api/ai-analyze-scoring', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ video: currentVideoName, data: summaryData })
});
const { analysis } = await res1.json();
// 将 analysis 展示在页面

// 发送（用户提问）
const res2 = await fetch('/api/ai-chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    video: currentVideoName,
    data: summaryData,
    question: userInput.trim(),
    history: chatHistory  // 可选，多轮对话时传入
  })
});
const { answer } = await res2.json();
// 将 answer 展示，并可选追加到 chatHistory
```

只要同源且由同一 `start_web.py` 服务提供页面与 API，上述调用即可用；**无需在每个页面单独配置 API Key**，密钥仅在服务端 `configs/ai_api_secret.json`（或环境变量）中维护即可。

---

## 四、常见问题

- **每次进界面都要重新配置？**  
  - 不需要。确认 `configs/ai_api_secret.json` 存在且 `api_key` 正确，用 `python start_web.py` 启动一次即可；之后只要服务在运行，所有使用该服务的页面都会自动具备 AI 能力。

- **和其他界面结合时要再配一遍 key 吗？**  
  - 不需要。其他界面只需调用 `POST /api/ai-analyze-scoring` 和 `POST /api/ai-chat`，AI 密钥只在当前后端配置一次。

- **如何确认 AI 是否可用？**  
  - 调用 `GET /api/ai-config-status` 查看 `api_key_loaded`；或在当前评分可视化页面点击「开始分析」/「发送」看是否有正常返回。
