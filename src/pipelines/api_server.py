from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path
from src.utils.config import load_config
from src.pipelines.steps import build_pipeline
from src.reporting.reporter import generate_report
from src.api.video_detection_api import get_detector


app = FastAPI(title="Ramen QC API")
cfg = None
pipeline = None
detector = None
project_root = Path(__file__).parent.parent.parent

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 添加静态文件服务
app.mount("/reports", StaticFiles(directory=str(project_root / "reports")), name="reports")

# 静态文件目录（Web界面）
web_dir = project_root / "web"
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

# 根路径返回检测界面
@app.get("/")
async def root():
    web_file = web_dir / "video_detection.html"
    if web_file.exists():
        return FileResponse(str(web_file))
    return {"message": "Ramen QC System API", "docs": "/docs"}


@app.on_event("startup")
async def startup_event():
    global cfg, pipeline, detector
    if hasattr(app.state, 'config_path'):
        cfg = load_config(app.state.config_path)
        pipeline = build_pipeline(cfg)
    
    # 初始化检测器
    try:
        detector = get_detector()
    except Exception as e:
        print(f"[WARN] 检测器初始化失败: {e}")
        detector = None


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    results = pipeline.run(tmp_path)
    report_filename = f"report_{os.path.basename(tmp_path)}.html"
    report_path = os.path.join("reports", report_filename)
    generate_report(results, cfg, report_path)
    return {"report": f"/reports/{report_filename}", "summary": results.get("summary", {})}


@app.post("/api/detect_video")
async def detect_video(file: UploadFile = File(...)):
    """
    检测视频中的目标（手、面条等），返回每一帧的检测结果
    """
    global detector
    
    if detector is None:
        detector = get_detector()
    
    if detector.model is None:
        return {
            "success": False,
            "error": "检测模型未加载。请先运行训练脚本训练模型。"
        }
    
    try:
        result = await detector.process_uploaded_video(file, conf_threshold=0.25)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def run_api(host: str = "0.0.0.0", port: int = 8000, config_path: str = None):
    """
    运行API服务器
    
    Args:
        host: 主机地址
        port: 端口号
        config_path: 配置文件路径（可选）
    """
    if config_path:
        app.state.config_path = config_path
    else:
        # 使用默认配置
        default_config = project_root / "configs" / "default.yaml"
        if default_config.exists():
            app.state.config_path = str(default_config)
    
    print(f"\n{'='*60}")
    print(f"启动Ramen QC检测系统")
    print(f"{'='*60}")
    print(f"服务器地址: http://{host}:{port}")
    print(f"Web界面: http://{host}:{port}/")
    print(f"API文档: http://{host}:{port}/docs")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=host, port=port)
