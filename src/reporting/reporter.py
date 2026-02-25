from typing import Any, Dict
import json
import os


def generate_report(results: Dict[str, Any], cfg: Dict[str, Any], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 生成带视频播放的详细报告
    html = """
    <html>
    <head>
        <title>Ramen QC Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                color: #2c3e50;
            }
            .video-section {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .video-wrapper {
                position: relative;
                width: 100%;
                aspect-ratio: 16/9;
                margin-bottom: 10px;
            }
            video {
                width: 100%;
                height: 100%;
                border-radius: 4px;
            }
            .analysis-panel {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
            }
            .frame-info {
                margin-bottom: 15px;
                padding: 10px;
                background-color: white;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }
            .detections {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 10px;
            }
            .detection-item {
                background-color: #e3f2fd;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 14px;
            }
            .score-section {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .score-card {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .score-label {
                font-size: 18px;
                font-weight: bold;
            }
            .score-value {
                font-size: 36px;
                font-weight: bold;
                color: #2ecc71;
            }
            .score-value.low {
                color: #e74c3c;
            }
            .violations {
                margin-top: 20px;
                padding: 15px;
                background-color: #ffebee;
                border-radius: 4px;
                border-left: 4px solid #e74c3c;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>兰州拉面标准化视觉质检报告</h1>
            
            <div class="score-section">
                <div class="score-card">
                    <div class="score-label">最终评分</div>
                    <div class="score-value {{ 'low' if summary.final_score < 0.6 else '' }}">
                        {{ summary.final_score }}
                    </div>
                </div>
                
                <div class="violations">
                    <h3>违规项</h3>
                    {% if events|length > 0 %}
                        {% for event in events %}
                            {% if event.violations|length > 0 %}
                                <ul>
                                    {% for violation in event.violations %}
                                        <li>{{ violation.feature }}: 当前值 {{ violation.value|round(4) }}, 基准值 {{ violation.baseline|round(4) }}, 得分 {{ violation.score|round(4) }}</li>
                                    {% endfor %}
                                </ul>
                            {% endif %}
                        {% endfor %}
                    {% else %}
                        <p>未发现明显违规</p>
                    {% endif %}
                </div>
            </div>
            
            <div class="video-section">
                <h2>视频分析</h2>
                <div class="video-wrapper">
                    <video id="videoPlayer" controls>
                        <source src="{{ video_path }}" type="video/mp4">
                        您的浏览器不支持HTML5视频
                    </video>
                </div>
                
                <div class="analysis-panel">
                    <h3>实时分析结果</h3>
                    <div class="frame-info">
                        <strong>当前帧:</strong> <span id="currentFrame">0</span>
                        <div class="detections" id="currentDetections">
                            播放视频查看检测结果
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // 视频分析数据
            const analysisData = {{ analysis_data|tojson }};
            
            // 获取DOM元素
            const videoPlayer = document.getElementById('videoPlayer');
            const currentFrameSpan = document.getElementById('currentFrame');
            const currentDetectionsDiv = document.getElementById('currentDetections');
            
            // 视频帧速率
            const fps = {{ fps }};
            
            // 当视频播放时，更新分析结果
            videoPlayer.addEventListener('timeupdate', function() {
                const currentTime = videoPlayer.currentTime;
                const currentFrame = Math.floor(currentTime * fps);
                
                currentFrameSpan.textContent = currentFrame;
                
                // 获取当前帧的分析结果
                const frameData = analysisData[currentFrame];
                if (frameData) {
                    // 显示检测结果
                    let detectionsHTML = '';
                    if (frameData.detections && frameData.detections.length > 0) {
                        frameData.detections.forEach(detection => {
                            detectionsHTML += '<div class="detection-item">' + 
                                detection.class + ' (' + detection.conf.toFixed(2) + ')' + 
                                '</div>';
                        });
                    } else {
                        detectionsHTML = '<div class="detection-item">未检测到目标</div>';
                    }
                    currentDetectionsDiv.innerHTML = detectionsHTML;
                } else {
                    currentDetectionsDiv.innerHTML = '<div class="detection-item">无分析数据</div>';
                }
            });
        </script>
    </body>
    </html>
    """
    
    # 生成分析数据（简化版，包含每一帧的检测结果）
    analysis_data = []
    fps = cfg.get("input", {}).get("frame_rate", 15)
    
    # 获取检测结果
    if results.get("events"):
        for event in results["events"]:
            if "detections" in event:
                # 每个event.detections是一个列表，每个元素是一帧的检测结果
                for frame_dets in event["detections"]:
                    # 为每一帧创建一个包含detections字段的对象
                    analysis_data.append({
                        "detections": frame_dets
                    })
    
    # 渲染HTML报告
    from jinja2 import Template
    template = Template(html)
    rendered_html = template.render(
        summary=results.get("summary", {}),
        events=results.get("events", []),
        analysis_data=analysis_data,
        fps=fps,
        video_path=""
    )
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)
    
    # 保存JSON报告
    json_path = out_path.replace(".html", ".json")
    if cfg.get("report", {}).get("save_json", True):
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(results, jf, ensure_ascii=False, indent=2)
