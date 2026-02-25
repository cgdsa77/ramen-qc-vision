"""
可视化姿态估计结果
在视频帧上绘制姿态关键点和骨架（类似图中的可视化效果）
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.pose import PoseEstimator
except ImportError:
    print("[错误] 无法导入PoseEstimator")
    sys.exit(1)


def load_config() -> Dict[str, Any]:
    """加载配置"""
    import yaml
    config_path = project_root / "configs" / "default.yaml"
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(config_path, 'r', encoding='gbk', errors='ignore') as f:
                config = yaml.safe_load(f)
    else:
        config = {
            "models": {
                "pose": {
                    "backend": "mediapipe"
                }
            }
        }
    
    return config


def draw_pose_skeleton(frame: np.ndarray, keypoints: List[Dict[str, Any]], 
                     colors: Dict[str, tuple] = None) -> np.ndarray:
    """
    在帧上绘制姿态骨架（类似图中的效果）
    
    Args:
        frame: 原始帧（BGR格式）
        keypoints: 关键点列表
        colors: 颜色配置字典
        
    Returns:
        绘制了骨架的帧
    """
    if not keypoints:
        return frame
    
    if colors is None:
        # 定义颜色配置（类似图中的颜色）
        colors = {
            'head': (0, 255, 0),      # 绿色 - 头部
            'torso': (255, 0, 255),   # 洋红色 - 躯干
            'left_arm': (255, 0, 0),  # 蓝色 - 左臂
            'right_arm': (255, 0, 0), # 蓝色 - 右臂
            'left_leg': (0, 165, 255), # 橙色 - 左腿
            'right_leg': (0, 165, 255), # 橙色 - 右腿
            'keypoint': (0, 0, 255),   # 红色 - 关键点
        }
    
    result_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # 建立关键点名称到索引的映射
    kp_dict = {kp['name']: kp for kp in keypoints}
    
    # 定义连接关系（基于MediaPipe关键点）
    connections = [
        # 头部（绿色）
        ('nose', 'left_eye_inner', colors['head']),
        ('left_eye_inner', 'left_eye', colors['head']),
        ('left_eye', 'left_eye_outer', colors['head']),
        ('left_eye_outer', 'left_ear', colors['head']),
        ('nose', 'right_eye_inner', colors['head']),
        ('right_eye_inner', 'right_eye', colors['head']),
        ('right_eye', 'right_eye_outer', colors['head']),
        ('right_eye_outer', 'right_ear', colors['head']),
        ('left_ear', 'right_ear', colors['head']),
        
        # 躯干（洋红色）
        ('left_shoulder', 'right_shoulder', colors['torso']),
        ('left_shoulder', 'left_hip', colors['torso']),
        ('right_shoulder', 'right_hip', colors['torso']),
        ('left_hip', 'right_hip', colors['torso']),
        
        # 左臂（蓝色）
        ('left_shoulder', 'left_elbow', colors['left_arm']),
        ('left_elbow', 'left_wrist', colors['left_arm']),
        ('left_wrist', 'left_index', colors['left_arm']),
        ('left_wrist', 'left_pinky', colors['left_arm']),
        ('left_index', 'left_thumb', colors['left_arm']),
        ('left_pinky', 'left_thumb', colors['left_arm']),
        
        # 右臂（蓝色）
        ('right_shoulder', 'right_elbow', colors['right_arm']),
        ('right_elbow', 'right_wrist', colors['right_arm']),
        ('right_wrist', 'right_index', colors['right_arm']),
        ('right_wrist', 'right_pinky', colors['right_arm']),
        ('right_index', 'right_thumb', colors['right_arm']),
        ('right_pinky', 'right_thumb', colors['right_arm']),
        
        # 左腿（橙色）
        ('left_hip', 'left_knee', colors['left_leg']),
        ('left_knee', 'left_ankle', colors['left_leg']),
        ('left_ankle', 'left_heel', colors['left_leg']),
        ('left_ankle', 'left_foot_index', colors['left_leg']),
        ('left_heel', 'left_foot_index', colors['left_leg']),
        
        # 右腿（橙色）
        ('right_hip', 'right_knee', colors['right_leg']),
        ('right_knee', 'right_ankle', colors['right_leg']),
        ('right_ankle', 'right_heel', colors['right_leg']),
        ('right_ankle', 'right_foot_index', colors['right_leg']),
        ('right_heel', 'right_foot_index', colors['right_leg']),
    ]
    
    # 绘制连接线（骨架）
    for start_name, end_name, color in connections:
        if start_name in kp_dict and end_name in kp_dict:
            start_kp = kp_dict[start_name]
            end_kp = kp_dict[end_name]
            
            # 只绘制置信度足够高的连接
            if (start_kp.get('confidence', 0) > 0.3 and 
                end_kp.get('confidence', 0) > 0.3):
                
                pt1 = (int(start_kp['x']), int(start_kp['y']))
                pt2 = (int(end_kp['x']), int(end_kp['y']))
                cv2.line(result_frame, pt1, pt2, color, 2)
    
    # 绘制关键点（节点）
    for kp in keypoints:
        confidence = kp.get('confidence', 0)
        if confidence > 0.3:  # 只绘制置信度足够高的点
            x, y = int(kp['x']), int(kp['y'])
            # 绘制关键点（红色圆圈）
            cv2.circle(result_frame, (x, y), 5, colors['keypoint'], -1)
            # 可选：显示关键点名称（注释掉以避免画面太乱）
            # cv2.putText(result_frame, kp['name'], (x+5, y-5), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return result_frame


def safe_imwrite(path: Path, image: np.ndarray) -> bool:
    """兼容中文路径的安全写图函数"""
    try:
        ok, buf = cv2.imencode('.jpg', image)
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def visualize_video_pose(video_path: Path, pose_estimator: PoseEstimator,
                         output_dir: Path, sample_count: int = 10):
    """
    可视化视频中的姿态估计结果
    
    Args:
        video_path: 视频文件路径
        pose_estimator: 姿态估计器
        output_dir: 输出目录
        sample_count: 采样帧数（提取多少个帧进行可视化）
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[警告] 无法打开视频: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // sample_count)
    
    video_name = video_path.stem
    print(f"\n处理视频: {video_name}")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 采样间隔: 每 {frame_interval} 帧提取一次")
    print(f"  - 将生成 {sample_count} 个可视化帧")
    
    frame_count = 0
    saved_count = 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    while saved_count < sample_count:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按采样间隔提取
        if frame_count % frame_interval == 0:
            # 提取关键点
            keypoints = pose_estimator.run_frame(frame)
            
            # 绘制姿态骨架
            frame_with_pose = draw_pose_skeleton(frame.copy(), keypoints)
            
            # 添加信息文本
            info_text = [
                f"Frame: {frame_count}/{total_frames}",
                f"Keypoints: {len(keypoints)}",
                f"Time: {frame_count/fps:.2f}s" if fps > 0 else "Time: N/A"
            ]
            y_offset = 30
            for text in info_text:
                cv2.putText(frame_with_pose, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 25
            
            # 保存结果
            output_file = output_dir / f"{video_name}_frame_{frame_count:06d}_pose.jpg"
            if safe_imwrite(output_file, frame_with_pose):
                saved_count += 1
            else:
                print(f"[警告] 保存失败: {output_file}")
            
            if saved_count % 5 == 0:
                print(f"  已保存 {saved_count}/{sample_count} 帧")
        
        frame_count += 1
    
    cap.release()
    print(f"  [完成] 已保存 {saved_count} 个可视化帧到: {output_dir}")


def main():
    """主函数"""
    print("=" * 60)
    print("可视化姿态估计结果")
    print("=" * 60)
    print()
    
    # 加载配置
    config = load_config()
    
    # 强制使用 mediapipe tasks
    if 'models' not in config:
        config['models'] = {}
    if 'pose' not in config['models']:
        config['models']['pose'] = {}
    config['models']['pose']['backend'] = 'mediapipe'
    
    # 初始化姿态估计器
    print("[步骤1] 初始化姿态估计器...")
    try:
        pose_estimator = PoseEstimator(config)
        if pose_estimator.pose is None:
            print("[错误] 姿态估计器初始化失败")
            sys.exit(1)
        print("[OK] 姿态估计器初始化成功")
    except Exception as e:
        print(f"[错误] 姿态估计器初始化失败: {e}")
        sys.exit(1)
    
    print()
    
    # 准备视频路径
    video_names = ['cm1', 'cm2', 'cm3']
    video_dir = project_root / "data" / "raw" / "抻面"
    output_dir = project_root / "data" / "scores" / "抻面" / "pose_visualization"
    
    # 可视化每个视频
    print("[步骤2] 生成可视化结果...")
    print()
    
    for video_name in video_names:
        video_path = video_dir / f"{video_name}.mp4"
        
        if not video_path.exists():
            print(f"[警告] 视频不存在: {video_path}")
            continue
        
        video_output_dir = output_dir / video_name
        visualize_video_pose(video_path, pose_estimator, video_output_dir, sample_count=10)
    
    print()
    print("=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print()
    print(f"输出目录: {output_dir}")
    print()
    print("可视化说明：")
    print("  - 绿色线条/点: 头部")
    print("  - 洋红色线条: 躯干")
    print("  - 蓝色线条: 手臂（左臂和右臂）")
    print("  - 橙色线条: 腿部（左腿和右腿）")
    print("  - 红色圆点: 关键点（节点）")
    print()
    print("可以打开输出目录查看可视化结果，确认姿态检测效果！")


if __name__ == "__main__":
    main()