"""
从cm1-cm3提取标准姿态模板
使用预训练模型（MediaPipe）自动提取关键点序列，无需手动标注
"""
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

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
            # 如果UTF-8失败，尝试其他编码
            try:
                with open(config_path, 'r', encoding='gbk') as f:
                    config = yaml.safe_load(f)
            except:
                # 如果都失败，使用默认配置
                config = {
                    "models": {
                        "pose": {
                            "backend": "mediapipe"
                        }
                    }
                }
    else:
        # 默认配置
        config = {
            "models": {
                "pose": {
                    "backend": "mediapipe"
                }
            }
        }
    
    return config


def extract_pose_from_video(video_path: Path, pose_estimator: PoseEstimator, 
                           sample_fps: float = 2.0) -> List[Dict[str, Any]]:
    """
    从视频中提取姿态关键点序列
    
    Args:
        video_path: 视频文件路径
        pose_estimator: 姿态估计器
        sample_fps: 采样帧率（每秒提取多少帧）
        
    Returns:
        姿态序列列表，每个元素包含 {frame_index, timestamp, keypoints}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[警告] 无法打开视频: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps / sample_fps) if fps > 0 else 30
    
    print(f"[信息] 视频: {video_path.name}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 采样间隔: 每 {frame_interval} 帧提取一次")
    
    pose_sequence = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按采样间隔提取
        if frame_count % frame_interval == 0:
            # 提取关键点
            keypoints = pose_estimator.run_frame(frame)
            
            if keypoints:
                timestamp = frame_count / fps if fps > 0 else 0.0
                pose_sequence.append({
                    'frame_index': frame_count,
                    'timestamp': timestamp,
                    'keypoints': keypoints,
                    'keypoint_count': len(keypoints),
                    'detected': True
                })
                extracted_count += 1
            else:
                # 未检测到姿态
                timestamp = frame_count / fps if fps > 0 else 0.0
                pose_sequence.append({
                    'frame_index': frame_count,
                    'timestamp': timestamp,
                    'keypoints': [],
                    'keypoint_count': 0,
                    'detected': False
                })
        
        frame_count += 1
        
        # 显示进度
        if frame_count % 100 == 0:
            print(f"  处理中: {frame_count}/{total_frames} 帧, 已提取 {extracted_count} 个姿态")
    
    cap.release()
    
    print(f"  [完成] 共提取 {extracted_count} 个有效姿态（共 {len(pose_sequence)} 个采样点）")
    
    return pose_sequence


def calculate_keypoint_statistics(pose_sequences: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    计算关键点统计信息（均值、标准差等）
    
    Args:
        pose_sequences: 多个视频的姿态序列列表
        
    Returns:
        统计信息字典
    """
    # 收集所有检测到的关键点
    all_keypoints_by_name = defaultdict(list)
    
    for sequence in pose_sequences:
        for frame_data in sequence:
            if frame_data.get('detected', False):
                for kp in frame_data['keypoints']:
                    kp_name = kp.get('name', 'unknown')
                    all_keypoints_by_name[kp_name].append({
                        'x': kp['x'],
                        'y': kp['y'],
                        'z': kp.get('z', 0),
                        'confidence': kp.get('confidence', 0)
                    })
    
    # 计算统计信息
    statistics = {}
    for kp_name, points in all_keypoints_by_name.items():
        if not points:
            continue
        
        x_values = [p['x'] for p in points]
        y_values = [p['y'] for p in points]
        confidences = [p['confidence'] for p in points]
        
        statistics[kp_name] = {
            'x': {
                'mean': float(np.mean(x_values)),
                'std': float(np.std(x_values)),
                'min': float(np.min(x_values)),
                'max': float(np.max(x_values))
            },
            'y': {
                'mean': float(np.mean(y_values)),
                'std': float(np.std(y_values)),
                'min': float(np.min(y_values)),
                'max': float(np.max(y_values))
            },
            'confidence': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            },
            'sample_count': len(points)
        }
    
    return statistics


def calculate_angle_statistics(pose_sequences: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    计算关键角度统计信息（如手臂角度）
    
    Args:
        pose_sequences: 多个视频的姿态序列列表
        
    Returns:
        角度统计信息字典
    """
    # 定义关键角度（基于MediaPipe关键点索引）
    angle_definitions = {
        'left_arm_angle': {
            'points': ['left_shoulder', 'left_elbow', 'left_wrist'],
            'description': '左臂角度（肩-肘-腕）'
        },
        'right_arm_angle': {
            'points': ['right_shoulder', 'right_elbow', 'right_wrist'],
            'description': '右臂角度（肩-肘-腕）'
        },
        'left_shoulder_angle': {
            'points': ['left_elbow', 'left_shoulder', 'right_shoulder'],
            'description': '左肩角度'
        },
        'right_shoulder_angle': {
            'points': ['right_elbow', 'right_shoulder', 'left_shoulder'],
            'description': '右肩角度'
        }
    }
    
    def calculate_angle(p1, p2, p3):
        """计算三点之间的角度（p2为顶点）"""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return float(angle)
    
    angle_values = defaultdict(list)
    
    for sequence in pose_sequences:
        for frame_data in sequence:
            if not frame_data.get('detected', False):
                continue
            
            keypoints_dict = {kp['name']: kp for kp in frame_data['keypoints']}
            
            for angle_name, angle_def in angle_definitions.items():
                point_names = angle_def['points']
                
                # 检查所有关键点是否都存在且置信度足够
                if all(name in keypoints_dict and 
                       keypoints_dict[name].get('confidence', 0) > 0.5 
                       for name in point_names):
                    
                    points = [keypoints_dict[name] for name in point_names]
                    angle = calculate_angle(points[0], points[1], points[2])
                    angle_values[angle_name].append(angle)
    
    # 计算统计信息
    angle_statistics = {}
    for angle_name, values in angle_values.items():
        if values:
            angle_statistics[angle_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'sample_count': len(values),
                'description': angle_definitions[angle_name]['description']
            }
    
    return angle_statistics


def main():
    """主函数"""
    print("=" * 60)
    print("从cm1-cm3提取标准姿态模板")
    print("=" * 60)
    print()
    
    # 加载配置
    config = load_config()
    
    # 确保使用mediapipe（因为movenet还未实现）
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
            print("[错误] 姿态估计器初始化失败，请检查MediaPipe是否安装")
            print("  安装命令: pip install mediapipe")
            sys.exit(1)
        print("[OK] 姿态估计器初始化成功（使用MediaPipe）")
    except Exception as e:
        print(f"[错误] 姿态估计器初始化失败: {e}")
        sys.exit(1)
    
    print()
    
    # 准备视频路径
    video_names = ['cm1', 'cm2', 'cm3']
    video_dir = project_root / "data" / "raw" / "抻面"
    
    # 提取姿态序列
    print("[步骤2] 提取姿态关键点序列...")
    print()
    
    all_sequences = {}
    
    for video_name in video_names:
        video_path = video_dir / f"{video_name}.mp4"
        
        if not video_path.exists():
            print(f"[警告] 视频不存在: {video_path}")
            continue
        
        print(f"\n处理视频: {video_name}")
        pose_sequence = extract_pose_from_video(video_path, pose_estimator, sample_fps=2.0)
        all_sequences[video_name] = pose_sequence
    
    print()
    print("[步骤3] 计算统计信息...")
    
    # 计算关键点统计信息
    sequences_list = list(all_sequences.values())
    keypoint_stats = calculate_keypoint_statistics(sequences_list)
    angle_stats = calculate_angle_statistics(sequences_list)
    
    # 计算整体统计
    total_frames = sum(len(seq) for seq in sequences_list)
    detected_frames = sum(sum(1 for f in seq if f.get('detected', False)) 
                         for seq in sequences_list)
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0.0
    
    print(f"  - 总采样帧数: {total_frames}")
    print(f"  - 检测到姿态的帧数: {detected_frames}")
    print(f"  - 姿态检测率: {detection_rate:.2%}")
    print(f"  - 关键点类型数: {len(keypoint_stats)}")
    print(f"  - 角度类型数: {len(angle_stats)}")
    
    print()
    print("[步骤4] 保存标准模板...")
    
    # 构建标准模板
    template = {
        'metadata': {
            'source_videos': video_names,
            'extraction_method': 'MediaPipe自动提取',
            'sample_fps': 2.0,
            'total_frames': total_frames,
            'detected_frames': detected_frames,
            'detection_rate': float(detection_rate),
            'extraction_date': str(Path().cwd())
        },
        'keypoint_statistics': keypoint_stats,
        'angle_statistics': angle_stats,
        'pose_sequences': {
            # 为了节省空间，只保存关键信息，不保存完整的每帧关键点
            # 如果需要完整序列，可以单独保存
            video_name: {
                'frame_count': len(seq),
                'detected_count': sum(1 for f in seq if f.get('detected', False)),
                'detection_rate': sum(1 for f in seq if f.get('detected', False)) / len(seq) if seq else 0.0,
                'sample_indices': [f['frame_index'] for f in seq[:10]]  # 只保存前10个采样点作为示例
            }
            for video_name, seq in all_sequences.items()
        }
    }
    
    # 保存到文件
    output_dir = project_root / "data" / "scores" / "抻面"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "pose_template_cm1_cm3.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] 标准模板已保存: {output_file}")
    
    # 保存完整的姿态序列到单独文件（用于后续分析）
    full_sequence_file = output_dir / "pose_sequences_cm1_cm3.json"
    with open(full_sequence_file, 'w', encoding='utf-8') as f:
        # 为了节省空间，只保存必要信息
        compact_sequences = {}
        for video_name, sequence in all_sequences.items():
            compact_sequences[video_name] = [
                {
                    'frame_index': f['frame_index'],
                    'timestamp': f['timestamp'],
                    'detected': f['detected'],
                    'keypoint_count': f['keypoint_count'],
                    # 只保存关键关键点（手臂相关）
                    'key_arm_keypoints': {
                        kp['name']: {'x': kp['x'], 'y': kp['y'], 'confidence': kp.get('confidence', 0)}
                        for kp in f['keypoints']
                        if kp['name'] in ['left_shoulder', 'left_elbow', 'left_wrist', 
                                         'right_shoulder', 'right_elbow', 'right_wrist']
                    } if f.get('detected', False) else {}
                }
                for f in sequence
            ]
        
        json.dump({
            'metadata': template['metadata'],
            'sequences': compact_sequences
        }, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] 完整序列已保存: {full_sequence_file}")
    
    print()
    print("=" * 60)
    print("提取完成！")
    print("=" * 60)
    print()
    print("输出文件：")
    print(f"  1. 标准模板（统计信息）: {output_file}")
    print(f"  2. 完整序列（详细数据）: {full_sequence_file}")
    print()
    print("下一步：")
    print("  1. 检查姿态检测率是否满足要求（建议>80%）")
    print("  2. 查看关键点统计信息，确认关键点位置合理")
    print("  3. 验证角度统计信息，确认动作范围合理")


if __name__ == "__main__":
    main()