"""
生成带骨架线的视频文件
将手部关键点直接绘制到视频上，生成新的视频文件
这样前端直接播放处理好的视频，不存在同步问题
使用moviepy确保浏览器兼容的H.264编码
"""
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入moviepy（用于生成浏览器兼容的视频）
MOVIEPY_AVAILABLE = False
try:
    # 尝试多种导入方式
    try:
        import moviepy
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        MOVIEPY_AVAILABLE = True
    except (ImportError, AttributeError):
        try:
            from moviepy.editor import ImageSequenceClip
            MOVIEPY_AVAILABLE = True
        except (ImportError, AttributeError):
            pass
except Exception:
    pass

if not MOVIEPY_AVAILABLE:
    print("[警告] 未安装moviepy，将使用OpenCV（可能浏览器不兼容）")
    print("[建议] 安装moviepy: pip install moviepy")


def draw_hand_skeleton(frame: np.ndarray, hands: List[Dict[str, Any]], scale_x=1.0, scale_y=1.0, 
                        confidence_threshold=0.2):
    """
    在帧上绘制手部骨架线
    颜色区分：
    - 手掌骨架：绿色
    - 关键点：红色
    - 前臂：橙色（从手腕短距离延伸，基于手掌方向）
    
    参数：
    - confidence_threshold: 置信度阈值，默认0.2（降低以提高召回率）
    
    注意：由于 MediaPipe Hand 模型只提供手部21个关键点，
    不包含肘部和肩膀的真实位置，因此不绘制后臂（推断的位置不准确）
    """
    if not hands or len(hands) == 0:
        return frame
    
    H, W = frame.shape[:2]
    
    # 手部关键点连接关系（MediaPipe HandLandmarker 21个点）
    HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
        [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
        [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
        [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
        [0, 17], [17, 18], [18, 19], [19, 20],  # 小指
    ]
    
    for hand in hands:
        if not hand.get('keypoints') or len(hand['keypoints']) < 21:
            continue
        
        kps = hand['keypoints']
        
        # 计算手部整体置信度（用于判断是否绘制）
        valid_kps = [kp for kp in kps if kp.get('confidence', 0) > confidence_threshold]
        if len(valid_kps) < 10:  # 至少需要10个有效关键点
            continue
        
        # 绘制手部连接线（绿色，手掌部分）
        # 降低阈值以提高召回率
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(kps) and end_idx < len(kps):
                p1 = kps[start_idx]
                p2 = kps[end_idx]
                # 只要有一个端点置信度足够高，或两个都在阈值以上就绘制
                conf1 = p1.get('confidence', 0)
                conf2 = p2.get('confidence', 0)
                if conf1 > confidence_threshold and conf2 > confidence_threshold:
                    x1 = int(p1['x'] * scale_x)
                    y1 = int(p1['y'] * scale_y)
                    x2 = int(p2['x'] * scale_x)
                    y2 = int(p2['y'] * scale_y)
                    # 根据置信度调整线条粗细
                    avg_conf = (conf1 + conf2) / 2
                    thickness = 2 if avg_conf > 0.5 else 1
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)  # 绿色
        
        # 绘制关键点（红色圆点）
        for kp in kps:
            if kp.get('confidence', 0) > confidence_threshold:
                x = int(kp['x'] * scale_x)
                y = int(kp['y'] * scale_y)
                # 根据置信度调整圆点大小
                radius = 3 if kp.get('confidence', 0) > 0.5 else 2
                cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)
        
        # 绘制前臂（从手腕短距离延伸）
        if len(kps) >= 21:
            wrist = kps[0]  # 手腕（关键点0）
            pinkyBase = kps[17]  # 小指根部
            indexBase = kps[5]   # 食指根部
            
            # 降低前臂绘制的阈值
            if (wrist.get('confidence', 0) > confidence_threshold and 
                pinkyBase.get('confidence', 0) > confidence_threshold and 
                indexBase.get('confidence', 0) > confidence_threshold):
                
                # 计算手掌中心（用于确定前臂方向）
                midPalmX = (pinkyBase['x'] + indexBase['x']) / 2
                midPalmY = (pinkyBase['y'] + indexBase['y']) / 2
                
                # 前臂方向：从手腕向手掌中心的反方向延伸
                dirX = wrist['x'] - midPalmX
                dirY = wrist['y'] - midPalmY
                
                length = np.sqrt(dirX * dirX + dirY * dirY)
                if length > 0:
                    dirX /= length
                    dirY /= length
                else:
                    continue  # 无法确定方向，跳过
                
                wristX = int(wrist['x'] * scale_x)
                wristY = int(wrist['y'] * scale_y)
                
                # 计算前臂长度（基于手掌宽度，保守延伸）
                palmWidth = np.sqrt(
                    (pinkyBase['x'] - indexBase['x'])**2 + 
                    (pinkyBase['y'] - indexBase['y'])**2
                )
                # 前臂长度约为手掌宽度的1.5倍，保守延伸避免误识别
                forearmLength = max(palmWidth * 1.5, 30)
                
                # 前臂终点
                forearmEndX = wrist['x'] + dirX * forearmLength
                forearmEndY = wrist['y'] + dirY * forearmLength
                forearmEndX_scaled = int(forearmEndX * scale_x)
                forearmEndY_scaled = int(forearmEndY * scale_y)
                
                # 绘制前臂（橙色，从手腕向外短距离延伸）
                forearm_in_frame = (0 <= forearmEndX_scaled < W and 0 <= forearmEndY_scaled < H)
                if forearm_in_frame:
                    cv2.line(frame, (wristX, wristY), (forearmEndX_scaled, forearmEndY_scaled), 
                            (0, 165, 255), 3)  # 橙色
    
    return frame


def process_video(video_path: Path, keypoints_file: Path, output_path: Path):
    """处理视频，生成带骨架线的视频"""
    if not video_path.exists():
        print(f"[错误] 视频文件不存在: {video_path}")
        return False
    
    if not keypoints_file.exists():
        print(f"[错误] 关键点文件不存在: {keypoints_file}")
        return False
    
    # 读取关键点数据
    with open(keypoints_file, 'r', encoding='utf-8') as f:
        keypoints_data = json.load(f)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n处理视频: {video_path.name}")
    print(f"  分辨率: {width}x{height}")
    print(f"  总帧数: {total_frames}, FPS: {fps:.2f}")
    
    # 创建视频写入器
    # 优先使用H.264编码器（浏览器兼容）
    # Windows上OpenCV可能不支持H.264，所以先尝试，失败则使用mp4v然后用ffmpeg转换
    fourcc_options = [
        ('avc1', 'H.264/AVC'),  # 浏览器最兼容
        ('H264', 'H.264'),      # 备选
    ]
    
    out = None
    used_fourcc = None
    temp_file = None
    
    # 先尝试H.264编码器
    for fourcc_code, fourcc_name in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            test_out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if test_out.isOpened():
                out = test_out
                used_fourcc = fourcc_name
                print(f"  使用编码器: {fourcc_name} ({fourcc_code})")
                break
            else:
                test_out.release()
        except Exception:
            continue
    
    # 如果H.264不可用，使用临时文件+ffmpeg转换
    if out is None or not out.isOpened():
        print(f"  [INFO] H.264编码器不可用，将使用临时文件+ffmpeg转换")
        temp_file = output_path.with_suffix('.tmp.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(temp_file), fourcc, fps, (width, height))
        if not out.isOpened():
            # 最后备选：使用mp4v
            print(f"  [INFO] XVID不可用，使用mp4v（可能需要ffmpeg转换）")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"[错误] 无法创建视频写入器")
                cap.release()
                return False
            used_fourcc = "mp4v"
        else:
            used_fourcc = "XVID (临时)"
    
    # 构建帧索引映射
    frame_map = {}
    for frame_data in keypoints_data.get('frames', []):
        frame_map[frame_data['frame_index']] = frame_data
    
    # 补全缺失的骨架线：使用前后帧插值（增强版）
    def interpolate_missing_frames(frame_map: dict, total_frames: int):
        """
        补全缺失帧的关键点数据
        改进：
        1. 同时处理"不存在的帧"和"存在但hands为空的帧"
        2. 扩大插值范围到60帧
        3. 提高插值帧的置信度保留
        """
        filled_map = {}
        interpolate_range = 60  # 扩大插值范围
        
        # 首先找出所有有效帧（有手部数据的帧）
        valid_frames = {}
        for frame_idx, frame_data in frame_map.items():
            if frame_data.get('hands') and len(frame_data['hands']) > 0:
                valid_frames[frame_idx] = frame_data
        
        for frame_idx in range(total_frames):
            # 检查当前帧是否有效（存在且有手部数据）
            current_frame = frame_map.get(frame_idx)
            has_valid_hands = (current_frame and current_frame.get('hands') and 
                              len(current_frame['hands']) > 0)
            
            if has_valid_hands:
                # 帧有效，直接使用
                filled_map[frame_idx] = current_frame
            else:
                # 帧无效（不存在或hands为空），尝试插值
                # 向前查找最近的有效帧
                prev_frame = None
                prev_distance = 0
                for i in range(frame_idx - 1, max(-1, frame_idx - interpolate_range), -1):
                    if i in valid_frames:
                        prev_frame = valid_frames[i]
                        prev_distance = frame_idx - i
                        break
                
                # 向后查找最近的有效帧
                next_frame = None
                next_distance = 0
                for i in range(frame_idx + 1, min(total_frames, frame_idx + interpolate_range)):
                    if i in valid_frames:
                        next_frame = valid_frames[i]
                        next_distance = i - frame_idx
                        break
                
                # 如果找到前后帧，进行插值
                if prev_frame and next_frame:
                    # 线性插值
                    total_distance = prev_distance + next_distance
                    alpha = prev_distance / total_distance
                    interpolated_hands = []
                    prev_hands = prev_frame.get('hands', [])
                    next_hands = next_frame.get('hands', [])
                    
                    # 对每只手进行插值
                    max_hands = max(len(prev_hands), len(next_hands))
                    for hand_idx in range(max_hands):
                        if hand_idx < len(prev_hands) and hand_idx < len(next_hands):
                            prev_hand = prev_hands[hand_idx]
                            next_hand = next_hands[hand_idx]
                            interpolated_hand = {'id': prev_hand.get('id', hand_idx), 'keypoints': []}
                            
                            # 对每个关键点进行插值
                            prev_kps = prev_hand.get('keypoints', [])
                            next_kps = next_hand.get('keypoints', [])
                            max_kps = min(len(prev_kps), len(next_kps))  # 使用最小值确保对齐
                            
                            for kp_idx in range(max_kps):
                                prev_kp = prev_kps[kp_idx]
                                next_kp = next_kps[kp_idx]
                                
                                # 计算插值置信度：距离越近置信度越高
                                # 不再取最小值，而是根据距离加权
                                prev_conf = prev_kp.get('confidence', 0.5)
                                next_conf = next_kp.get('confidence', 0.5)
                                # 根据距离加权平均置信度，并根据总距离稍微降低
                                distance_factor = max(0.5, 1.0 - total_distance / 120.0)
                                interpolated_conf = (prev_conf * (1 - alpha) + next_conf * alpha) * distance_factor
                                
                                interpolated_kp = {
                                    'x': prev_kp['x'] * (1 - alpha) + next_kp['x'] * alpha,
                                    'y': prev_kp['y'] * (1 - alpha) + next_kp['y'] * alpha,
                                    'z': prev_kp.get('z', 0) * (1 - alpha) + next_kp.get('z', 0) * alpha,
                                    'confidence': interpolated_conf
                                }
                                interpolated_hand['keypoints'].append(interpolated_kp)
                            
                            if len(interpolated_hand['keypoints']) >= 21:
                                interpolated_hands.append(interpolated_hand)
                    
                    if interpolated_hands:
                        filled_map[frame_idx] = {
                            'frame_index': frame_idx,
                            'hands': interpolated_hands,
                            'interpolated': True  # 标记为插值帧
                        }
                elif prev_frame and prev_distance <= 15:
                    # 只有前帧且距离不太远，复制前帧数据
                    # 根据距离降低置信度
                    distance_factor = max(0.3, 1.0 - prev_distance / 30.0)
                    copied_hands = []
                    for hand in prev_frame.get('hands', []):
                        copied_hand = {'id': hand.get('id', 0), 'keypoints': []}
                        for kp in hand.get('keypoints', []):
                            copied_kp = kp.copy()
                            copied_kp['confidence'] = kp.get('confidence', 0.5) * distance_factor
                            copied_hand['keypoints'].append(copied_kp)
                        if len(copied_hand['keypoints']) >= 21:
                            copied_hands.append(copied_hand)
                    if copied_hands:
                        filled_map[frame_idx] = {
                            'frame_index': frame_idx,
                            'hands': copied_hands,
                            'interpolated': True
                        }
                elif next_frame and next_distance <= 15:
                    # 只有后帧且距离不太远，复制后帧数据
                    distance_factor = max(0.3, 1.0 - next_distance / 30.0)
                    copied_hands = []
                    for hand in next_frame.get('hands', []):
                        copied_hand = {'id': hand.get('id', 0), 'keypoints': []}
                        for kp in hand.get('keypoints', []):
                            copied_kp = kp.copy()
                            copied_kp['confidence'] = kp.get('confidence', 0.5) * distance_factor
                            copied_hand['keypoints'].append(copied_kp)
                        if len(copied_hand['keypoints']) >= 21:
                            copied_hands.append(copied_hand)
                    if copied_hands:
                        filled_map[frame_idx] = {
                            'frame_index': frame_idx,
                            'hands': copied_hands,
                            'interpolated': True
                        }
        
        return filled_map
    
    # 补全缺失帧
    print("  补全缺失的骨架线...")
    original_frame_indices = set(frame_map.keys())
    frame_map = interpolate_missing_frames(frame_map, total_frames)
    filled_count = len(set(frame_map.keys()) - original_frame_indices)
    if filled_count > 0:
        print(f"  已补全 {filled_count} 帧的骨架线数据")
    
    frame_index = 0
    processed_count = 0
    
    # 优先使用moviepy（生成H.264，浏览器兼容）
    # 注意：moviepy需要先收集所有帧，然后一次性生成视频
    use_moviepy = MOVIEPY_AVAILABLE
    frames_list = [] if use_moviepy else None
    
    if use_moviepy:
        print("  使用moviepy生成视频（H.264编码，浏览器兼容）...")
        # 不创建OpenCV写入器，直接收集帧
        if out and out.isOpened():
            out.release()
        out = None
        temp_file = None
    
    print("开始处理...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 查找对应帧的关键点数据
        frame_data = frame_map.get(frame_index)
        
        if frame_data and frame_data.get('hands'):
            # 绘制骨架线
            frame = draw_hand_skeleton(frame.copy(), frame_data['hands'])
            processed_count += 1
        
        # 如果使用moviepy，收集帧（RGB格式）
        if frames_list is not None:
            # 转换为RGB（moviepy需要）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame_rgb)
        elif out:
            # 写入帧到视频（OpenCV）
            out.write(frame)
        
        frame_index += 1
        
        # 进度显示
        if frame_index % 30 == 0:
            progress = (frame_index / total_frames) * 100
            print(f"  进度: {frame_index}/{total_frames} ({progress:.1f}%), "
                  f"已绘制骨架: {processed_count} 帧")
    
    cap.release()
    
    # 如果使用moviepy，用moviepy生成视频（浏览器兼容）
    if frames_list is not None and len(frames_list) > 0:
        print(f"  使用moviepy生成H.264编码视频（浏览器兼容）...")
        print(f"  总帧数: {len(frames_list)}, 分辨率: {width}x{height}, FPS: {fps}")
        try:
            # 使用moviepy生成视频
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError:
                try:
                    from moviepy.editor import ImageSequenceClip
                except ImportError:
                    raise ImportError("无法导入ImageSequenceClip")
            
            clip = ImageSequenceClip(frames_list, fps=fps)
            clip.write_videofile(
                str(output_path),
                codec='libx264',
                fps=fps,
                preset='medium',
                bitrate='5000k',
                audio=False,
                logger=None,  # 减少输出
                threads=4  # 多线程加速
            )
            clip.close()
            used_fourcc = "H.264 (moviepy)"
            # 仅用 ASCII，避免 Windows GBK 控制台在打印 emoji 时抛错，误触发下方回退覆盖已生成的 H.264
            print("  [OK] moviepy H.264 done (browser compatible)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [错误] moviepy生成失败: {e}")
            # moviepy 可能已成功写出文件，仅后续日志打印失败；切勿用 XVID 覆盖已有 mp4
            if output_path.exists() and output_path.stat().st_size > 1000:
                print("  [提示] 输出文件已存在且非空，保留为 H.264，跳过 OpenCV 回退。")
                used_fourcc = "H.264 (moviepy)"
            else:
                print("  [回退] 使用OpenCV生成（可能浏览器不兼容）...")
                temp_file = output_path.with_suffix('.tmp.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(str(temp_file), fourcc, fps, (width, height))
                for frame_bgr in [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_list]:
                    out.write(frame_bgr)
                out.release()
                if temp_file.exists():
                    if output_path.exists():
                        output_path.unlink()
                    temp_file.rename(output_path)
                used_fourcc = "XVID (回退)"
    else:
        # 使用OpenCV写入器
        if out:
            out.release()
        
        # 如果使用了临时文件，尝试转换
        if temp_file and temp_file.exists():
            print(f"  [警告] 使用临时文件，需要ffmpeg转换（但ffmpeg未安装）")
            print(f"  [建议] 安装ffmpeg或确保moviepy可用以获得浏览器兼容的视频")
            if output_path.exists():
                output_path.unlink()
            temp_file.rename(output_path)
            used_fourcc = "XVID (未转换)"
    
    print(f"\n[完成] {video_path.name}")
    print(f"  总帧数: {frame_index}")
    print(f"  已绘制骨架: {processed_count} 帧 ({processed_count/frame_index*100:.1f}%)")
    print(f"  输出文件: {output_path}")
    print(f"  编码器: {used_fourcc}")
    
    return True


def main():
    """主函数：处理所有视频；--video cm16 仅处理单个抻面视频"""
    import argparse
    parser = argparse.ArgumentParser(description="由原片 + hand_keypoints JSON 生成带骨架叠加的 mp4")
    parser.add_argument("--video", type=str, default=None, metavar="NAME", help="仅处理该抻面视频名（如 cm16），需 data/raw/抻面 与 hand_keypoints 已就绪")
    args = parser.parse_args()

    print("=" * 60)
    print("生成带骨架线的视频文件")
    print("=" * 60)
    
    # 处理抻面视频
    stretch_video_dir = project_root / "data" / "raw" / "抻面"
    stretch_keypoints_dir = project_root / "data" / "scores" / "抻面" / "hand_keypoints"
    stretch_output_dir = project_root / "data" / "processed_videos" / "抻面"
    stretch_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理下面及捞面视频
    boiling_video_dir = project_root / "data" / "raw" / "下面及捞面"
    boiling_keypoints_dir = project_root / "data" / "scores" / "下面及捞面" / "hand_keypoints"
    boiling_output_dir = project_root / "data" / "processed_videos" / "下面及捞面"
    boiling_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_tasks = []

    if args.video:
        vn = args.video.strip()
        video_file = None
        if stretch_video_dir.exists():
            for ext in (".mp4", ".MP4", ".mov", ".MOV"):
                p = stretch_video_dir / f"{vn}{ext}"
                if p.is_file():
                    video_file = p
                    break
        keypoints_file = stretch_keypoints_dir / f"hand_keypoints_{vn}.json"
        output_file = stretch_output_dir / f"{vn}_with_skeleton.mp4"
        if not video_file:
            print(f"\n[错误] 未找到抻面原片: {stretch_video_dir / (vn + '.mp4')}")
            print("  请将视频放入 data\\raw\\抻面\\")
            return
        if not keypoints_file.is_file():
            print(f"\n[错误] 未找到骨架数据: {keypoints_file}")
            print(f"  请先运行: python scripts/extract_hand_keypoints_from_video.py --video {vn}")
            return
        print(f"\n仅处理: {vn}")
        if process_video(video_file, keypoints_file, output_file):
            print("\n完成。可在「手部姿态视频展示」页选择该视频播放。")
        return
    
    # 收集抻面视频任务
    if stretch_video_dir.exists() and stretch_keypoints_dir.exists():
        for video_file in stretch_video_dir.glob("*.mp4"):
            video_name = video_file.stem
            keypoints_file = stretch_keypoints_dir / f"hand_keypoints_{video_name}.json"
            if keypoints_file.exists():
                output_file = stretch_output_dir / f"{video_name}_with_skeleton.mp4"
                all_tasks.append((video_file, keypoints_file, output_file, "抻面"))
    
    # 收集下面及捞面视频任务
    if boiling_video_dir.exists() and boiling_keypoints_dir.exists():
        for video_file in boiling_video_dir.glob("*.mp4"):
            video_name = video_file.stem
            keypoints_file = boiling_keypoints_dir / f"hand_keypoints_{video_name}.json"
            if keypoints_file.exists():
                output_file = boiling_output_dir / f"{video_name}_with_skeleton.mp4"
                all_tasks.append((video_file, keypoints_file, output_file, "下面及捞面"))
    
    if not all_tasks:
        print("\n[警告] 未找到可处理的视频")
        print("  请先运行预处理脚本生成关键点数据：")
        print("  python scripts/extract_hand_keypoints_from_video.py")
        return
    
    print(f"\n总共需要处理 {len(all_tasks)} 个视频")
    print("=" * 60)
    
    # 处理每个视频
    success_count = 0
    for i, (video_path, keypoints_file, output_path, stage) in enumerate(all_tasks, 1):
        print(f"\n[{i}/{len(all_tasks)}] 处理 {stage} 视频: {video_path.name}")
        if process_video(video_path, keypoints_file, output_path):
            success_count += 1
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"成功处理 {success_count}/{len(all_tasks)} 个视频")
    print("\n输出目录：")
    print(f"  抻面: {stretch_output_dir}")
    print(f"  下面及捞面: {boiling_output_dir}")
    print("\n说明：")
    print("  - 生成的视频文件已包含骨架线，可以直接播放")
    print("  - 前端可以直接播放这些视频，不存在同步问题")


if __name__ == "__main__":
    main()
