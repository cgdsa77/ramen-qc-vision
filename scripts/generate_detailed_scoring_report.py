"""
生成详细的评分报告，展示各项评分细节
"""
import json
from pathlib import Path
from typing import Dict, Any

project_root = Path(__file__).parent.parent
reports_dir = project_root / "reports" / "comprehensive_scores_final"
summary_file = reports_dir / "summary.json"

def format_score(score: float) -> str:
    """格式化分数显示"""
    return f"{score:.2f}"

def get_class_status(available_classes: list, class_name: str) -> str:
    """获取类别状态"""
    if class_name in available_classes:
        return "✅ 已评分"
    else:
        return "❌ 无标注（未纳入评分）"

def generate_detailed_report():
    """生成详细评分报告"""
    
    # 加载汇总数据
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    # 加载每个视频的详细数据
    video_details = {}
    for video_name in summary_data['results'].keys():
        detail_file = reports_dir / f"comprehensive_score_{video_name}.json"
        if detail_file.exists():
            with open(detail_file, 'r', encoding='utf-8') as f:
                video_details[video_name] = json.load(f)
    
    # 生成报告
    report_lines = []
    report_lines.append("# 抻面综合评分详细报告（动态权重调整版）")
    report_lines.append("")
    report_lines.append("## 一、评分说明")
    report_lines.append("")
    report_lines.append("### 评分规则")
    report_lines.append("- **动态权重调整**：只对实际存在标注数据的类别进行评分")
    report_lines.append("- 如果某个类别没有标注，自动重新归一化其他类别的权重")
    report_lines.append("- 避免因缺失类别而拉低总分")
    report_lines.append("")
    report_lines.append("### 类别说明")
    report_lines.append("- **hand（手部）**：位置、动作、角度、协调性")
    report_lines.append("- **noodle_rope（面条绳）**：粗细、弹性、光泽、完整性")
    report_lines.append("- **noodle_bundle（面条束）**：紧实度、均匀度（可选，不是所有视频都有）")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # 总体统计
    report_lines.append("## 二、总体统计")
    report_lines.append("")
    results = summary_data['results']
    
    # 计算平均值
    total_scores = [r['total_score'] for r in results.values()]
    hand_scores = [r['hand_score'] for r in results.values()]
    rope_scores = [r['noodle_rope_score'] for r in results.values()]
    bundle_scores = [r['noodle_bundle_score'] for r in results.values() if r['noodle_bundle_score'] > 0]
    
    report_lines.append(f"- **处理视频数量**：{summary_data['total_videos']}个")
    report_lines.append(f"- **成功评分数量**：{summary_data['success_count']}个")
    report_lines.append(f"- **平均总分**：{sum(total_scores)/len(total_scores):.2f}分")
    report_lines.append(f"- **平均手部得分**：{sum(hand_scores)/len(hand_scores):.2f}分")
    report_lines.append(f"- **平均面条得分**：{sum(rope_scores)/len(rope_scores):.2f}分")
    if bundle_scores:
        report_lines.append(f"- **平均面条束得分**（仅统计有标注的视频）：{sum(bundle_scores)/len(bundle_scores):.2f}分")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # 详细评分表格
    report_lines.append("## 三、各视频详细评分")
    report_lines.append("")
    
    # 按总分排序
    sorted_videos = sorted(results.items(), key=lambda x: x[1]['total_score'], reverse=True)
    
    for video_name, result in sorted_videos:
        detail = video_details.get(video_name, {})
        available_classes = detail.get('available_classes', [])
        dynamic_weights = detail.get('dynamic_weights', {})
        
        report_lines.append(f"### {video_name}")
        report_lines.append("")
        report_lines.append("#### 基本信息")
        report_lines.append(f"- **总帧数**：{result['total_frames']}")
        report_lines.append(f"- **评分帧数**：{result['scored_frames']}")
        report_lines.append("")
        
        report_lines.append("#### DTW序列匹配")
        report_lines.append(f"- **DTW距离**：{format_score(result['dtw_distance'])}")
        report_lines.append(f"- **DTW评分**：{format_score(result['dtw_score'])}")
        if 'dtw_result' in detail:
            matched_seq = detail['dtw_result'].get('matched_sequence', 'N/A')
            report_lines.append(f"- **匹配的标准序列**：cm{matched_seq + 1 if isinstance(matched_seq, int) else matched_seq}")
        report_lines.append("")
        
        report_lines.append("#### 各类别评分")
        report_lines.append("")
        report_lines.append("| 类别 | 得分 | 状态 | 权重 |")
        report_lines.append("|------|------|------|------|")
        
        # 手部评分
        hand_status = get_class_status(available_classes, 'hand')
        hand_weight = dynamic_weights.get('hand', 0)
        report_lines.append(f"| **手部（hand）** | {format_score(result['hand_score'])} | {hand_status} | {hand_weight:.1%} |")
        
        # 面条评分
        rope_status = get_class_status(available_classes, 'noodle_rope')
        rope_weight = dynamic_weights.get('noodle_rope', 0)
        report_lines.append(f"| **面条绳（noodle_rope）** | {format_score(result['noodle_rope_score'])} | {rope_status} | {rope_weight:.1%} |")
        
        # 面条束评分
        bundle_status = get_class_status(available_classes, 'noodle_bundle')
        bundle_weight = dynamic_weights.get('noodle_bundle', 0)
        bundle_score = result['noodle_bundle_score']
        if bundle_score > 0:
            report_lines.append(f"| **面条束（noodle_bundle）** | {format_score(bundle_score)} | {bundle_status} | {bundle_weight:.1%} |")
        else:
            report_lines.append(f"| **面条束（noodle_bundle）** | - | {bundle_status} | - |")
        
        report_lines.append("")
        
        # 手部详细属性（如果有）
        if 'frame_scores' in detail and detail['frame_scores']:
            # 统计手部属性平均分
            hand_attrs = {'position': [], 'action': [], 'angle': [], 'coordination': []}
            for frame_score in detail['frame_scores']:
                hand_data = frame_score.get('hand', {})
                if isinstance(hand_data, dict):
                    for attr in hand_attrs.keys():
                        if attr in hand_data:
                            hand_attrs[attr].append(hand_data[attr])
            
            if any(hand_attrs.values()):
                report_lines.append("#### 手部属性详细评分")
                report_lines.append("")
                report_lines.append("| 属性 | 平均分 |")
                report_lines.append("|------|--------|")
                for attr, values in hand_attrs.items():
                    if values:
                        avg = sum(values) / len(values)
                        report_lines.append(f"| {attr} | {format_score(avg)} |")
                report_lines.append("")
        
        # 面条详细属性（如果有）
        if 'frame_scores' in detail and detail['frame_scores']:
            rope_attrs = {'thickness': [], 'elasticity': [], 'gloss': [], 'integrity': []}
            for frame_score in detail['frame_scores']:
                rope_data = frame_score.get('noodle_rope', {})
                if isinstance(rope_data, dict):
                    for attr in rope_attrs.keys():
                        if attr in rope_data:
                            rope_attrs[attr].append(rope_data[attr])
            
            if any(rope_attrs.values()):
                report_lines.append("#### 面条绳属性详细评分")
                report_lines.append("")
                report_lines.append("| 属性 | 平均分 |")
                report_lines.append("|------|--------|")
                for attr, values in rope_attrs.items():
                    if values:
                        avg = sum(values) / len(values)
                        report_lines.append(f"| {attr} | {format_score(avg)} |")
                report_lines.append("")
        
        # 面条束详细属性（如果有）
        if 'noodle_bundle' in available_classes and 'frame_scores' in detail:
            bundle_attrs = {'tightness': [], 'uniformity': []}
            for frame_score in detail['frame_scores']:
                bundle_data = frame_score.get('noodle_bundle', {})
                if isinstance(bundle_data, dict):
                    for attr in bundle_attrs.keys():
                        if attr in bundle_data:
                            bundle_attrs[attr].append(bundle_data[attr])
            
            if any(bundle_attrs.values()):
                report_lines.append("#### 面条束属性详细评分")
                report_lines.append("")
                report_lines.append("| 属性 | 平均分 |")
                report_lines.append("|------|--------|")
                for attr, values in bundle_attrs.items():
                    if values:
                        avg = sum(values) / len(values)
                        report_lines.append(f"| {attr} | {format_score(avg)} |")
                report_lines.append("")
        
        report_lines.append(f"#### 综合评分")
        report_lines.append(f"- **总分**：**{format_score(result['total_score'])}** 分")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # 总结分析
    report_lines.append("## 四、评分分析")
    report_lines.append("")
    
    # 统计有/无面条束标注的视频
    videos_with_bundle = [v for v, r in results.items() if r['noodle_bundle_score'] > 0]
    videos_without_bundle = [v for v, r in results.items() if r['noodle_bundle_score'] == 0]
    
    report_lines.append("### 4.1 面条束标注情况")
    report_lines.append("")
    report_lines.append(f"- **有面条束标注的视频**（{len(videos_with_bundle)}个）：{', '.join(sorted(videos_with_bundle))}")
    report_lines.append(f"- **无面条束标注的视频**（{len(videos_without_bundle)}个）：{', '.join(sorted(videos_without_bundle))}")
    report_lines.append("")
    report_lines.append("> 注意：无面条束标注是正常情况，不是所有抻面视频都有面条束。")
    report_lines.append("> 动态权重调整确保这些视频只基于实际存在的类别进行评分。")
    report_lines.append("")
    
    # 动态权重分析
    report_lines.append("### 4.2 动态权重调整效果")
    report_lines.append("")
    report_lines.append("| 视频 | 参与评分的类别 | 权重分配 |")
    report_lines.append("|------|----------------|----------|")
    
    for video_name, result in sorted(results.items(), key=lambda x: x[0]):
        detail = video_details.get(video_name, {})
        available_classes = detail.get('available_classes', [])
        dynamic_weights = detail.get('dynamic_weights', {})
        
        classes_str = ', '.join(available_classes)
        weights_str = ', '.join([f"{cls}: {w:.1%}" for cls, w in sorted(dynamic_weights.items())])
        report_lines.append(f"| {video_name} | {classes_str} | {weights_str} |")
    
    report_lines.append("")
    
    # 评分排名
    report_lines.append("### 4.3 评分排名")
    report_lines.append("")
    report_lines.append("| 排名 | 视频 | 总分 | 手部得分 | 面条得分 | 面条束得分 |")
    report_lines.append("|------|------|------|----------|----------|------------|")
    
    for idx, (video_name, result) in enumerate(sorted_videos, 1):
        bundle_score = result['noodle_bundle_score']
        bundle_display = format_score(bundle_score) if bundle_score > 0 else "-"
        report_lines.append(
            f"| {idx} | {video_name} | {format_score(result['total_score'])} | "
            f"{format_score(result['hand_score'])} | {format_score(result['noodle_rope_score'])} | {bundle_display} |"
        )
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 五、改进建议")
    report_lines.append("")
    report_lines.append("1. **增加关键帧标注**：为得分较低的视频增加更多关键帧标注，提高评分准确性")
    report_lines.append("2. **优化骨架线检测**：提高手部骨架线检测的召回率，减少缺失帧")
    report_lines.append("3. **细化评分标准**：根据实际表现调整各属性的评分标准")
    report_lines.append("")
    
    # 保存报告
    report_file = reports_dir / "detailed_scoring_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"详细评分报告已生成：{report_file}")
    return report_file

if __name__ == "__main__":
    generate_detailed_report()
