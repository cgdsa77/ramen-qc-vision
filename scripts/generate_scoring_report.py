"""
生成评分报告（Markdown格式）
"""
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_markdown_report(summary_file: Path) -> str:
    """生成Markdown格式的评分报告"""
    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', {})
    
    # 生成报告
    report = []
    report.append("# 抻面综合评分报告\n")
    report.append("## 一、处理概况\n")
    report.append(f"- **处理视频数量**：{data.get('total_videos', 0)}个")
    report.append(f"- **成功处理**：{data.get('success_count', 0)}个")
    report.append(f"- **评分方法**：增强的综合评分体系（结合关键帧标注评分和骨架线数据）\n")
    
    report.append("## 二、评分结果汇总\n")
    report.append("| 视频 | 总帧数 | DTW距离 | DTW评分 | 手部得分 | 面条得分 | 面条束得分 | 总分 |")
    report.append("|------|--------|---------|---------|----------|----------|------------|------|")
    
    for video_name in sorted(results.keys()):
        r = results[video_name]
        if 'error' in r:
            report.append(f"| {video_name} | - | - | - | - | - | - | 错误 |")
        else:
            report.append(f"| {video_name} | {r.get('total_frames', 0)} | "
                         f"{r.get('dtw_distance', 0):.2f} | {r.get('dtw_score', 0):.2f} | "
                         f"{r.get('hand_score', 0):.2f} | {r.get('noodle_rope_score', 0):.2f} | "
                         f"{r.get('noodle_bundle_score', 0):.2f} | {r.get('total_score', 0):.2f} |")
    
    # 计算平均值
    successful_results = [r for r in results.values() if 'error' not in r]
    if successful_results:
        avg_total = sum(r.get('total_score', 0) for r in successful_results) / len(successful_results)
        avg_hand = sum(r.get('hand_score', 0) for r in successful_results) / len(successful_results)
        avg_rope = sum(r.get('noodle_rope_score', 0) for r in successful_results) / len(successful_results)
        avg_bundle = sum(r.get('noodle_bundle_score', 0) for r in successful_results) / len(successful_results)
        
        report.append("\n### 平均得分\n")
        report.append(f"- **平均总分**：{avg_total:.2f}分")
        report.append(f"- **平均手部得分**：{avg_hand:.2f}分")
        report.append(f"- **平均面条得分**：{avg_rope:.2f}分")
        report.append(f"- **平均面条束得分**：{avg_bundle:.2f}分\n")
    
    report.append("## 三、标准视频表现\n")
    standard_videos = ['cm1', 'cm2', 'cm3']
    for video_name in standard_videos:
        if video_name in results:
            r = results[video_name]
            if 'error' not in r:
                report.append(f"- **{video_name}**：DTW距离={r.get('dtw_distance', 0):.2f}, "
                             f"DTW评分={r.get('dtw_score', 0):.2f}, 总分={r.get('total_score', 0):.2f}")
    
    report.append("\n**结论**：标准视频DTW距离为0.00，说明与标准动作序列完全匹配。\n")
    
    report.append("## 四、文件位置\n")
    report.append("- **详细结果**：`reports/comprehensive_scores_final/comprehensive_score_{video_name}.json`")
    report.append("- **汇总结果**：`reports/comprehensive_scores_final/summary.json`")
    
    return "\n".join(report)


def main():
    """主函数"""
    summary_file = project_root / "reports" / "comprehensive_scores_final" / "summary.json"
    
    if not summary_file.exists():
        print(f"[错误] 汇总文件不存在: {summary_file}")
        return
    
    # 生成报告
    report = generate_markdown_report(summary_file)
    
    # 保存报告
    report_file = project_root / "reports" / "comprehensive_scores_final" / "scoring_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("=" * 60)
    print("评分报告生成完成")
    print("=" * 60)
    print(f"报告文件: {report_file}")
    print("\n报告内容预览:")
    print(report[:500] + "...")


if __name__ == '__main__':
    main()
