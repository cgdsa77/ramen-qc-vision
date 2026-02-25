#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试评分器"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    from src.scoring.stretch_scorer import StretchScorer
    print("正在加载评分器...")
    scorer = StretchScorer()
    print("✅ 评分器加载成功！")
    print(f"评分规则版本: {scorer.rules.get('version', '未知')}")
    print(f"阶段: {scorer.rules.get('stage', '未知')}")
except Exception as e:
    print(f"❌ 评分器加载失败: {e}")
    import traceback
    traceback.print_exc()

