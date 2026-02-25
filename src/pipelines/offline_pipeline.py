from typing import Dict, Any
from utils.config import load_config
from pipelines.steps import build_pipeline
from reporting.reporter import generate_report


def run_offline(input_path: str, config_path: str, report_path: str):
    cfg = load_config(config_path)
    pipeline = build_pipeline(cfg)
    results = pipeline.run(input_path)
    generate_report(results, cfg, report_path)
