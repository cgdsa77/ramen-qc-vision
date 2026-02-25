from typing import Any, Dict
import yaml


def load_config(path: str) -> Dict[str, Any]:
    encodings = ['utf-8', 'gbk', 'gb2312']
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取配置文件 {path}，已尝试编码: {', '.join(encodings)}")
