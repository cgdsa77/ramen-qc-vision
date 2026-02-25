import argparse
from pipelines.offline_pipeline import run_offline
from pipelines.api_server import run_api


def parse_args():
    parser = argparse.ArgumentParser(description="Ramen QC vision system")
    parser.add_argument("--mode", choices=["offline", "api"], default="offline")
    parser.add_argument("--input", type=str, help="input video path or camera id", default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out", type=str, default="reports/report.html")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "offline":
        run_offline(args.input, args.config, args.out)
    else:
        run_api(args.host, args.port, args.config)


if __name__ == "__main__":
    main()
