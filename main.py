import argparse
from pathlib import Path

from scripts.pipeline import PipelineConfig, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Toy pipeline for video-text alignment dataset creation.")
    parser.add_argument(
        "--config",
        default="configs/pipeline.json",
        help="Path to pipeline config (JSON).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run external tools (ffmpeg/whisper).",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.load(Path(args.config))
    execute = args.execute or cfg.execute
    run_pipeline(cfg, execute)


if __name__ == "__main__":
    main()
