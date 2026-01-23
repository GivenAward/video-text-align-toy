#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_paths(
    repo_root: Path,
    config_path: Optional[Path],
    videos_path: Optional[Path],
    output_dir: Optional[Path],
) -> Tuple[Path, Path]:
    cfg_videos = None
    cfg_output = None
    if config_path:
        cfg = load_json(config_path, {})
        paths = cfg.get("paths", {})
        if "videos_meta" in paths:
            cfg_videos = Path(paths["videos_meta"])
        if "raw_videos_dir" in paths:
            cfg_output = Path(paths["raw_videos_dir"])

    final_videos = videos_path or cfg_videos or Path("metadata/videos.json")
    final_output = output_dir or cfg_output or Path("data/raw_videos")

    if not final_videos.is_absolute():
        final_videos = repo_root / final_videos
    if not final_output.is_absolute():
        final_output = repo_root / final_output

    return final_videos, final_output


def find_downloader(preferred: Optional[str]) -> str:
    if preferred:
        exe = shutil.which(preferred)
        if exe:
            return exe
        raise SystemExit(f"Downloader not found: {preferred}")

    for name in ("yt-dlp", "youtube-dl"):
        exe = shutil.which(name)
        if exe:
            return exe
    raise SystemExit("Downloader not found. Install yt-dlp or youtube-dl.")


def build_command(
    downloader: str,
    url: str,
    output_dir: Path,
    video_id: str,
    fmt: str,
    merge_format: Optional[str],
    retries: int,
    cookies: Optional[Path],
    cookies_from_browser: Optional[str],
) -> List[str]:
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    cmd = [
        downloader,
        "--no-playlist",
        "-f",
        fmt,
        "-o",
        output_template,
        "--retries",
        str(retries),
    ]
    if merge_format:
        cmd += ["--merge-output-format", merge_format]
    if cookies:
        cmd += ["--cookies", str(cookies)]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd.append(url)
    return cmd


def load_videos(path: Path) -> List[Dict]:
    videos = load_json(path, [])
    if not isinstance(videos, list):
        raise SystemExit("videos.json must be a list of objects.")
    return videos


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Download videos listed in metadata/videos.json.")
    parser.add_argument(
        "--config",
        help="Optional pipeline config (JSON) to resolve paths.",
    )
    parser.add_argument(
        "--videos",
        help="Path to videos.json (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for raw videos (overrides config).",
    )
    parser.add_argument(
        "--downloader",
        help="Downloader executable (yt-dlp or youtube-dl).",
    )
    parser.add_argument(
        "--format",
        default="bestvideo+bestaudio/best",
        help="Format selector for downloader.",
    )
    parser.add_argument(
        "--merge-format",
        default="mp4",
        help="Merge output format (e.g. mp4).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for downloads.",
    )
    parser.add_argument(
        "--cookies",
        help="Path to cookies.txt for private videos.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        help="Use browser cookies (e.g. chrome, firefox).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if output exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without downloading.",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    videos_path = Path(args.videos) if args.videos else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    videos_path, output_dir = resolve_paths(repo_root, config_path, videos_path, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    downloader = find_downloader(args.downloader)

    videos = load_videos(videos_path)
    for entry in videos:
        video_id = entry.get("video_id")
        url = entry.get("url")
        if not video_id or not url:
            continue

        target = output_dir / f"{video_id}.mp4"
        if target.exists() and not args.force:
            continue

        cmd = build_command(
            downloader=downloader,
            url=url,
            output_dir=output_dir,
            video_id=video_id,
            fmt=args.format,
            merge_format=args.merge_format,
            retries=args.retries,
            cookies=Path(args.cookies) if args.cookies else None,
            cookies_from_browser=args.cookies_from_browser,
        )
        if args.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
