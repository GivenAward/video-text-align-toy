#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


FILLER_WORDS = {
    "uh",
    "um",
    "okay",
    "ok",
    "like",
    "you know",
    "right",
}


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class PipelineConfig:
    repo_root: Path
    videos_meta: Path
    raw_videos_dir: Path
    audio_dir: Path
    transcripts_dir: Path
    clips_dir: Path
    align_candidates_dir: Path
    align_filtered_dir: Path
    dataset_dir: Path
    logs_dir: Path
    selection_single_speaker: bool
    selection_minimal_scene_cuts: bool
    selection_no_captions: bool
    clip_min_seconds: float
    clip_max_seconds: float
    text_min_chars: int
    filter_dedupe: bool
    filter_black_frame_threshold: float
    filter_static_var_threshold: float
    filter_motion_var_threshold: float
    use_clip_filter: bool
    clip_score_threshold: float
    train_ratio: float
    execute: bool
    whisper_model: str
    whisper_language: Optional[str]
    whisper_task: str
    whisper_verbose: Optional[bool]

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        cfg = load_json(path, {})
        root = path.parent.parent if path.name else Path(".")
        paths = cfg.get("paths", {})
        selection = cfg.get("selection", {})
        clip = cfg.get("clip", {})
        text = cfg.get("text_filter", {})
        video = cfg.get("video_filter", {})
        clip_filter = cfg.get("clip_filter", {})
        split = cfg.get("split", {})
        whisper_cfg = cfg.get("whisper", {})
        return cls(
            repo_root=root,
            videos_meta=Path(paths.get("videos_meta", "metadata/videos.json")),
            raw_videos_dir=Path(paths.get("raw_videos_dir", "data/raw_videos")),
            audio_dir=Path(paths.get("audio_dir", "data/audio")),
            transcripts_dir=Path(paths.get("transcripts_dir", "data/transcripts/whisper_raw")),
            clips_dir=Path(paths.get("clips_dir", "data/clips/sentences")),
            align_candidates_dir=Path(paths.get("align_candidates_dir", "data/alignments/candidates")),
            align_filtered_dir=Path(paths.get("align_filtered_dir", "data/alignments/filtered")),
            dataset_dir=Path(paths.get("dataset_dir", "data/dataset")),
            logs_dir=Path(paths.get("logs_dir", "logs/pipeline")),
            selection_single_speaker=bool(selection.get("single_speaker", True)),
            selection_minimal_scene_cuts=bool(selection.get("minimal_scene_cuts", True)),
            selection_no_captions=bool(selection.get("no_captions", True)),
            clip_min_seconds=float(clip.get("min_seconds", 3.0)),
            clip_max_seconds=float(clip.get("max_seconds", 7.0)),
            text_min_chars=int(text.get("min_chars", 5)),
            filter_dedupe=bool(text.get("dedupe", True)),
            filter_black_frame_threshold=float(video.get("black_frame_threshold", 0.1)),
            filter_static_var_threshold=float(video.get("static_var_threshold", 0.001)),
            filter_motion_var_threshold=float(video.get("motion_var_threshold", 100.0)),
            use_clip_filter=bool(clip_filter.get("enabled", False)),
            clip_score_threshold=float(clip_filter.get("score_threshold", 0.25)),
            train_ratio=float(split.get("train_ratio", 0.9)),
            execute=bool(cfg.get("execute", False)),
            whisper_model=str(whisper_cfg.get("model", "base")),
            whisper_language=whisper_cfg.get("language"),
            whisper_task=str(whisper_cfg.get("task", "transcribe")),
            whisper_verbose=whisper_cfg.get("verbose", True),
        )


def select_videos(cfg: PipelineConfig) -> List[dict]:
    # Step 1: video selection based on metadata flags.
    videos = load_json(cfg.videos_meta, [])
    selected = []
    for v in videos:
        if cfg.selection_single_speaker and not v.get("single_speaker", False):
            continue
        if cfg.selection_minimal_scene_cuts and not v.get("minimal_scene_cuts", False):
            continue
        if cfg.selection_no_captions and v.get("has_captions", False):
            continue
        selected.append(v)
    return selected


def run(cmd: List[str], execute: bool) -> None:
    if not execute:
        return
    subprocess.run(cmd, check=True)


def extract_audio(cfg: PipelineConfig, videos: List[dict], execute: bool) -> Dict[str, Path]:
    # Step 2: audio extraction using ffmpeg.
    audio_paths = {}
    cfg.audio_dir.mkdir(parents=True, exist_ok=True)
    for v in videos:
        vid = v["video_id"]
        video_path = cfg.raw_videos_dir / f"{vid}.mp4"
        audio_path = cfg.audio_dir / f"{vid}.wav"
        audio_paths[vid] = audio_path
        if audio_path.exists():
            continue
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
        run(cmd, execute)
    return audio_paths


def transcribe_whisper(cfg: PipelineConfig, audio_paths: Dict[str, Path], execute: bool) -> Dict[str, Path]:
    # Step 3: Whisper ASR with timestamps.
    transcript_paths = {}
    cfg.transcripts_dir.mkdir(parents=True, exist_ok=True)
    if execute:
        try:
            import whisper
        except ImportError as exc:
            raise SystemExit("whisper package is required for execute mode.") from exc

        model = whisper.load_model(cfg.whisper_model)
    for vid, audio_path in audio_paths.items():
        out_path = cfg.transcripts_dir / f"{vid}.json"
        transcript_paths[vid] = out_path
        if out_path.exists():
            continue
        if execute:
            kwargs = {}
            if cfg.whisper_language:
                kwargs["language"] = cfg.whisper_language
            if cfg.whisper_task:
                kwargs["task"] = cfg.whisper_task
            if cfg.whisper_verbose is not None:
                kwargs["verbose"] = cfg.whisper_verbose
            result = model.transcribe(str(audio_path), **kwargs)
            save_json(out_path, result)
        else:
            placeholder = [
                {
                    "start": 0.0,
                    "end": 3.5,
                    "text": "TODO: add transcript",
                }
            ]
            save_json(out_path, placeholder)
    return transcript_paths


def segment_clips(cfg: PipelineConfig, transcripts: Dict[str, Path], execute: bool) -> Dict[str, List[dict]]:
    # Step 4: segment video clips by sentence-level timestamps.
    all_segments: Dict[str, List[dict]] = {}
    for vid, t_path in transcripts.items():
        segments = load_json(t_path, [])
        if isinstance(segments, dict):
            segments = segments.get("segments", [])
        filtered = []
        for idx, seg in enumerate(segments):
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            text = str(seg.get("text", "")).strip()
            duration = max(0.0, end - start)
            if duration < cfg.clip_min_seconds or duration > cfg.clip_max_seconds:
                continue
            if not text:
                continue
            filtered.append(
                {
                    "clip_id": f"clip_{idx:04d}",
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )
        all_segments[vid] = filtered

        # Optional: create clip files with ffmpeg if execute is true.
        if execute:
            video_path = cfg.raw_videos_dir / f"{vid}.mp4"
            out_dir = cfg.clips_dir / vid
            out_dir.mkdir(parents=True, exist_ok=True)
            for seg in filtered:
                clip_path = out_dir / f"{seg['clip_id']}.mp4"
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-ss",
                    str(seg["start"]),
                    "-to",
                    str(seg["end"]),
                    "-c",
                    "copy",
                    str(clip_path),
                ]
                run(cmd, execute)
    return all_segments


def align_segments(cfg: PipelineConfig, segments: Dict[str, List[dict]]) -> List[dict]:
    # Step 5: align video clips with transcript segments.
    alignments = []
    for vid, segs in segments.items():
        for seg in segs:
            clip_path = str((cfg.clips_dir / vid / f"{seg['clip_id']}.mp4"))
            alignments.append(
                {
                    "video_id": vid,
                    "clip_id": seg["clip_id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "caption": seg["text"],
                    "clip_path": clip_path,
                }
            )
    return alignments


def clean_text(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^\w\s']", " ", t)  # Remove special characters to " "
    t = re.sub(r"\s+", " ", t).strip()  # Remove at least 2 spaces to " "
    for w in FILLER_WORDS:
        t = re.sub(rf"\b{re.escape(w)}\b", "", t)  # Remove 'w' to ""
    return t


def filter_text(cfg: PipelineConfig, rows: List[dict]) -> List[dict]:
    # Step 6 (text): remove short, filler-only, or duplicate captions.
    seen = set()
    filtered = []
    for row in rows:
        cleaned = clean_text(row["caption"])
        if len(cleaned) < cfg.text_min_chars:
            continue
        if cfg.filter_dedupe:
            key = cleaned
            if key in seen:
                continue
            seen.add(key)
        new_row = dict(row)
        new_row["caption"] = cleaned
        filtered.append(new_row)
    return filtered


def load_clip_stats(cfg: PipelineConfig) -> Dict[str, dict]:
    # Optional metadata file produced by a separate analysis job.
    stats_path = cfg.repo_root / "metadata/clip_stats.json"
    return load_json(stats_path, {})


def filter_video(cfg: PipelineConfig, rows: List[dict]) -> List[dict]:
    # Step 6 (video): drop black/static/unstable clips if stats exist.
    stats = load_clip_stats(cfg)
    if not stats:
        return rows
    filtered = []
    for row in rows:
        key = f"{row['video_id']}/{row['clip_id']}"
        s = stats.get(key)
        if not s:
            filtered.append(row)
            continue
        if s.get("avg_brightness", 1.0) < cfg.filter_black_frame_threshold:
            continue
        if s.get("frame_var", 1.0) < cfg.filter_static_var_threshold:
            continue
        if s.get("flow_var", 0.0) > cfg.filter_motion_var_threshold:
            continue
        filtered.append(row)
    return filtered


def load_clip_scores(cfg: PipelineConfig) -> Dict[str, float]:
    scores_path = cfg.repo_root / "metadata/clip_scores.json"
    return load_json(scores_path, {})


def filter_clip(cfg: PipelineConfig, rows: List[dict]) -> List[dict]:
    # Step 6 (cross-modal): optional CLIP similarity filter.
    if not cfg.use_clip_filter:
        return rows
    scores = load_clip_scores(cfg)
    if not scores:
        return rows
    filtered = []
    for row in rows:
        key = f"{row['video_id']}/{row['clip_id']}"
        score = scores.get(key, 1.0)
        if score < cfg.clip_score_threshold:
            continue
        filtered.append(row)
    return filtered


def split_dataset(cfg: PipelineConfig, rows: List[dict]) -> Dict[str, List[dict]]:
    train, val = [], []
    for row in rows:
        key = f"{row['video_id']}::{row['clip_id']}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        if bucket < cfg.train_ratio:
            train.append(row)
        else:
            val.append(row)
    return {"train": train, "val": val}


def run_pipeline(cfg: PipelineConfig, execute: bool) -> None:
    videos = select_videos(cfg)
    audio = extract_audio(cfg, videos, execute)
    transcripts = transcribe_whisper(cfg, audio, execute)
    segments = segment_clips(cfg, transcripts, execute)
    alignments = align_segments(cfg, segments)

    candidates_path = cfg.align_candidates_dir / "alignments.jsonl"
    save_jsonl(candidates_path, alignments)

    filtered = filter_text(cfg, alignments)
    filtered = filter_video(cfg, filtered)
    filtered = filter_clip(cfg, filtered)

    filtered_path = cfg.align_filtered_dir / "alignments.jsonl"
    save_jsonl(filtered_path, filtered)

    split = split_dataset(cfg, filtered)
    save_jsonl(cfg.dataset_dir / "train.jsonl", split["train"])
    save_jsonl(cfg.dataset_dir / "val.jsonl", split["val"])


def main() -> int:
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
    raise SystemExit(main())
