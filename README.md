# Video–Text Alignment Toy Pipeline

A toy project for building a **video–text aligned dataset** from YouTube videos, designed for Visual-Language Model (VLM) experiments.

This project focuses on **pipeline design, alignment logic, and data quality control**, rather than code optimization or model performance.

---

## 🎯 Project Goal

The goal of this project is to automatically generate **(video clip, aligned text)** pairs that can be directly used for VLM training or prototyping.

Key objectives:
- Build an end-to-end multimodal data pipeline
- Align video clips with spoken text using timestamps
- Apply explicit filtering rules to improve data quality
- Clearly explain *why* each design decision was made

---

## 🧩 Pipeline Overview

The pipeline consists of several steps to convert raw YouTube videos into **aligned video–text pairs** suitable for VLM experiments:

```text
YouTube Videos (10)
      ↓
Audio Extraction
      ↓
Whisper ASR (Timestamped Transcript)
      ↓
Sentence-level Video Clip Segmentation
      ↓
Video–Text Alignment
      ↓
Filtering & Quality Control
      ↓
Final Dataset (clip.mp4, caption)
```
```
video-text-align-toy
├─ .python-version
├─ .venv
│  ├─ CACHEDIR.TAG
│  ├─ Lib
│  │  └─ site-packages
│  │     ├─ _virtualenv.pth
│  │     ├─ _virtualenv.py
│  │     └─ __pycache__
│  │        └─ _virtualenv.cpython-311.pyc
│  ├─ pyvenv.cfg
│  └─ Scripts
│     ├─ activate
│     ├─ activate.bat
│     ├─ activate.csh
│     ├─ activate.fish
│     ├─ activate.nu
│     ├─ activate.ps1
│     ├─ activate_this.py
│     ├─ deactivate.bat
│     ├─ pydoc.bat
│     ├─ python.exe
│     └─ pythonw.exe
├─ configs
│  └─ pipeline.json
├─ data
│  ├─ alignments
│  │  ├─ candidates
│  │  │  └─ alignments.jsonl
│  │  └─ filtered
│  │     └─ alignments.jsonl
│  ├─ audio
│  ├─ clips
│  │  └─ sentences
│  ├─ dataset
│  │  ├─ train
│  │  ├─ train.jsonl
│  │  ├─ val
│  │  └─ val.jsonl
│  ├─ raw_videos
│  └─ transcripts
│     ├─ clean_text
│     └─ whisper_raw
│        └─ yt_01.json
├─ docs
├─ logs
│  ├─ pipeline
│  └─ qc
├─ main.py
├─ metadata
│  └─ videos.json
├─ notebooks
├─ pyproject.toml
├─ README.md
└─ scripts
   ├─ align
   ├─ download
   │  └─ download_videos.py
   ├─ pipeline.py
   ├─ process
   ├─ qc
   └─ __pycache__
      └─ pipeline.cpython-311.pyc

```