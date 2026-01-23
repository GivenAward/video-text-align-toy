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

## 🗝️ Environment Setting(Install uv)
### Windows
 - cmd
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


### macOS, Linux
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### pip
```
pip install uv
```

### Install Environment
```
uv sync
```

## 🍔 Install ffmpeg 
### Windows
```
1. Install ffmpeg windows version: https://www.ffmpeg.org/
2. Environmental variables -> Path of the System variable -> Add installed ffmpeg bin path(e.g. C:\ffmpeg\bin)
```

### Linux
```
sudo apt update
sudo apt install ffmpeg
```

### macOS
```
brew install ffmpeg
```

### How to start
 - Firstly you should write Youtube url to 'data\metadata\videos.json'
 - Start with the code below

```
python scripts\download\download_videos.py
python scripts\pipeline.py
```