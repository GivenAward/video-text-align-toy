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
