# Speech Emotion Recognition System
*Created by Ghostblaster08*
*Last Updated: 2025-02-14*

## Overview
A real-time speech emotion recognition system using Convolutional Neural Networks (CNN) to detect 8 different emotions from audio input. The system processes audio in real-time and provides visual feedback of emotion detection with confidence ratings.

## Emotions Detected
- ANG (Anger)
- CAL (Calm)
- DIS (Disgust)
- FEA (Fear)
- HAP (Happy)
- NEU (Neutral)
- SAD (Sad)
- SUR (Surprise)

## Requirements

### Hardware Requirements
- Microphone for real-time analysis
- GPU recommended (but not required)
- Minimum 4GB RAM
- x64 architecture system

### Software Requirements
```bash
# Python version
Python 3.11 or higher

# Required packages
pip install torch
pip install librosa
pip install numpy
pip install sounddevice
pip install pygame
pip install soundfile
