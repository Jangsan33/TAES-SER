# TAES-SER

Core implementation of **TAES-SER**, including task-aware routing, training objectives, and routing behavior analysis.

## Overview

This repository provides the core code of **TAES-SER**, a multitask mixture-of-experts framework for speech emotion recognition. The released code is aligned with the methodological description in the paper and preserves the main project structure, including main.py, model.py, and trainer.py, so that readers can clearly understand the correspondence between the method described in the paper and its code implementation.
The repository focuses on the core technical components of the proposed method, including:

- task-aware Top-K routing
- expert aggregation for multi-task learning
- routing-related regularization terms
- task–expert mutual information
- routing behavior analysis metrics such as Top-K coverage and Jaccard similarity

## Repository Structure

```text
TAES-SER/
├── main.py
├── model.py
├── trainer.py
├── requirements.txt
└── dataset/
    ├── emotion_speaker_text.train.csv
    └── emotion_speaker_text.test.csv
```

## File Description

- **main.py**
Used to construct and organize the main workflow for model training and evaluation, reflecting the overall implementation logic of the proposed method in the paper.
- **model.py**
Contains the core implementation of TAES-SER, including the multitask MoE model, task-aware routing, expert selection, and routing-related objectives.
- **trainer.py**
Provides the paper-oriented training framework, including the custom training logic and routing behavior analysis modules.
- **requirements.txt**
Lists the main dependencies required for inspecting the released code.
- **dataset/**
Contains the released train/test CSV files used to organize the dataset annotations for this project:
  emotion_speaker_text.train.csv、emotion_speaker_text.test.csv

## Dataset

This project is based on the IEMOCAP dataset. The official dataset can be requested from the USC SAIL website:
- **IEMOCAP homepage**: https://sail.usc.edu/iemocap/
- **IEMOCAP release page**: https://sail.usc.edu/iemocap/iemocap_release.htm

Please note that the original IEMOCAP data is distributed separately by the dataset provider and is subject to its own release and usage conditions. This repository does not redistribute the raw IEMOCAP audio data.

## Requirements

Install the main dependencies with:
```text
pip install -r requirements.txt
```
