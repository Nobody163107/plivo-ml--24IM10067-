# End-of-Turn (EOT) Prediction using Prosodic and Spectral Features

This repository contains my solution for the Plivo Machine Learning Engineer assignment on End-of-Turn (EOT) prediction.

The objective is to determine whether a pause in a spoken conversation represents a temporary hesitation (**hold**) or the end of the speaker's turn (**EOT**) using only information available before the pause (strictly causal inference).

---

# Problem Statement

Given an audio segment and the start time of each pause, predict the probability that the pause corresponds to the end of the speaker's turn.

The implementation strictly follows the assignment constraint that **no information after `pause_start` is used** during feature extraction.

---

# Methodology

## Feature Engineering

Features are extracted from the **1.5-second causal speech window** immediately preceding each pause.

### Prosodic Features

- Mean frame energy
- Energy standard deviation
- Peak energy
- Final frame energy
- Energy slope
- Mean pitch (F0)
- Pitch standard deviation
- Final pitch
- Pitch slope
- Voiced frame ratio
- Speech context duration

### Spectral Features

Using Librosa:

- 13 MFCC mean coefficients
- 13 MFCC standard deviation coefficients

Total feature dimension:

```
37
```

---

# Model

Final pipeline:

```
StandardScaler
        ↓
LogisticRegression(
    class_weight="balanced",
    max_iter=3000
)
```

The model is serialized using Joblib and deployed through a standalone `predict.py` script.

---

# Repository Structure

```
.
├── features.py
├── train.py
├── predict.py
├── score.py
├── baseline.py
├── model.pkl
├── README.md
├── RUNLOG.md
├── NOTES.md
└── requirements.txt
```

---

# Running

## Train

```bash
python train.py \
    --data_dir eot_data/eot_data/english \
    --out predictions.csv
```

---

## Predict

```bash
python predict.py \
    --data_dir eot_data/eot_data/english \
    --out predictions.csv
```

---

## Evaluate

```bash
python score.py \
    --data_dir eot_data/eot_data/english \
    --pred predictions.csv
```

---

# Experimental Results

| Dataset | Held-out Accuracy | AUC | Mean Delay |
|---------|------------------:|----:|-----------:|
| English | **0.585** | **0.745** | **1045 ms** |
| Hindi | **0.552** | **0.814** | **850 ms** |

---

# Experiments

The complete experimentation process is documented in **RUNLOG.md**, including:

- Baseline evaluation
- Prosodic feature engineering
- MFCC feature extraction
- Gradient Boosting comparison
- Feature standardization
- Final model selection

---

# Notes

Additional implementation details, assumptions, and future improvements are documented in **NOTES.md**.

# Starter-Kit reference: 

- `baseline.py` — the silence-only baseline; also shows the exact predict.py
  interface you must ship.
- `features.py` — audio loading, framing, energy, autocorrelation pitch
  tracker. Utilities only; the features are your job.
- `train.py` — runnable skeleton (weak on purpose).
- `score.py` — the official scorer. Your dev loop:

```
python baseline.py --data_dir ../eot_data/english --out base.csv
python score.py    --data_dir ../eot_data/english --pred base.csv
python train.py    --data_dir ../eot_data/english --out mine.csv
python score.py    --data_dir ../eot_data/english --pred mine.csv
```

Log every score in RUNLOG.md. Listen to your errors — that is where the
points are.

(Appending my work): 