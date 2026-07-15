"""Skeleton: prosodic features + classifier. Runs as-is, scores poorly ON
PURPOSE. Your hour goes into extract_features() and what you learn from
your errors.

    python train.py --data_dir eot_data/english --out predictions.csv

Ideas worth testing (this is the assignment, not a checklist):
  - F0 slope over the last voiced region (statements fall, continuations
    often stay level or rise)
  - final-syllable lengthening: last voiced stretch duration vs the
    speaker's average
  - energy decay rate into the pause
  - speaking-rate context, position of the pause within the turn so far
  - anything you discover by LISTENING to your misclassified pauses
"""
import argparse
import csv
import os
import librosa
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from features import load_wav, speech_before, frame_energy_db, f0_contour


def extract_features(x, sr, pause_start):
    """Features from audio STRICTLY BEFORE pause_start.  <-- YOUR WORK"""
    seg = speech_before(x, sr, pause_start, window_s=1.5)

    # Very short segment
    if len(seg) < sr // 10:
        return np.zeros(11, dtype=np.float32)

    features = []

    # ENERGY FEATURES
    e = frame_energy_db(seg, sr)

    features.append(np.mean(e))          # Mean energy
    features.append(np.std(e))           # Energy variation
    features.append(np.max(e))           # Peak energy
    features.append(e[-1])               # Final energy

    if len(e) > 1:
        energy_slope = np.polyfit(np.arange(len(e)), e, 1)[0]
    else:
        energy_slope = 0.0

    features.append(energy_slope)

  
    # PITCH FEATURES
    
    f0 = f0_contour(seg, sr)

    voiced = f0[f0 > 0]

    if len(voiced):

        features.append(np.mean(voiced))
        features.append(np.std(voiced))
        features.append(voiced[-1])

        if len(voiced) > 1:
            pitch_slope = np.polyfit(
                np.arange(len(voiced)),
                voiced,
                1
            )[0]
        else:
            pitch_slope = 0.0

        features.append(pitch_slope)

        features.append(len(voiced) / len(f0))

    else:
        features.extend([0, 0, 0, 0, 0])

  
    # CONTEXT FEATURE
   

    features.append(len(seg) / sr)
    
    
    
    # MFCC FEATURES

    mfcc = librosa.feature.mfcc(
        y=seg,
        sr=sr,
        n_mfcc=13
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    

    return np.array(features, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    X, y, groups, keys = [], [], [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], r["pause_index"]))
    X, y = np.array(X), np.array(y)

    # quick sanity check on held-out TURNS (never split a turn across sets)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                  .split(X, y, groups))
    clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=0
    )
)
#     clf = HistGradientBoostingClassifier(
#     learning_rate=0.05,
#     max_depth=3,
#     max_iter=200,
#     random_state=0
# )
    clf.fit(X[tr], y[tr])
    print(f"held-out turn accuracy: {clf.score(X[te], y[te]):.3f} "
          f"(chance ~ {max(np.mean(y), 1-np.mean(y)):.3f})")

    # refit on everything, write predictions for the scorer
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")
    print("NOTE for your final predict.py: it must load a SAVED model and "
          "predict on unseen data, not refit like this sanity script.")


if __name__ == "__main__":
    main()
