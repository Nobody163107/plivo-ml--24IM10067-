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

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from features import frame_energy_db, f0_contour, load_wav, speech_before


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the starter training script."""
    parser = argparse.ArgumentParser(description="Train the starter EOT classifier.")
    parser.add_argument("--data_dir", required=True, help="Directory containing labels.csv")
    parser.add_argument("--out", default="predictions.csv", help="Output CSV path")
    return parser


def extract_features(x, sr, pause_start):
    """Extract a compact feature vector from audio strictly before the pause."""
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10:
        return np.zeros(3, dtype=np.float32)

    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    return np.array(
        [
            e[-5:].mean(),
            voiced[-3:].mean() if len(voiced) >= 3 else 0.0,
            len(seg) / sr,
        ],
        dtype=np.float32,
    )


def main() -> None:
    args = build_parser().parse_args()

    labels_path = os.path.join(args.data_dir, "labels.csv")
    with open(labels_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    cache = {}
    X, y, groups, keys = [], [], [], []
    for row in rows:
        path = os.path.join(args.data_dir, row["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(row["pause_start"])))
        y.append(1 if row["label"] == "eot" else 0)
        groups.append(row["turn_id"])
        keys.append((row["turn_id"], row["pause_index"]))
    X, y = np.array(X), np.array(y)

    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups)
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X[train_idx], y[train_idx])
    print(
        f"held-out turn accuracy: {clf.score(X[test_idx], y[test_idx]):.3f} "
        f"(chance ~ {max(np.mean(y), 1 - np.mean(y)):.3f})"
    )

    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for (turn_id, pause_index), probability in zip(keys, p):
            writer.writerow([turn_id, pause_index, f"{probability:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")
    print(
        "NOTE for your final predict.py: it must load a SAVED model and "
        "predict on unseen data, not refit like this sanity script."
    )


if __name__ == "__main__":
    main()
