"""Load a trained model and write EOT probabilities for new audio files."""

import argparse
import csv
import os

import joblib

from features import load_wav
from train import extract_features


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for prediction."""
    parser = argparse.ArgumentParser(description="Generate EOT predictions from a saved model.")
    parser.add_argument("--data_dir", required=True, help="Directory containing labels.csv")
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "model.pkl"),
        help="Path to the trained model",
    )
    parser.add_argument("--out", default="predictions.csv", help="Output CSV path")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    clf = joblib.load(args.model)

    labels_path = os.path.join(args.data_dir, "labels.csv")
    with open(labels_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    cache = {}
    predictions = []

    for row in rows:
        path = os.path.join(args.data_dir, row["audio_file"])

        if path not in cache:
            cache[path] = load_wav(path)

        x, sr = cache[path]
        feature_vector = extract_features(x, sr, float(row["pause_start"]))
        p = clf.predict_proba(feature_vector.reshape(1, -1))[0, 1]

        predictions.append(
            [
                row["turn_id"],
                row["pause_index"],
                f"{p:.4f}",
            ]
        )

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        writer.writerows(predictions)

    print(f"Wrote {len(predictions)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
