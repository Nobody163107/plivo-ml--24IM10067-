"""Silence-only baseline: every pause looks like end-of-turn (p_eot = 1).

The agent then relies purely on the swept action delay — i.e., a silence
timer. This is what naive VAD endpointing does, and it is the number you
must beat.

Also the interface your own predict.py must match:
    python baseline.py --data_dir eot_data/english --out predictions.csv
"""

import argparse
import csv
import os


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the baseline script."""
    parser = argparse.ArgumentParser(
        description="Write silence-only end-of-turn predictions."
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing labels.csv")
    parser.add_argument("--out", default="predictions.csv", help="Output CSV path")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    rows = []
    labels_path = os.path.join(args.data_dir, "labels.csv")
    with open(labels_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "turn_id": row["turn_id"],
                    "pause_index": row["pause_index"],
                    "p_eot": 1.0,
                }
            )

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["turn_id", "pause_index", "p_eot"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
