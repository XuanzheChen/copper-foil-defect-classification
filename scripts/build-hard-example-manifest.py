#!/usr/bin/env python3
"""Build a hard-example manifest from train-set inference predictions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hard_example_sampling import build_hard_example_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--copper-boundary-margin-max", type=float, default=0.5)
    parser.add_argument("--foreign-negative-margin-min", type=float, default=-0.5)
    args = parser.parse_args()

    summary = build_hard_example_manifest(
        predictions_path=args.predictions,
        dataset_root=args.dataset_root,
        output_path=args.output,
        copper_boundary_margin_max=args.copper_boundary_margin_max,
        foreign_negative_margin_min=args.foreign_negative_margin_min,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
