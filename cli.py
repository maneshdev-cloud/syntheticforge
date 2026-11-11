"""Command-line interface for SyntheticForge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate import evaluate_synthetic
from generate import generate_synthetic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate synthetic tabular data.")
    parser.add_argument("input", type=Path, help="Input CSV file")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the synthetic CSV")
    parser.add_argument("--report", type=Path, required=True, help="Where to write the metrics JSON")
    parser.add_argument("--method", choices=["copula", "marginal"], default="copula")
    parser.add_argument("--rows", type=int, default=None, help="Rows to generate (default: input row count)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.input)
    synthetic = generate_synthetic(real, n_rows=args.rows, method=args.method, seed=args.seed)
    report = evaluate_synthetic(real, synthetic)
    report["generation"] = {"method": args.method, "seed": args.seed}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {len(synthetic)} synthetic rows to {args.output}")
    print(f"Wrote evaluation report to {args.report}")


if __name__ == "__main__":
    main()
