#!/usr/bin/env python3
"""Render the compact CIFAR result artifact as a Markdown table."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "summaries" / "cifar100_core_results.json"


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    print("| D | Train RMS | Stages | J builds | Test teacher agreement |")
    print("|---:|---:|---:|---:|---:|")
    for row in payload["exact_endpoint_scaling"]:
        print(
            f"| {row['d']} | {row['train_rms']:.3e} | {row['accepted_stages']} "
            f"| {row['jacobian_builds']} | {100 * row['test_teacher_agreement']:.2f}% |"
        )


if __name__ == "__main__":
    main()

