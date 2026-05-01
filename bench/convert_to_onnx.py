#!/usr/bin/env python3
"""
bench/convert_to_onnx.py

Convert a frozen TF .pb graph or SavedModel to ONNX using tf2onnx.

Usage:
  python bench/convert_to_onnx.py bench/models/bert_model.pb
  python bench/convert_to_onnx.py bench/models/bench_small.pb --opset 17

Requirements:
  pip install tf2onnx onnx

Behavior:
  - If a corresponding .savedmodel directory exists (e.g., bert_model.savedmodel/),
    it will be converted automatically. SavedModel format does not require
    manual input/output specification.
  - For GraphDef (.pb) files without a SavedModel, you may need to specify
    --inputs and --outputs explicitly. Uses opset 17 by default for better
    operator coverage (Erfc and other ops require opset 13+).

Output: <same directory>/<same stem>.onnx
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def convert(pb_path: Path, opset: int, output: Path | None, inputs: str | None = None, outputs: str | None = None) -> Path:
    out = output or pb_path.with_suffix(".onnx")

    # Check if a corresponding SavedModel directory exists (preferred — no manual I/O spec needed).
    saved_model_dir = pb_path.with_suffix(".savedmodel")
    use_saved_model = saved_model_dir.is_dir()
    
    # tf2onnx CLI is the most reliable path — avoids API version skew.
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--output", str(out),
        "--opset", str(opset),
    ]
    
    if use_saved_model:
        # SavedModel format doesn't require manual input/output discovery.
        cmd.extend(["--saved-model", str(saved_model_dir)])
        print(f"Using SavedModel: {saved_model_dir}")
    else:
        # GraphDef format — explicit inputs/outputs may be required.
        cmd.extend(["--input", str(pb_path)])
        if inputs:
            cmd.extend(["--inputs", inputs])
        if outputs:
            cmd.extend(["--outputs", outputs])

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        # Provide troubleshooting hints based on the format being used.
        print(
            "\n[hint] Conversion failed. Troubleshooting:\n"
            "  1. For GraphDef (.pb) models, try providing explicit inputs/outputs:\n"
            "     python bench/convert_to_onnx.py bench/models/bert_model.pb \\\n"
            "       --inputs input_ids:0,attention_mask:0,token_type_ids:0 \\\n"
            "       --outputs Identity:0\n"
            "  2. If a corresponding .savedmodel directory exists, it will be used automatically.\n"
            "     (SavedModel format does not require manual input/output specification.)\n",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\nOK  {out}  ({size_mb:.1f} MB)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Convert frozen .pb to ONNX")
    p.add_argument("model", help="Path to frozen .pb file")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version (default: 17)")
    p.add_argument("--output", help="Output .onnx path (default: same dir as input)")
    p.add_argument("--inputs", help="Comma-separated input names (e.g., input_ids:0,attention_mask:0,token_type_ids:0)")
    p.add_argument("--outputs", help="Comma-separated output names (e.g., Identity:0)")
    args = p.parse_args()

    pb = Path(args.model).resolve()
    if not pb.exists():
        sys.exit(f"Not found: {pb}")
    if pb.suffix != ".pb":
        sys.exit(f"Expected a .pb file, got: {pb}")

    out = Path(args.output).resolve() if args.output else None
    convert(pb, args.opset, out, args.inputs, args.outputs)


if __name__ == "__main__":
    main()