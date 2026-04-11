#!/usr/bin/env python3
"""Generate frozen benchmark graphs (.pb) compatible with the TF C API.

Models:
  small:  MobileNetV2 (224×224×3, 1000 classes)
  medium: ResNet50    (224×224×3, 1000 classes)
  large:  Dense stack (~44M params, input 4096)

Produces properly frozen graphs (variables → constants) via
convert_variables_to_constants_v2. These are compatible with
TF_SessionRun / importGraphDef in the TF C API.

SavedModel-derived graphs (saved_model.pb) contain StatefulPartitionedCall
wrapper ops which the TF C API session cannot execute directly — never use
those as frozen graphs.

Usage:
  python bench/generate_models.py
  python bench/generate_models.py --root bench/models --models small,medium
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


# ── Model builders ────────────────────────────────────────────────────────────

def build_small() -> tf.keras.Model:
    return tf.keras.applications.MobileNetV2(
        include_top=True,
        weights=None,
        input_shape=(224, 224, 3),
        classes=1000,
    )


def build_medium() -> tf.keras.Model:
    return tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(224, 224, 3),
        classes=1000,
    )


def build_large() -> tf.keras.Model:
    # ~44M params ≈ 168 MB fp32
    x   = tf.keras.Input(shape=(4096,), name="inputs", dtype=tf.float32)
    y   = tf.keras.layers.Dense(4096, activation="gelu", name="dense_1")(x)
    y   = tf.keras.layers.Dense(4096, activation="gelu", name="dense_2")(y)
    y   = tf.keras.layers.Dense(2048, activation="gelu", name="dense_3")(y)
    out = tf.keras.layers.Dense(1000, activation=None,   name="output")(y)
    return tf.keras.Model(inputs=x, outputs=out, name="bench_large_dense")


BUILDERS: dict[str, Callable[[], tf.keras.Model]] = {
    "small":  build_small,
    "medium": build_medium,
    "large":  build_large,
}

SPECS = {
    "small":  {"frozen_pb": "bench_small.pb",  "description": "MobileNetV2 224×224×3"},
    "medium": {"frozen_pb": "bench_medium.pb", "description": "ResNet50 224×224×3"},
    "large":  {"frozen_pb": "bench_large.pb",  "description": "Dense stack ~44M params"},
}


# ── Freezing ──────────────────────────────────────────────────────────────────

def infer_batch_shape(model: tf.keras.Model) -> list[int]:
    """Return input shape with batch dim fixed to 1."""
    return [1 if d is None else int(d) for d in model.input_shape]


def freeze_model(model: tf.keras.Model, out_path: Path) -> tuple[str, list[int]]:
    """
    Freeze a Keras model to a TF C API-compatible frozen graph.

    Steps:
      1. Warm up the model so all variables are initialised.
      2. Wrap in a tf.function with a concrete input spec.
      3. Call convert_variables_to_constants_v2 — replaces VarHandleOp +
         ReadVariableOp with Const nodes, eliminating StatefulPartitionedCall.
      4. Write the frozen graph_def as a binary .pb file.

    Returns:
      (placeholder_op_name, concrete_input_shape)
    """
    batch_shape = infer_batch_shape(model)
    input_dtype = model.input.dtype

    # Warm-up: ensures all layers are built before tracing.
    model(np.zeros(batch_shape, dtype=np.float32), training=False)

    # Trace a concrete function with a fixed batch=1 shape.
    # Using a fixed batch dimension avoids unknown-rank issues in the frozen
    # graph — the TF C API allocator needs a concrete shape to size the tensor.
    @tf.function(input_signature=[
        tf.TensorSpec(batch_shape, input_dtype, name="inputs")
    ])
    def serving(x):
        return model(x, training=False)

    cf = serving.get_concrete_function()

    # Freeze: variables → constants.
    # This replaces VarHandleOp/ReadVariableOp/AssignVariableOp with Const
    # ops and removes the StatefulPartitionedCall wrapper entirely.
    frozen_cf = convert_variables_to_constants_v2(cf)
    graph_def = frozen_cf.graph.as_graph_def()

    # Discover the input op name from the frozen graph.
    placeholders = [
        op.name for op in frozen_cf.graph.get_operations()
        if op.type == "Placeholder"
    ]
    input_op = placeholders[0] if placeholders else "inputs"

    # Write binary proto.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tf.io.write_graph(graph_def, str(out_path.parent), out_path.name, as_text=False)

    return input_op, batch_shape


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     default="bench/models",
                        help="Output directory for .pb files")
    parser.add_argument("--manifest", default="bench/manifest.json",
                        help="Path to write the JSON manifest")
    parser.add_argument("--models",   default="small,medium,large",
                        help="Comma-separated list of models to generate")
    args = parser.parse_args()

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    bad   = [m for m in names if m not in BUILDERS]
    if bad:
        raise SystemExit(f"Unknown model names: {bad}. Valid: {list(BUILDERS)}")

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"generated_by": "bench/generate_models.py", "models": {}}

    for name in names:
        print(f"\n[{name}] Building model…")
        model = BUILDERS[name]()

        spec     = SPECS[name]
        out_path = root / spec["frozen_pb"]

        # Save the SavedModel directory FIRST — before freeze_model, because
        # convert_variables_to_constants_v2 modifies the model's trackable
        # graph state in a way that makes tf.saved_model.save fail afterwards.
        # benchmarks/inference_pool/tfjs_node.ts loads from this directory
        # via tf.node.loadSavedModel (the correct modern API for tfjs-node).
        sm_dir = out_path.with_suffix(".savedmodel")
        print(f"[{name}] SavedModel → {sm_dir}")
        if sm_dir.exists():
            tf.io.gfile.rmtree(str(sm_dir))
        
        if hasattr(model, "export"):
            model.export(str(sm_dir))
        else:
            model.save(str(sm_dir))

        print(f"[{name}] Freezing → {out_path}")
        input_op, batch_shape = freeze_model(model, out_path)

        params    = int(model.count_params())
        est_mb    = round((params * 4) / (1024 * 1024), 2)
        pb_bytes  = out_path.stat().st_size

        print(
            f"[{name}] OK  input_op={input_op!r}  shape={batch_shape}  "
            f"params={params:,}  fp32≈{est_mb:.1f}MB  pb={pb_bytes/1024/1024:.1f}MB"
        )

        manifest["models"][name] = {
            "frozen_pb":         spec["frozen_pb"],
            "savedmodel_dir":    sm_dir.name,
            "input_op":          input_op,
            "input_shape":       batch_shape,
            "params":            params,
            "estimated_fp32_mb": est_mb,
            "description":       spec["description"],
        }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest → {manifest_path}")


if __name__ == "__main__":
    main()