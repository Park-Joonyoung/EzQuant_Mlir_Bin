#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

for file in *.tflite; do
  [[ -f "$file" ]] || continue
  base=$(basename "$file" .tflite)

  echo "▶ Converting $base.tflite…"
  "./flatbuffer_translate" --tflite-flatbuffer-to-mlir "$file" \
    > "${base}.mlir"
  "./tf-opt" -- --bias-contract "${base}.mlir" \
    > "${base}_modified.mlir"
  "./tf-opt" -- --dequant-contract "${base}.mlir" \
    > "${base}_modified.mlir"
  "./tf-opt" -- --dequant-split "${base}.mlir" \
    > "${base}_modified.mlir"
  "./tf-opt" -- --reshape-to-4D "${base}.mlir" \
    > "${base}_modified.mlir"
  "./flatbuffer_translate" --mlir-to-tflite-flatbuffer \
    "${base}_modified.mlir" \
    > "${base}_modified.tflite"

  echo "✔ Done: ${base}_modified.tflite"
done