#!/usr/bin/env python3
"""
Export fine-tuned ByT5 model to ONNX for faster inference.

Uses HuggingFace Optimum for encoder-decoder ONNX export.

Usage:
    python export_onnx.py --model_path outputs/byt5-typo/best --output_dir outputs/byt5-typo-onnx

Expected speed-up: 2-4x on CPU, 1.5-2x on GPU vs PyTorch.
"""

import argparse
import time
from pathlib import Path


def export_to_onnx(model_path: str, output_dir: str):
    """Export a ByT5/T5 model to ONNX using optimum."""
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    print(f"  Exporting {model_path} -> {output_dir} ...")

    # Export (this also downloads/loads the model)
    model = ORTModelForSeq2SeqLM.from_pretrained(model_path, export=True)
    model.save_pretrained(output_dir)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_dir)

    print(f"  ONNX model saved to {output_dir}")

    # Validate
    print("\n  Running validation ...")
    test_queries = [
        "correct: iphnoe 15 pro",
        "correct: samsng galxy s24",
        "correct: macbok air m2",
    ]

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(output_dir)

    for q in test_queries:
        start = time.perf_counter()
        inputs = tokenizer(q, return_tensors="pt")
        outputs = ort_model.generate(**inputs, max_length=128, num_beams=2)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"    '{q}' -> '{result}'  ({elapsed:.0f}ms)")

    print("\n  Export complete!")


def main():
    parser = argparse.ArgumentParser(description="Export ByT5 to ONNX")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="ONNX output directory (default: <model_path>-onnx)")
    args = parser.parse_args()

    output_dir = args.output_dir or f"{args.model_path}-onnx"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    export_to_onnx(args.model_path, output_dir)


if __name__ == "__main__":
    main()
