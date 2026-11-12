"""Run upstream classification eval with the project's local settings."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch

from classifier_checkpoints import (
    DATA_DIR,
    DEFAULT_CHECKPOINT,
    INPUT_SIZE,
    NUM_CLASSES,
    PROJECT_DIR,
    UPSTREAM_DIR,
    checkpoint_globals,
)

sys.path.insert(0, str(UPSTREAM_DIR))
import image_finetune


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-path", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "output" / "classification_eval")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    options = parser.parse_args()
    checkpoint = options.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Classifier checkpoint not found: {checkpoint}")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment.")

    upstream_flags = [
        "--eval", "--resume", str(checkpoint),
        "--data_path", str(options.data_path.resolve()),
        "--output_dir", str(options.output_dir.resolve()),
        "--model", "vit_base_patch16", "--cls_token",
        "--nb_classes", str(NUM_CLASSES), "--input_size", str(INPUT_SIZE),
        "--batch_size", "16", "--accum_iter", "8",
        "--num_workers", "0", "--device", options.device,
    ]
    args = image_finetune.get_args_parser().parse_args(upstream_flags)
    upstream_evaluate = image_finetune.evaluate
    result = None

    def record_evaluation(loader, model, device, epoch, log_writer=None):
        nonlocal result
        metrics = upstream_evaluate(loader, model, device, epoch, log_writer)
        result = {
            "checkpoint": str(checkpoint),
            "validation_images": len(loader.dataset),
            "top1_pct": metrics["acc1"],
            "top5_pct": metrics["acc5"],
            "loss": metrics["loss"],
            "upstream_flags": upstream_flags,
        }
        # Finish the eval branch before its summary uses the effective training batch size.
        raise SystemExit(0)

    # The upstream resume helper leaves weights_only implicit.
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    image_finetune.evaluate = record_evaluation
    try:
        with checkpoint_globals(checkpoint):
            image_finetune.main(args)
    except SystemExit as exc:
        if exc.code != 0:
            raise
    except pickle.UnpicklingError as exc:
        parser.exit(1, f"Restricted checkpoint loading failed for {checkpoint}: {exc}\n")
    finally:
        image_finetune.evaluate = upstream_evaluate

    if result is None:
        parser.exit(1, "Upstream evaluation did not produce metrics.\n")
    result_path = Path(args.output_dir) / "validation.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Validation images evaluated: {result['validation_images']}. Metrics: {result_path}")


if __name__ == "__main__":
    main()
