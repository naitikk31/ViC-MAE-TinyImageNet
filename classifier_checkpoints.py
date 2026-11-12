"""Restricted loading of Tiny-ImageNet classifier checkpoints."""

from argparse import Namespace
import json
from pathlib import Path
import re

from torch.serialization import safe_globals


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "tiny-imagenet-200"
UPSTREAM_DIR = PROJECT_DIR / "ViC-MAE"
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / "output" / "20260918-044542-vit_base_patch16-224" / "checkpoint-20.pth"
)
OWN_CHECKPOINTS = {DEFAULT_CHECKPOINT, DEFAULT_CHECKPOINT.with_name("checkpoint-0.pth")}
NUM_CLASSES = 200
INPUT_SIZE = 224


def checkpoint_globals(path: Path, *, allow_saved_args=False):
    """Allow the upstream argparse metadata for a selected local project checkpoint."""
    return safe_globals([Namespace] if allow_saved_args or path in OWN_CHECKPOINTS else [])


def newest_classifier_checkpoint(project_dir=PROJECT_DIR):
    """Find the latest epoch in each normal ViT-B/16 finetuning output directory."""
    output = project_dir / "output"
    directories = [project_dir, output, *output.glob("*-vit_base_patch16-224")]
    pattern = re.compile(r"checkpoint-(\d+)\.(?:pth|pt)")
    latest = []
    for directory in directories:
        epochs = []
        for path in directory.glob("checkpoint-*"):
            match = pattern.fullmatch(path.name)
            if match and path.is_file():
                epochs.append((int(match[1]), path.stat().st_mtime_ns, path))
        if epochs:
            latest.append(max(epochs))
    if not latest:
        raise FileNotFoundError(f"No Tiny-ImageNet classifier checkpoint found in {output}. Pass --checkpoint PATH.")
    return max(latest, key=lambda candidate: (candidate[1], candidate[0], str(candidate[2])))[2].resolve()


def measured_accuracy(checkpoint, project_dir=PROJECT_DIR):
    """Read the newest valid evaluation for this exact path, never by filename alone."""
    checkpoint = checkpoint.resolve()
    checkpoint_time = checkpoint.stat().st_mtime_ns
    records = set((project_dir / "output").rglob("validation.json"))
    if (sibling := checkpoint.parent / "validation.json").is_file():
        records.add(sibling)
    recent = []
    for record in records:
        try:
            modified = record.stat().st_mtime_ns
        except OSError:
            continue
        if modified >= checkpoint_time:
            recent.append((modified, record))
    for _, record in sorted(recent, reverse=True):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
            recorded_path = (record.parent / Path(data["checkpoint"])).resolve()
            first, fifth, count = data["top1_pct"], data["top5_pct"], data["validation_images"]
            if (recorded_path == checkpoint and type(first) in (int, float) and type(fifth) in (int, float)
                    and 0 <= first <= fifth <= 100 and type(count) is int and count > 0):
                return {"top1": f"{first:.2f}", "top5": f"{fifth:.2f}", "images": f"{count:,}"}
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None
