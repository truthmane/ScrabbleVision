"""Trains the rack-tile object detector (RF-DETR) on the COCO-format
dataset `build_rack_dataset.py` produces. Meant to run on a real GPU
(RunPod or similar) for a full run -- RF-DETR is a real-time-oriented
detection transformer (Roboflow), light enough to fine-tune on a single
24GB consumer/prosumer card (RTX 4090, A5000, L4), no datacenter GPU
needed for a dataset this size.

Also runs on CPU/MPS for a quick smoke test (small dataset, 1 epoch) to
catch integration bugs -- bad category IDs, malformed boxes, missing
files -- before spending any GPU time on it.

Usage (RunPod, full run):
    python -m training.detect.train_rack_detector /workspace/rack_detect_dataset \
        --model nano --epochs 50 --device cuda --output-dir /workspace/output

Usage (local smoke test, CPU/MPS, tiny dataset):
    python -m training.detect.train_rack_detector /path/to/tiny_dataset \
        --epochs 1 --device cpu --batch-size 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def build_model(model_size: str):
    import rfdetr
    cls = getattr(rfdetr, MODEL_CLASSES[model_size])
    return cls()


def train(
    dataset_dir: Path,
    output_dir: Path,
    model_size: str = "nano",
    epochs: int = 50,
    batch_size: int = 4,
    device: str = "cuda",
    resolution: int = None,
) -> None:
    model = build_model(model_size)
    kwargs = dict(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    if resolution is not None:
        kwargs["resolution"] = resolution
    model.train(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model", choices=list(MODEL_CLASSES), default="nano")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=None)
    args = parser.parse_args()

    train(
        args.dataset_dir, args.output_dir, args.model,
        args.epochs, args.batch_size, args.device, args.resolution,
    )


if __name__ == "__main__":
    main()
