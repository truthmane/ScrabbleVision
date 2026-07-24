"""Honest evaluation: per-class precision/recall on a held-out directory,
plus overall accuracy and a confusion list. Never point this at a split
that shares source tiles OR source frames/games with training data --
see docs/classifier-accuracy-plan.md's WS5 for why that distinction matters.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from training.classify.infer import TileClassifierModel


def evaluate(checkpoint_path: Path, val_dir: Path) -> Dict:
    clf = TileClassifierModel(checkpoint_path, device="cpu")

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    confusions: List[Tuple[str, str, float, str]] = []
    correct = 0
    total = 0

    for class_dir in sorted(val_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        true_label = class_dir.name
        for f in sorted(class_dir.glob("*.png")):
            crop = cv2.imread(str(f))
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pred, conf = clf.predict(crop_rgb)
            total += 1
            if pred == true_label:
                correct += 1
                tp[true_label] += 1
            else:
                fp[pred] += 1
                fn[true_label] += 1
                confusions.append((true_label, pred, conf, f.name))

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "tp": tp, "fp": fp, "fn": fn,
        "confusions": confusions,
        "classes": clf.classes,
    }


def print_report(result: Dict) -> None:
    print(f"Overall accuracy: {result['correct']}/{result['total']} = {result['accuracy']*100:.1f}%\n")
    print(f"{'class':8s} {'precision':>10s} {'recall':>8s} {'support':>8s}")
    for label in sorted(result["classes"]):
        tp, fp, fn = result["tp"][label], result["fp"][label], result["fn"][label]
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        note = "  <-- no support in val" if support == 0 else ""
        print(f"{label:8s} {precision:10.2f} {recall:8.2f} {support:8d}{note}")

    print("\nConfusions (true -> predicted, confidence):")
    for true_l, pred_l, conf, fname in sorted(result["confusions"]):
        print(f"  {true_l:6s} -> {pred_l:6s}  ({conf:.2f})  {fname}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("val_dir", type=Path)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.val_dir)
    print_report(result)


if __name__ == "__main__":
    main()
