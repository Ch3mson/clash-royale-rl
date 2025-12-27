#!/usr/bin/env python3
"""
Evaluate YOLOv8 card hand classifier performance

Tests the trained classifier on the test set and reports:
- Overall accuracy
- Per-card accuracy
- Confusion matrix
- Misclassified examples

Usage:
    python3 scripts/evaluate_card_classifier.py
    python3 scripts/evaluate_card_classifier.py --model models/card_hand_classifier.pt
    python3 scripts/evaluate_card_classifier.py --data datasets/card_hand_classifier
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO


def evaluate_classifier(
    model_path: str = "models/card_hand_classifier.pt",
    data_path: str = "datasets/card_hand_classifier/test",
    threshold: float = 0.5
):
    """
    Evaluate card classifier on test set

    Args:
        model_path: Path to trained model
        data_path: Path to test dataset
        threshold: Minimum confidence threshold
    """
    model_file = Path(model_path)
    test_dir = Path(data_path)

    if not model_file.exists():
        print(f"❌ Error: Model not found at '{model_path}'")
        print("   Train a model first using TRAIN_CARD_CLASSIFIER.md")
        return

    if not test_dir.exists():
        print(f"❌ Error: Test data not found at '{data_path}'")
        print("   Create dataset first using scripts/split_dataset.py")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Get all card directories
    card_dirs = [d for d in test_dir.iterdir() if d.is_dir()]

    if not card_dirs:
        print(f"❌ Error: No card directories found in '{test_dir}'")
        return

    print(f"Found {len(card_dirs)} card types in test set")
    print(f"Confidence threshold: {threshold:.2f}\n")

    # Track results
    total_correct = 0
    total_images = 0
    per_card_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    misclassified = []

    # Test each card
    for card_dir in sorted(card_dirs):
        true_label = card_dir.name

        # Get all test images for this card
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            image_files.extend(list(card_dir.glob(f'*{ext}')))

        for img_file in image_files:
            # Load image
            img = cv2.imread(str(img_file))

            if img is None:
                print(f"⚠️  Warning: Could not load {img_file}")
                continue

            # Run classifier
            results = model(img, verbose=False)

            if len(results) > 0:
                probs = results[0].probs
                predicted_label = model.names[int(probs.top1)]
                confidence = float(probs.top1conf)

                # Check if correct
                is_correct = (predicted_label == true_label and confidence >= threshold)

                if is_correct:
                    total_correct += 1
                    per_card_stats[true_label]['correct'] += 1
                else:
                    # Record misclassification
                    misclassified.append({
                        'file': img_file.name,
                        'true': true_label,
                        'predicted': predicted_label,
                        'confidence': confidence
                    })

                total_images += 1
                per_card_stats[true_label]['total'] += 1
                per_card_stats[true_label]['confidences'].append(confidence)

    # Print results
    print(f"{'='*80}")
    print(f"OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"Total images tested: {total_images}")
    print(f"Correct predictions: {total_correct}")
    print(f"Overall accuracy:    {total_correct/total_images*100:.2f}%\n")

    print(f"{'='*80}")
    print(f"PER-CARD ACCURACY")
    print(f"{'='*80}")
    print(f"{'Card Name':<25s} {'Accuracy':<12s} {'Avg Conf':<12s} {'Samples':<10s}")
    print(f"{'-'*80}")

    for card_name in sorted(per_card_stats.keys()):
        stats = per_card_stats[card_name]
        accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0

        # Color code by accuracy
        if accuracy >= 95:
            status = "✅"
        elif accuracy >= 80:
            status = "⚠️ "
        else:
            status = "❌"

        print(f"{status} {card_name:<23s} {accuracy:>5.1f}%      {avg_conf:>5.3f}       {stats['total']:<10d}")

    # Show misclassifications
    if misclassified:
        print(f"\n{'='*80}")
        print(f"MISCLASSIFIED EXAMPLES ({len(misclassified)} total)")
        print(f"{'='*80}")

        # Show first 20 misclassifications
        for item in misclassified[:20]:
            print(f"  {item['file']:<40s} True: {item['true']:<15s} → Predicted: {item['predicted']:<15s} (conf={item['confidence']:.2f})")

        if len(misclassified) > 20:
            print(f"  ... and {len(misclassified) - 20} more")

    print(f"\n{'='*80}")

    # Performance assessment
    overall_accuracy = total_correct / total_images * 100

    if overall_accuracy >= 95:
        print("🎉 Excellent! Model is ready for deployment.")
    elif overall_accuracy >= 90:
        print("✅ Good performance. Consider collecting more data for cards with low accuracy.")
    elif overall_accuracy >= 80:
        print("⚠️  Acceptable but could be improved. Collect more training data or use larger model.")
    else:
        print("❌ Poor performance. You need more training data or a different approach.")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate card hand classifier')
    parser.add_argument('--model', type=str, default='models/card_hand_classifier.pt',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='datasets/card_hand_classifier/test',
                       help='Path to test dataset')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold (0-1)')

    args = parser.parse_args()

    evaluate_classifier(
        model_path=args.model,
        data_path=args.data,
        threshold=args.threshold
    )
