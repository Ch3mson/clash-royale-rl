#!/usr/bin/env python3
"""
Split card images into train/val/test sets for YOLOv8 classification

This script takes a directory of card images organized by card name
and splits them into training, validation, and test sets.

Input structure:
    raw_cards/
    ├── archers/
    │   ├── img001.png
    │   └── img002.png
    ├── wizard/
    └── ...

Output structure:
    card_hand_classifier/
    ├── train/
    │   ├── archers/
    │   ├── wizard/
    │   └── ...
    ├── val/
    │   ├── archers/
    │   └── ...
    └── test/
        ├── archers/
        └── ...

Usage:
    python3 scripts/split_dataset.py <input_dir> <output_dir>
    python3 scripts/split_dataset.py datasets/raw_cards datasets/card_hand_classifier
"""

import os
import shutil
from pathlib import Path
import random
import sys


def split_dataset(
    input_dir: str,
    output_dir: str,
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
    min_images_per_card: int = 5
):
    """
    Split card images into train/val/test sets

    Args:
        input_dir: Directory containing card folders
        output_dir: Output directory for split dataset
        train_split: Fraction for training (default 70%)
        val_split: Fraction for validation (default 20%)
        test_split: Fraction for testing (default 10%)
        min_images_per_card: Minimum images required per card
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Find all card directories
    card_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not card_dirs:
        print(f"Error: No card directories found in '{input_dir}'")
        print("Expected structure: <input_dir>/<card_name>/<images>")
        return

    print(f"Found {len(card_dirs)} card types")

    total_train = 0
    total_val = 0
    total_test = 0
    skipped_cards = []

    for card_dir in card_dirs:
        card_name = card_dir.name

        # Get all image files
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            image_files.extend(list(card_dir.glob(f'*{ext}')))

        n_images = len(image_files)

        # Skip cards with too few images
        if n_images < min_images_per_card:
            skipped_cards.append(f"{card_name} ({n_images} images)")
            continue

        # Shuffle for random split
        random.shuffle(image_files)

        # Calculate split sizes
        n_train = max(1, int(n_images * train_split))
        n_val = max(1, int(n_images * val_split))
        # Remaining goes to test (at least 1 if possible)

        # Ensure we don't over-allocate
        n_test = n_images - n_train - n_val
        if n_test < 0:
            # Adjust if total > n_images
            n_val = max(1, n_images - n_train - 1)
            n_test = n_images - n_train - n_val

        # Create card directories in each split
        for split in ['train', 'val', 'test']:
            (output_path / split / card_name).mkdir(exist_ok=True)

        # Copy training images
        for img_file in image_files[:n_train]:
            dst = output_path / 'train' / card_name / img_file.name
            shutil.copy2(img_file, dst)
            total_train += 1

        # Copy validation images
        for img_file in image_files[n_train:n_train+n_val]:
            dst = output_path / 'val' / card_name / img_file.name
            shutil.copy2(img_file, dst)
            total_val += 1

        # Copy test images
        for img_file in image_files[n_train+n_val:]:
            dst = output_path / 'test' / card_name / img_file.name
            shutil.copy2(img_file, dst)
            total_test += 1

        print(f"  {card_name:20s}: {n_images:3d} images -> train={n_train:3d}, val={n_val:3d}, test={n_test:3d}")

    print(f"\n{'='*60}")
    print(f"Dataset created at: {output_path}")
    print(f"{'='*60}")
    print(f"Training images:   {total_train}")
    print(f"Validation images: {total_val}")
    print(f"Test images:       {total_test}")
    print(f"Total images:      {total_train + total_val + total_test}")

    if skipped_cards:
        print(f"\n⚠️  Skipped {len(skipped_cards)} cards (less than {min_images_per_card} images):")
        for card in skipped_cards:
            print(f"    - {card}")

    print(f"\n✅ Next step: Zip dataset and upload to Google Drive")
    print(f"   cd {output_path.parent}")
    print(f"   zip -r {output_path.name}.zip {output_path.name}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 split_dataset.py <input_dir> <output_dir>")
        print("\nExample:")
        print("  python3 split_dataset.py datasets/raw_cards datasets/card_hand_classifier")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    split_dataset(input_dir, output_dir)
