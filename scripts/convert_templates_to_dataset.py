#!/usr/bin/env python3
"""
Convert existing card templates into a YOLOv8 classification dataset

This script takes your existing card template images and organizes them
into the format required for YOLOv8 image classification training.

Usage:
    python3 scripts/convert_templates_to_dataset.py
"""

import os
import shutil
from pathlib import Path
import random


def convert_templates_to_dataset(
    template_dir: str = "detection/card_templates",
    output_dir: str = "datasets/card_hand_classifier",
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1
):
    """
    Convert card templates to YOLOv8 classification dataset

    Args:
        template_dir: Directory containing card template images
        output_dir: Output directory for organized dataset
        train_split: Fraction of images for training
        val_split: Fraction of images for validation
        test_split: Fraction of images for testing
    """
    template_path = Path(template_dir)
    output_path = Path(output_dir)

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Group templates by card name
    card_templates = {}

    for file in os.listdir(template_path):
        if not (file.endswith('.png') or file.endswith('.jpg')):
            continue

        # Skip grayed out templates
        if '_grayed' in file:
            continue

        # Extract card name (remove variant number and extension)
        base_name = file.replace('.png', '').replace('.jpg', '')

        # Remove variant number (e.g., cannon_1 -> cannon)
        if '_' in base_name and base_name.split('_')[-1].isdigit():
            card_name = '_'.join(base_name.split('_')[:-1])
        else:
            card_name = base_name

        # Add to card templates dict
        if card_name not in card_templates:
            card_templates[card_name] = []

        card_templates[card_name].append(template_path / file)

    print(f"Found {len(card_templates)} unique cards")
    print(f"Total template images: {sum(len(imgs) for imgs in card_templates.values())}")

    # Split and copy images for each card
    total_train = 0
    total_val = 0
    total_test = 0

    for card_name, image_files in card_templates.items():
        # Shuffle images for random split
        random.shuffle(image_files)

        n_images = len(image_files)
        n_train = int(n_images * train_split)
        n_val = int(n_images * val_split)
        # Remaining goes to test

        # Create card directories in each split
        for split in ['train', 'val', 'test']:
            (output_path / split / card_name).mkdir(exist_ok=True)

        # Copy training images
        for i, img_file in enumerate(image_files[:n_train]):
            dst = output_path / 'train' / card_name / f'{card_name}_{i:03d}.png'
            shutil.copy2(img_file, dst)
            total_train += 1

        # Copy validation images
        for i, img_file in enumerate(image_files[n_train:n_train+n_val]):
            dst = output_path / 'val' / card_name / f'{card_name}_{i:03d}.png'
            shutil.copy2(img_file, dst)
            total_val += 1

        # Copy test images
        for i, img_file in enumerate(image_files[n_train+n_val:]):
            dst = output_path / 'test' / card_name / f'{card_name}_{i:03d}.png'
            shutil.copy2(img_file, dst)
            total_test += 1

        print(f"  {card_name:20s}: {n_images} images -> train={n_train}, val={n_val}, test={n_images-n_train-n_val}")

    print(f"\n{'='*60}")
    print(f"Dataset created at: {output_path}")
    print(f"{'='*60}")
    print(f"Training images:   {total_train}")
    print(f"Validation images: {total_val}")
    print(f"Test images:       {total_test}")
    print(f"Total images:      {total_train + total_val + total_test}")
    print(f"\nNext step: Train on Google Colab using TRAIN_CARD_CLASSIFIER.md")


if __name__ == "__main__":
    convert_templates_to_dataset()
