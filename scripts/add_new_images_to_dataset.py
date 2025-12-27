#!/usr/bin/env python3
"""
Add newly classified images from training sessions to the dataset

This script:
1. Finds all manually classified images in training_data/*/unclassified_cards/
2. Normalizes card names (e.g., "mini pekka" -> "mini_pekka")
3. Adds them to the dataset with proper train/val/test split
4. Skips "empty" class (not a real card)

Usage:
    python3 scripts/add_new_images_to_dataset.py
"""

import os
import shutil
from pathlib import Path
import random


def normalize_card_name(name: str) -> str:
    """Normalize card name to match card_info.py format"""
    # Convert to lowercase and replace spaces with underscores
    normalized = name.lower().strip().replace(' ', '_')

    # Special case mappings
    mappings = {
        'electron_spirit': 'electro_spirit',
        'mini_pekka': 'mini_pekka',
        'mega_minion': 'mega_minion',
    }

    return mappings.get(normalized, normalized)


def add_images_to_dataset(
    training_data_dir: str = "training_data",
    dataset_dir: str = "datasets/card_hand_classifier",
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
    skip_classes: list = ['empty']
):
    """
    Add newly classified images to the dataset

    Args:
        training_data_dir: Directory containing session folders
        dataset_dir: Target dataset directory
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        skip_classes: Card types to skip (e.g., 'empty')
    """
    training_path = Path(training_data_dir)
    dataset_path = Path(dataset_dir)

    # Find all classified card folders across all sessions
    card_images = {}

    for session_dir in training_path.glob("session_*/unclassified_cards"):
        if not session_dir.is_dir():
            continue

        # Look for card folders (manually organized)
        for card_folder in session_dir.iterdir():
            if not card_folder.is_dir():
                continue

            card_name = card_folder.name

            # Skip empty class
            if card_name in skip_classes:
                print(f"⏭️  Skipping '{card_name}' (in skip list)")
                continue

            # Normalize card name
            normalized_name = normalize_card_name(card_name)

            if normalized_name not in card_images:
                card_images[normalized_name] = []

            # Collect all images for this card
            for img_file in card_folder.glob("*.png"):
                card_images[normalized_name].append(img_file)

    if not card_images:
        print("❌ No classified images found!")
        print(f"   Looking in: {training_path}/session_*/unclassified_cards/<card_name>/")
        print("   Make sure you've organized images into card folders")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(card_images)} card types:")
    print(f"{'='*60}")

    total_added = 0

    for card_name, images in sorted(card_images.items()):
        print(f"\n📦 {card_name}: {len(images)} new images")

        # Shuffle for random split
        random.shuffle(images)

        # Calculate split sizes
        n_images = len(images)
        n_train = max(1, int(n_images * train_split))
        n_val = max(1, int(n_images * val_split))
        # Remaining goes to test

        # Ensure we don't over-allocate
        n_test = n_images - n_train - n_val
        if n_test < 0:
            n_val = max(1, n_images - n_train - 1)
            n_test = n_images - n_train - n_val

        # Create card directories in each split if they don't exist
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / split / card_name
            split_dir.mkdir(parents=True, exist_ok=True)

        # Get existing image count for numbering
        train_dir = dataset_path / 'train' / card_name
        existing_count = len(list(train_dir.glob("*.png")))

        # Copy training images
        copied_train = 0
        for img_file in images[:n_train]:
            dst = dataset_path / 'train' / card_name / f'{card_name}_{existing_count + copied_train:03d}.png'
            shutil.copy2(img_file, dst)
            copied_train += 1

        # Copy validation images
        val_dir = dataset_path / 'val' / card_name
        existing_val_count = len(list(val_dir.glob("*.png")))
        copied_val = 0
        for img_file in images[n_train:n_train+n_val]:
            dst = dataset_path / 'val' / card_name / f'{card_name}_{existing_val_count + copied_val:03d}.png'
            shutil.copy2(img_file, dst)
            copied_val += 1

        # Copy test images
        test_dir = dataset_path / 'test' / card_name
        existing_test_count = len(list(test_dir.glob("*.png")))
        copied_test = 0
        for img_file in images[n_train+n_val:]:
            dst = dataset_path / 'test' / card_name / f'{card_name}_{existing_test_count + copied_test:03d}.png'
            shutil.copy2(img_file, dst)
            copied_test += 1

        print(f"   ✅ Added {copied_train} train, {copied_val} val, {copied_test} test")
        print(f"   📊 Total now: {existing_count + copied_train} train, {existing_val_count + copied_val} val, {existing_test_count + copied_test} test")

        total_added += len(images)

    print(f"\n{'='*60}")
    print(f"✅ Successfully added {total_added} images to dataset!")
    print(f"{'='*60}")
    print(f"\nDataset location: {dataset_path}")
    print(f"\nNext steps:")
    print(f"1. Zip the dataset:")
    print(f"   cd {dataset_path.parent}")
    print(f"   zip -r {dataset_path.name}.zip {dataset_path.name}/")
    print(f"2. Upload to Google Drive and train on Colab")
    print(f"   See TRAIN_CARD_CLASSIFIER.md for instructions")


if __name__ == "__main__":
    add_images_to_dataset()
