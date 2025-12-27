#!/usr/bin/env python3
"""
Calibrate tower HP regions

This script visualizes HP regions on the full screenshot with labeled boxes.
This makes it easy to see if the regions are correctly positioned.

Usage:
    python3 scripts/calibrate_tower_hp.py
    python3 scripts/calibrate_tower_hp.py --screenshot path/to/screenshot.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np
from detection.tower_hp_detector import TowerHPDetector


def calibrate_hp_regions(screenshot_path: str = None):
    """
    Visualize HP regions on screenshot with labeled boxes

    Args:
        screenshot_path: Path to screenshot, or None to use default from training data
    """
    print("\n" + "="*60)
    print("Tower HP Region Calibration")
    print("="*60)

    # Find a battle screenshot if not provided
    if screenshot_path is None:
        import glob
        screenshots = glob.glob("training_data/*/screenshot_*IN_BATTLE.png")
        if not screenshots:
            print("❌ No battle screenshots found in training_data/")
            print("   Run the bot with --screenshots flag to collect training data")
            return

        # Sort screenshots and use a later one (more likely to have king tower damage)
        screenshots.sort()
        # Try to use a screenshot from very late in the battle (e.g., 95% through the list)
        # to maximize chance of seeing enemy king activated
        idx = min(int(len(screenshots) * 0.95), len(screenshots) - 1)
        screenshot_path = screenshots[idx]
        print(f"\nUsing screenshot: {screenshot_path}")
        print(f"  (Screenshot {idx+1}/{len(screenshots)} - later screenshots more likely to show king tower HP)")

    # Load screenshot
    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"❌ Failed to load screenshot: {screenshot_path}")
        return

    print(f"✅ Screenshot loaded ({screenshot.shape[1]}x{screenshot.shape[0]})")

    # Initialize detector
    detector = TowerHPDetector()

    # Create annotated screenshot with boxes
    annotated = screenshot.copy()

    # Colors for different towers
    enemy_color = (0, 0, 255)  # Red for enemy
    ally_color = (255, 0, 0)   # Blue for ally

    for tower_name, (x1, y1, x2, y2) in detector.TOWER_REGIONS.items():
        # Choose color based on tower type
        color = enemy_color if 'enemy' in tower_name else ally_color

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Add label
        label = tower_name.replace('_', ' ').title()
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Position label above the box
        label_y = y1 - 5 if y1 > 20 else y2 + 15
        cv2.putText(annotated, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Save annotated screenshot
    output_path = "tower_hp_regions_visualized.png"
    cv2.imwrite(output_path, annotated)

    # Also save the cropped regions for detailed inspection
    detector.save_hp_regions(screenshot, output_dir="tower_hp_crops")

    print("\n" + "="*60)
    print("✅ Visualization complete!")
    print("="*60)
    print(f"\n📸 Full screenshot with boxes: {output_path}")
    print(f"📁 Cropped regions: tower_hp_crops/")
    print("\nNext steps:")
    print(f"1. Open {output_path} to see all HP regions at once")
    print("2. Check if boxes are positioned correctly around HP numbers")
    print("3. If not, adjust coordinates in detection/tower_hp_detector.py")
    print("\nCurrent regions (in tower_hp_detector.py):")
    print("-"*60)

    for tower, region in detector.TOWER_REGIONS.items():
        print(f"'{tower}': {region},")

    print("-"*60)
    print("\nFormat: (x1, y1, x2, y2)")
    print("  - x1, y1: Top-left corner")
    print("  - x2, y2: Bottom-right corner")
    print("\nColor coding:")
    print("  🔴 Red boxes: Enemy towers")
    print("  🔵 Blue boxes: Ally towers")


def main():
    parser = argparse.ArgumentParser(description="Calibrate tower HP regions")
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Path to battle screenshot (default: auto-find from training_data)"
    )

    args = parser.parse_args()

    calibrate_hp_regions(args.screenshot)


if __name__ == "__main__":
    main()
