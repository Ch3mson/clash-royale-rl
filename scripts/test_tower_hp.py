#!/usr/bin/env python3
"""
Test tower HP detection

This script tests the TowerHPDetector on the current screen.
Use it to verify HP regions are correctly positioned.

Usage:
    python3 scripts/test_tower_hp.py
    python3 scripts/test_tower_hp.py --save-crops  # Save HP region images for debugging
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from controllers.game_controller import GameController
from detection.tower_hp_detector import TowerHPDetector


def test_hp_detection(save_crops: bool = False, instance_id: int = 0):
    """
    Test HP detection on current screen

    Args:
        save_crops: If True, save HP region crops for debugging
        instance_id: BlueStacks instance ID
    """
    print("\n" + "="*60)
    print("Tower HP Detection Test")
    print("="*60)

    # Connect to game
    gc = GameController(instance_id)
    detector = TowerHPDetector()

    # Capture screenshot
    print("\nCapturing screenshot from BlueStacks...")
    screenshot = gc.take_screenshot()
    if screenshot is None:
        print("❌ Failed to capture screenshot")
        return

    print(f"✅ Screenshot captured ({screenshot.shape[1]}x{screenshot.shape[0]})")

    # Save crops if requested
    if save_crops:
        print("\nSaving HP region crops...")
        detector.save_hp_regions(screenshot)

    # Test HP detection
    print("\nDetecting tower HP values...")
    print("-"*60)

    hp_values = detector.get_tower_hp(screenshot, verbose=True)

    print("-"*60)
    print("\nResults:")
    print("="*60)

    # Group by team
    enemy_towers = {k: v for k, v in hp_values.items() if k.startswith('enemy_')}
    ally_towers = {k: v for k, v in hp_values.items() if k.startswith('ally_')}

    print("\n🔴 Enemy Towers:")
    for tower, hp in enemy_towers.items():
        tower_display = tower.replace('enemy_', '').replace('_', ' ').title()
        if hp is not None:
            print(f"  {tower_display:20s}: {hp:4d} HP")
        else:
            print(f"  {tower_display:20s}: ❌ Not detected")

    print("\n🔵 Ally Towers:")
    for tower, hp in ally_towers.items():
        tower_display = tower.replace('ally_', '').replace('_', ' ').title()
        if hp is not None:
            print(f"  {tower_display:20s}: {hp:4d} HP")
        else:
            print(f"  {tower_display:20s}: ❌ Not detected")

    print("="*60)

    # Check if any HP was detected
    detected_count = sum(1 for hp in hp_values.values() if hp is not None)

    if detected_count == 0:
        print("\n⚠️  No HP values detected!")
        print("\nTroubleshooting:")
        print("1. Run with --save-crops to verify HP regions")
        print("2. Make sure you're in a battle (not main menu)")
        print("3. Check that pytesseract is installed: pip install pytesseract")
        print("4. Install Tesseract OCR: brew install tesseract (macOS)")
        print("5. Adjust TOWER_REGIONS in tower_hp_detector.py if regions are wrong")
    elif detected_count < 6:
        print(f"\n⚠️  Only {detected_count}/6 towers detected")
        print("Run with --save-crops to check which regions need adjustment")
    else:
        print(f"\n✅ All {detected_count}/6 towers detected!")


def main():
    parser = argparse.ArgumentParser(description="Test tower HP detection")
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save HP region crops for debugging"
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=0,
        help="BlueStacks instance ID"
    )

    args = parser.parse_args()

    test_hp_detection(args.save_crops, args.instance)


if __name__ == "__main__":
    main()
