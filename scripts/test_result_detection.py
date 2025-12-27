#!/usr/bin/env python3
"""
Test battle result detection

This script tests the BattleResultDetector on the current screen or a saved screenshot.

Usage:
    # Test on current BlueStacks screen:
    python3 scripts/test_result_detection.py

    # Test on a specific screenshot:
    python3 scripts/test_result_detection.py --image path/to/screenshot.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import cv2
from controllers.game_controller import GameController
from detection.battle_result_detector import BattleResultDetector


def test_detection(screenshot_path: str = None, instance_id: int = 0):
    """
    Test result detection on a screenshot

    Args:
        screenshot_path: Path to screenshot, or None to capture from BlueStacks
        instance_id: BlueStacks instance ID
    """
    print("\n" + "="*60)
    print("Battle Result Detection Test")
    print("="*60)

    # Load screenshot
    if screenshot_path:
        print(f"\nLoading screenshot: {screenshot_path}")
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print(f"❌ Failed to load screenshot: {screenshot_path}")
            return
    else:
        print("\nCapturing screenshot from BlueStacks...")
        gc = GameController(instance_id)
        screenshot = gc.take_screenshot()
        if screenshot is None:
            print("❌ Failed to capture screenshot")
            return

    print(f"✅ Screenshot loaded ({screenshot.shape[1]}x{screenshot.shape[0]})")

    # Initialize detector
    detector = BattleResultDetector()

    # Test detection
    print("\nTesting result detection...")
    result = detector.detect_result(screenshot, threshold=0.7, verbose=True)

    print("\n" + "="*60)
    if result:
        print(f"✅ RESULT DETECTED: {result.upper()}")
    else:
        print("❌ No result detected")
        print("\nPossible reasons:")
        print("1. Not on a battle result screen")
        print("2. No templates created yet (run create_result_templates.py first)")
        print("3. Template threshold too high (try lower threshold)")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Test battle result detection")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to screenshot to test (default: capture from BlueStacks)"
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=0,
        help="BlueStacks instance ID"
    )

    args = parser.parse_args()

    test_detection(args.image, args.instance)


if __name__ == "__main__":
    main()
