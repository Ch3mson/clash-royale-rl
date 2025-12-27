#!/usr/bin/env python3
"""
Helper script to create battle result templates

This script helps you create templates for victory, defeat, and draw screens:
1. Connect to BlueStacks and capture screenshots
2. Save cropped regions of result text/icons
3. Templates are used by BattleResultDetector for automatic detection

Usage:
    python3 scripts/create_result_templates.py --type victory
    python3 scripts/create_result_templates.py --type defeat
    python3 scripts/create_result_templates.py --type draw

Instructions:
1. Start a battle in Clash Royale
2. When the battle ends with the desired result, run this script
3. The script will capture and save the result template
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import cv2
from controllers.game_controller import GameController
from detection.battle_result_detector import BattleResultDetector


def create_template(result_type: str, instance_id: int = 0, variant_num: int = None):
    """
    Create a result template from current screen

    Args:
        result_type: Type of result ('victory', 'defeat', 'draw')
        instance_id: BlueStacks instance ID
        variant_num: Optional variant number (e.g., 1, 2, 3 for multiple templates)
    """
    variant_str = f" (variant {variant_num})" if variant_num is not None else ""
    print(f"\nCreating {result_type} template{variant_str}...")
    print("="*60)

    # Connect to game
    gc = GameController(instance_id)
    detector = BattleResultDetector()

    # Capture screenshot
    screenshot = gc.take_screenshot()
    if screenshot is None:
        print("❌ Failed to capture screenshot")
        return

    print(f"✅ Screenshot captured ({screenshot.shape[1]}x{screenshot.shape[0]})")

    # Save template
    detector.save_result_template(screenshot, result_type, variant_num=variant_num)

    print(f"\n✅ {result_type.capitalize()} template{variant_str} created successfully!")
    print("="*60)
    print("\nNext steps:")
    print(f"1. Verify the template looks correct in detection/result_templates/")
    print(f"2. Create more variants if needed: --variant 2, --variant 3, etc.")
    print(f"3. Test detection with: python3 scripts/test_result_detection.py")


def main():
    parser = argparse.ArgumentParser(description="Create battle result templates")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=['victory', 'defeat', 'draw'],
        help="Type of result screen to capture"
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=0,
        help="BlueStacks instance ID"
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=None,
        help="Variant number for multiple templates (e.g., 1, 2, 3)"
    )

    args = parser.parse_args()

    variant_str = f" variant {args.variant}" if args.variant is not None else ""
    print("\n" + "="*60)
    print(f"Battle Result Template Creator")
    print("="*60)
    print(f"\nCapturing {args.type.upper()}{variant_str} screen...")
    print("Make sure the game is showing the result screen right now!")
    print("\nPress Enter when ready (Ctrl+C to cancel)...")
    input()

    create_template(args.type, args.instance, args.variant)


if __name__ == "__main__":
    main()
