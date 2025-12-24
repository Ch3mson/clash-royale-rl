#!/usr/bin/env python3
"""
Helper script to capture card template images from Clash Royale
Run this while in battle to save cropped card images
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from controllers.adb_controller import ADBController
from detection.card_hand_detector import CardHandDetector
import cv2


def main():
    print("="*60)
    print("Card Template Capture Tool")
    print("="*60)
    print("\nThis tool will help you capture card images for template matching.")
    print("\nInstructions:")
    print("1. Make sure BlueStacks is running with Clash Royale open")
    print("2. Start a battle so your cards are visible at the bottom")
    print("3. Press ENTER to capture the current cards")
    print("4. The script will save cropped images of each card slot")
    print("5. You'll rename them manually to match your deck\n")

    # Initialize
    adb = ADBController(adb_port=5555)
    detector = CardHandDetector()

    # Test connection
    if not adb.test_connection():
        print("❌ ADB connection failed. Is BlueStacks running?")
        return

    print("✓ Connected to BlueStacks")

    input("\nPress ENTER when you're in battle with cards visible...")

    # Take screenshot
    screenshot = adb.screenshot()
    if screenshot is None:
        print("❌ Failed to capture screenshot")
        return

    print("✓ Screenshot captured")

    # Save crops
    output_dir = "card_crops"
    detector.save_card_crops(screenshot, output_dir)

    # Also save full screenshot for reference
    cv2.imwrite("card_crops/full_screenshot.png", screenshot)
    print("✓ Also saved full screenshot for reference")

    print(f"\n{'='*60}")
    print("Card crops saved!")
    print(f"{'='*60}")
    print(f"\nCard images saved to: {output_dir}/")
    print("\nNext steps:")
    print("1. Open the card_crops/ folder")
    print("2. Look at each card_slot_X.png file")
    print("3. Rename them to match your cards:")
    print("   - card_slot_0.png → (your leftmost card name).png")
    print("   - card_slot_1.png → (2nd card).png")
    print("   - card_slot_2.png → (3rd card).png")
    print("   - card_slot_3.png → (rightmost card).png")
    print("\n4. Move renamed files to: detection/card_templates/")
    print("\nValid card names for your deck:")
    print("  - tombstone.png")
    print("  - bomber.png")
    print("  - valkyrie.png")
    print("  - goblins.png")
    print("  - spear_goblins.png")
    print("  - cannon.png")
    print("  - giant.png")
    print("  - skeletons.png")
    print("\n5. Repeat this process until you have all 8 cards")
    print("   (play a few battles to see different cards)")


if __name__ == "__main__":
    main()
