#!/usr/bin/env python3
"""
Test card hand detection with existing screenshots
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detection.card_hand_detector import CardHandDetector
import cv2

def main():
    print("="*60)
    print("Card Hand Detection Test")
    print("="*60)

    # Initialize detector
    detector = CardHandDetector()
    print()

    # Find a training screenshot
    training_dir = Path("training_data")
    sessions = sorted(training_dir.glob("session_*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not sessions:
        print("❌ No training screenshots found!")
        print("Run: python3 main.py --instance 0 --games 1 --screenshots")
        return

    # Get a screenshot from the most recent session
    screenshots = list(sessions[0].glob("screenshot_*_IN_BATTLE.png"))
    if not screenshots:
        print("❌ No IN_BATTLE screenshots found!")
        return

    screenshot_path = screenshots[0]
    print(f"Using screenshot: {screenshot_path.name}")

    # Load screenshot
    screenshot = cv2.imread(str(screenshot_path))
    if screenshot is None:
        print("❌ Failed to load screenshot")
        return

    print(f"Screenshot size: {screenshot.shape[1]}x{screenshot.shape[0]}\n")

    # Detect hand
    hand = detector.get_hand(screenshot, verbose=True)

    # Summary
    detected_cards = [c for c in hand if c is not None]
    print(f"\nDetected {len(detected_cards)}/4 cards")

    if len(detected_cards) == 0:
        print("\n⚠️  No cards detected!")
        print("This might mean:")
        print("  1. The card slot coordinates are wrong")
        print("  2. The template images don't match the in-game cards")
        print("  3. The confidence threshold is too high")
        print("\nTry running: python3 scripts/fix_card_coordinates.py")
    elif len(detected_cards) < 4:
        print(f"\n⚠️  Only detected {len(detected_cards)}/4 cards")
        print("You may need to:")
        print("  1. Capture better template images")
        print("  2. Lower the confidence threshold")
    else:
        print("\n✓ All cards detected successfully!")


if __name__ == "__main__":
    main()
