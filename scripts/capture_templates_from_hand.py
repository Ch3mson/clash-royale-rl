#!/usr/bin/env python3
"""
Capture card templates directly from the detection boxes
This ensures perfect alignment between templates and detection
"""
import cv2
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detection.card_hand_detector import CardHandDetector

def main():
    print("="*60)
    print("Card Template Capture Tool")
    print("="*60)
    print("\nThis tool will help you capture templates from screenshots.")
    print("You'll need to identify which card is in each slot.\n")

    # Find a training screenshot
    training_dir = Path("training_data")
    sessions = sorted(training_dir.glob("session_*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not sessions:
        print("❌ No training screenshots found!")
        print("Run: python3 main.py --instance 0 --games 1")
        return

    screenshots = list(sessions[0].glob("screenshot_*_IN_BATTLE.png"))
    if not screenshots:
        print("❌ No IN_BATTLE screenshots found!")
        return

    screenshot_path = screenshots[0]
    print(f"Using screenshot: {screenshot_path.name}\n")

    # Load screenshot
    screenshot = cv2.imread(str(screenshot_path))
    if screenshot is None:
        print("❌ Failed to load screenshot")
        return

    # Initialize detector to get card slots
    detector = CardHandDetector()

    # Create templates directory
    templates_dir = Path("detection/card_templates")
    templates_dir.mkdir(exist_ok=True, parents=True)

    print("Card types in your deck:")
    for i, card_name in enumerate(sorted(detector.card_types.keys()), 1):
        print(f"  {i}. {card_name}")

    print("\n" + "="*60)
    print("Now I'll show you each card slot crop.")
    print("="*60)

    # Save crops for manual inspection
    print("\nSaving card crops...")
    for slot_idx, (x1, y1, x2, y2) in enumerate(detector.CARD_SLOTS):
        card_img = screenshot[y1:y2, x1:x2]
        crop_path = f"temp_slot_{slot_idx}.png"
        cv2.imwrite(crop_path, card_img)
        print(f"  Slot {slot_idx}: {crop_path}")

    print("\n" + "="*60)
    print("Instructions:")
    print("="*60)
    print("1. Look at the temp_slot_*.png files to see what cards are in each slot")
    print("2. For each slot, identify which card it is from the list above")
    print("3. Run this command for each slot:")
    print("   python3 -c \"import cv2; img=cv2.imread('temp_slot_X.png'); cv2.imwrite('detection/card_templates/CARDNAME.png', img)\"")
    print("   (replace X with slot number and CARDNAME with the card name)")
    print("\nExample:")
    print("   If slot 0 contains 'skeletons':")
    print("   python3 -c \"import cv2; img=cv2.imread('temp_slot_0.png'); cv2.imwrite('detection/card_templates/skeletons.png', img)\"")
    print("\n" + "="*60)
    print("OR - Easier method:")
    print("="*60)
    print("Tell me which card is in each slot and I'll copy them for you!")
    print("\nFor example: 'slot 0 is skeletons, slot 1 is giant, ...'")

if __name__ == "__main__":
    main()
