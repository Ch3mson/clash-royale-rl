#!/usr/bin/env python3
"""
Capture card templates from a live game screenshot
"""
import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detection.card_hand_detector import CardHandDetector
import cv2

def main():
    print("="*60)
    print("Live Card Template Capture")
    print("="*60)
    print("\nMake sure your game is running and showing cards in battle!")
    print("Press Enter to take a screenshot...")
    input()

    # Take screenshot using BlueStacks ADB
    print("Taking screenshot...")
    adb_path = '/Applications/BlueStacks.app/Contents/MacOS/HD-Adb'
    result = subprocess.run(
        f'{adb_path} -s 127.0.0.1:5555 exec-out screencap -p',
        shell=True,
        capture_output=True,
        check=False
    )

    if result.returncode != 0:
        print(f"❌ Failed to take screenshot: {result.stderr.decode('utf-8', 'ignore')}")
        return

    # Decode screenshot
    import numpy as np
    img_array = np.frombuffer(result.stdout, dtype=np.uint8)
    screenshot = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if screenshot is None:
        print("❌ Failed to decode screenshot")
        return

    print("✓ Screenshot captured!")

    # Initialize detector to get card slots
    detector = CardHandDetector()

    print("\nSaving card crops...")
    for slot_idx, (x1, y1, x2, y2) in enumerate(detector.CARD_SLOTS):
        card_img = screenshot[y1:y2, x1:x2]
        crop_path = f"temp_slot_{slot_idx}.png"
        cv2.imwrite(crop_path, card_img)
        print(f"  Slot {slot_idx}: {crop_path}")

    print("\n" + "="*60)
    print("Card crops saved!")
    print("="*60)
    print("\nNow tell me which card is in each slot:")
    print("For example: 'slot 2 is cannon'")
    print("\nOr copy them manually with:")
    print("  python3 -c \"import cv2; img=cv2.imread('temp_slot_X.png'); cv2.imwrite('detection/card_templates/CARDNAME.png', img)\"")

if __name__ == "__main__":
    main()
