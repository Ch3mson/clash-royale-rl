#!/usr/bin/env python3
"""
Debug script to test main menu detection on a screenshot
"""
import cv2
import numpy as np
from pathlib import Path

# Load your main menu screenshot
screenshot_path = input("Enter path to main menu screenshot: ").strip()
screenshot = cv2.imread(screenshot_path)

if screenshot is None:
    print(f"Failed to load screenshot from: {screenshot_path}")
    exit(1)

h, w = screenshot.shape[:2]
print(f"Screenshot size: {w}x{h}")
print("\n" + "="*60)
print("Testing Battle Button Detection")
print("="*60)

# Test the same positions as _check_main_menu
battle_y_positions = [h - 250, h - 200, h - 150]

print("\nChecking for yellow/gold pixels in Battle button area:")
for y_pos in battle_y_positions:
    print(f"\nY position: {y_pos}")
    sample_positions = [
        (w // 2, y_pos),      # Center
        (w // 2 - 60, y_pos), # Left of center
        (w // 2 + 60, y_pos), # Right of center
    ]

    for x, y in sample_positions:
        if y >= h or x >= w or y < 0 or x < 0:
            continue

        pixel = screenshot[y, x]
        b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])

        # Check for yellow/gold/orange
        is_yellow = (
            r > 100 and
            g > 80 and
            b < 150 and
            r > b and
            g > b - 30
        )

        print(f"  ({x:4d}, {y:4d}): BGR=({b:3d}, {g:3d}, {r:3d}) -> {'YELLOW' if is_yellow else 'not yellow'}")

print("\n" + "="*60)
print("Testing Bottom Navigation Bar Detection")
print("="*60)

nav_y_positions = [h - 30, h - 50, h - 70]
print("\nChecking navigation bar brightness:")
for y_pos in nav_y_positions:
    print(f"\nY position: {y_pos}")
    nav_samples = [
        (w // 5, y_pos),
        (w // 2, y_pos),
        (4 * w // 5, y_pos),
    ]

    for x, y in nav_samples:
        if y >= h or x >= w or y < 0 or x < 0:
            continue

        pixel = screenshot[y, x]
        brightness = int(pixel[0]) + int(pixel[1]) + int(pixel[2])
        is_nav = brightness < 100 or brightness > 350

        print(f"  ({x:4d}, {y:4d}): brightness={brightness:3d} -> {'NAV' if is_nav else 'not nav'}")

print("\n" + "="*60)
print("Testing Template Matching")
print("="*60)

from detection.image_matcher import ImageMatcher
matcher = ImageMatcher()

result = matcher.find_template(screenshot, 'battle_button', threshold=0.6)
if result:
    x, y, conf = result
    print(f"✓ Battle button found at ({x}, {y}) with confidence {conf:.2%}")
else:
    print("✗ Battle button NOT found (threshold 0.6)")

# Try even lower threshold
result = matcher.find_template(screenshot, 'battle_button', threshold=0.4)
if result:
    x, y, conf = result
    print(f"✓ Battle button found at ({x}, {y}) with confidence {conf:.2%} (threshold 0.4)")
else:
    print("✗ Battle button NOT found even at threshold 0.4")
