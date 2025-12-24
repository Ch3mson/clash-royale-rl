"""
Create elixir templates from specified screenshots
"""
import cv2
import os
from pathlib import Path

# Elixir region coordinates
ELIXIR_REGION = (192, 1220, 238, 1254)

# Mapping: (session_dir, screenshot_num, elixir_value)
mappings = [
    # session_2025-12-19_21-12-41
    ("training_data/session_2025-12-19_21-12-41", 1, 8),
    ("training_data/session_2025-12-19_21-12-41", 2, 2),
    ("training_data/session_2025-12-19_21-12-41", 3, 1),
    ("training_data/session_2025-12-19_21-12-41", 4, 3),
    ("training_data/session_2025-12-19_21-12-41", 5, 0),
    ("training_data/session_2025-12-19_21-12-41", 7, 4),

    # session_2025-12-23_17-34-34
    ("training_data/session_2025-12-23_17-34-34", 1, 7),
    ("training_data/session_2025-12-23_17-34-34", 3, 9),
    ("training_data/session_2025-12-23_17-34-34", 4, 10),
    ("training_data/session_2025-12-23_17-34-34", 7, 6),
    ("training_data/session_2025-12-23_17-34-34", 9, 5),

    # session_2025-12-23_18-46-03
    ("training_data/session_2025-12-23_18-46-03", 1, 8),
    ("training_data/session_2025-12-23_18-46-03", 2, 9),
    ("training_data/session_2025-12-23_18-46-03", 3, 10),
    ("training_data/session_2025-12-23_18-46-03", 4, 10),
    ("training_data/session_2025-12-23_18-46-03", 5, 7),
    ("training_data/session_2025-12-23_18-46-03", 6, 6),
    ("training_data/session_2025-12-23_18-46-03", 8, 1),
    ("training_data/session_2025-12-23_18-46-03", 9, 3),
    ("training_data/session_2025-12-23_18-46-03", 10, 4),
    ("training_data/session_2025-12-23_18-46-03", 11, 5),
    ("training_data/session_2025-12-23_18-46-03", 12, 2),
    ("training_data/session_2025-12-23_18-46-03", 13, 1),
    ("training_data/session_2025-12-23_18-46-03", 14, 2),
    ("training_data/session_2025-12-23_18-46-03", 15, 4),
    ("training_data/session_2025-12-23_18-46-03", 17, 0),
    ("training_data/session_2025-12-23_18-46-03", 18, 1),
    ("training_data/session_2025-12-23_18-46-03", 21, 3),
    ("training_data/session_2025-12-23_18-46-03", 23, 5),
    ("training_data/session_2025-12-23_18-46-03", 24, 6),
    ("training_data/session_2025-12-23_18-46-03", 25, 3),
    ("training_data/session_2025-12-23_18-46-03", 52, 7),

    # session_2025-12-23_18-52-20
    ("training_data/session_2025-12-23_18-52-20", 2, 8),
    ("training_data/session_2025-12-23_18-52-20", 3, 9),
    ("training_data/session_2025-12-23_18-52-20", 4, 10),
    ("training_data/session_2025-12-23_18-52-20", 5, 8),
]

def create_templates():
    """Extract elixir regions and save as templates"""
    output_dir = Path("detection/elixir_templates")

    # Clear existing templates (backup already exists)
    if output_dir.exists():
        for old_template in output_dir.glob("elixir_*.png"):
            old_template.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    x1, y1, x2, y2 = ELIXIR_REGION

    # Count templates per elixir value
    counts = {}

    for session_dir, screenshot_num, elixir_value in mappings:
        # Find the screenshot file
        screenshot_pattern = f"screenshot_{screenshot_num:04d}_IN_BATTLE.png"
        screenshot_path = Path(session_dir) / screenshot_pattern

        if not screenshot_path.exists():
            print(f"⚠️  Not found: {screenshot_path}")
            continue

        # Load screenshot
        screenshot = cv2.imread(str(screenshot_path))
        if screenshot is None:
            print(f"⚠️  Could not load: {screenshot_path}")
            continue

        # Extract elixir region
        region = screenshot[y1:y2, x1:x2]

        # Verify size
        h, w = region.shape[:2]
        expected_w, expected_h = x2 - x1, y2 - y1

        if w != expected_w or h != expected_h:
            print(f"⚠️  Size mismatch for {screenshot_path}: {w}x{h} != {expected_w}x{expected_h}")
            continue

        # Count how many templates we have for this value
        if elixir_value not in counts:
            counts[elixir_value] = 1
        else:
            counts[elixir_value] += 1

        # Save template with variant number
        output_name = f"elixir_{elixir_value}_{counts[elixir_value]}.png"
        output_path = output_dir / output_name

        cv2.imwrite(str(output_path), region)
        print(f"✓ Created {output_name} from {screenshot_path.name}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total templates created: {sum(counts.values())}")
    print(f"Templates by elixir value:")
    for elixir_value in sorted(counts.keys()):
        print(f"  Elixir {elixir_value}: {counts[elixir_value]} template(s)")

    # Check for missing values
    missing = set(range(11)) - set(counts.keys())
    if missing:
        print(f"\n⚠️  Missing elixir values: {sorted(missing)}")
        print(f"   You'll need screenshots showing these values")

if __name__ == "__main__":
    create_templates()
