"""
Sort cropped troop images into card-specific folders.

Displays each crop one at a time and waits for text input to sort.
Type the card name to classify each crop.
"""
import cv2
from pathlib import Path
from collections import defaultdict


def sort_crops(
    crops_dir: str = "crops",
    sorted_dir: str = "sorted",
    skipped_dir: str = "skipped",
    window_name: str = "Sort Crops"
):
    """
    Interactive sorting of cropped troop images.

    Args:
        crops_dir: Directory containing cropped images
        sorted_dir: Base directory for sorted images
        skipped_dir: Directory for skipped images
        window_name: Name of display window
    """
    # Setup directories
    crops_path = Path(crops_dir)
    sorted_path = Path(sorted_dir)
    skipped_path = Path(skipped_dir)

    sorted_path.mkdir(parents=True, exist_ok=True)
    skipped_path.mkdir(parents=True, exist_ok=True)

    # Find all crop images
    crop_files = sorted(crops_path.glob("*.png"))

    if not crop_files:
        print(f"No images found in {crops_dir}")
        return

    print(f"Found {len(crop_files)} crops to sort")
    print("\n=== COMMANDS ===")
    print("  Type card name (e.g., 'knight', 'musketeer', 'giant')")
    print("  'skip' or 's' = skip this image")
    print("  'delete' or 'd' = delete this image")
    print("  'undo' or 'u' = undo last action")
    print("  'quit' or 'q' = quit sorting")
    print("==================\n")

    # Tracking
    counts = defaultdict(int)
    history = []  # For undo functionality
    current_idx = 0

    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while current_idx < len(crop_files):
        crop_path = crop_files[current_idx]

        # Load image
        img = cv2.imread(str(crop_path))
        if img is None:
            print(f"Warning: Could not load {crop_path.name}")
            current_idx += 1
            continue

        # Resize for better viewing (scale up small crops)
        h, w = img.shape[:2]
        scale = min(2400 / w, 1800 / h)
        if scale > 1:
            scale = min(scale, 8)  # Allow larger scaling
            new_w, new_h = int(w * scale), int(h * scale)
            display_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            display_img = img

        # Add progress text
        progress_text = f"Progress: {current_idx + 1}/{len(crop_files)}"
        cv2.putText(display_img, progress_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)  # Brief wait to update window

        # Get text input from user
        print(f"\n[{current_idx + 1}/{len(crop_files)}] Enter card name: ", end='', flush=True)
        user_input = input().strip().lower()

        # Handle quit
        if user_input in ['quit', 'q']:
            print("\nQuitting...")
            break

        # Handle undo
        if user_input in ['undo', 'u']:
            if history:
                last_action = history.pop()
                action_type, src, dst, card = last_action

                # Move file back
                if dst and dst.exists():
                    dst.rename(src)
                    counts[card] -= 1
                    print(f"Undid: {card}")

                # Go back one image
                current_idx = max(0, current_idx - 1)
                continue
            else:
                print("Nothing to undo")
                continue

        # Handle delete
        if user_input in ['delete', 'd']:
            crop_path.unlink()
            print(f"Deleted: {crop_path.name}")
            history.append(('delete', crop_path, None, 'deleted'))
            counts['deleted'] += 1
            current_idx += 1
            continue

        # Handle skip
        if user_input in ['skip', 's', '']:
            dest_path = skipped_path / crop_path.name
            crop_path.rename(dest_path)
            print(f"Skipped: {crop_path.name}")
            history.append(('skip', crop_path, dest_path, 'skipped'))
            counts['skipped'] += 1
            current_idx += 1
            continue

        # Handle card sorting - use whatever they typed as the card name
        if user_input:
            card_name = user_input.replace(' ', '_')

            # If this is a new card name, confirm with user
            if card_name not in counts:
                print(f"New card class: '{card_name}' - Is this correct? (y/n): ", end='', flush=True)
                confirm = input().strip().lower()
                if confirm not in ['y', 'yes']:
                    print("Cancelled. Re-enter card name.")
                    continue

            # Create card directory if needed
            card_dir = sorted_path / card_name
            card_dir.mkdir(parents=True, exist_ok=True)

            # Move file
            dest_path = card_dir / crop_path.name
            crop_path.rename(dest_path)

            # Update tracking
            counts[card_name] += 1
            history.append(('sort', crop_path, dest_path, card_name))

            print(f"Sorted to '{card_name}': {crop_path.name} ({counts[card_name]} total)")
            current_idx += 1
        else:
            print("Invalid input")

    cv2.destroyAllWindows()

    # Print summary
    print("\n=== SORTING SUMMARY ===")
    for card, count in sorted(counts.items()):
        print(f"{card}: {count}")
    print(f"\nTotal sorted: {sum(counts.values())}")
    print(f"Remaining: {len(list(crops_path.glob('*.png')))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sort cropped troop images")
    parser.add_argument("--crops", default="crops", help="Crops directory")
    parser.add_argument("--sorted", default="sorted", help="Output directory for sorted images")
    parser.add_argument("--skipped", default="skipped", help="Directory for skipped images")

    args = parser.parse_args()

    sort_crops(
        crops_dir=args.crops,
        sorted_dir=args.sorted,
        skipped_dir=args.skipped
    )
