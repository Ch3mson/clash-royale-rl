#!/usr/bin/env python3
"""
Evaluate CardHandDetector accuracy on training screenshots
"""
import cv2
import numpy as np
from pathlib import Path
from detection.card_hand_detector import CardHandDetector
from collections import defaultdict

def evaluate_hand_detector():
    print("="*80)
    print("CardHandDetector Accuracy Evaluation")
    print("="*80)

    # Initialize detector
    detector = CardHandDetector()

    # Check template coverage
    print("\n1. TEMPLATE COVERAGE ANALYSIS")
    print("-"*80)

    # Cards in card_types dictionary
    cards_in_dict = set(detector.card_types.keys())
    print(f"Cards in card_types dictionary ({len(cards_in_dict)}):")
    for card in sorted(cards_in_dict):
        print(f"  - {card}: {detector.card_types[card]}")

    # Cards with templates (excluding _grayed variants)
    template_dir = Path("detection/card_templates")
    template_files = list(template_dir.glob("*.png"))
    cards_with_templates = set()
    for f in template_files:
        name = f.stem  # filename without .png
        # Remove number suffix (e.g., cannon_1 -> cannon)
        if '_' in name and name.split('_')[-1].isdigit():
            base_name = '_'.join(name.split('_')[:-1])
        else:
            base_name = name
        cards_with_templates.add(base_name)

    cards_with_templates.discard('empty')  # Remove empty.png

    print(f"\nCards with templates ({len(cards_with_templates)}):")
    for card in sorted(cards_with_templates):
        count = len(list(template_dir.glob(f"{card}_*.png"))) + (1 if (template_dir / f"{card}.png").exists() else 0)
        print(f"  - {card}: {count} variants")

    # Find missing cards
    missing_in_dict = cards_with_templates - cards_in_dict
    missing_templates = cards_in_dict - cards_with_templates

    if missing_in_dict:
        print(f"\n⚠️  Cards with templates but NOT in card_types dictionary ({len(missing_in_dict)}):")
        for card in sorted(missing_in_dict):
            print(f"  - {card}")

    if missing_templates:
        print(f"\n⚠️  Cards in card_types dictionary but MISSING templates ({len(missing_templates)}):")
        for card in sorted(missing_templates):
            print(f"  - {card}")

    # Test on training screenshots if available
    print("\n2. TESTING ON TRAINING SCREENSHOTS")
    print("-"*80)

    training_dir = Path("training_data")
    if not training_dir.exists():
        print("No training_data directory found. Skipping screenshot tests.")
        return

    # Find recent training sessions
    sessions = sorted(list(training_dir.glob("session_*")), reverse=True)[:3]

    if not sessions:
        print("No training sessions found. Run with --screenshots to generate test data.")
        return

    print(f"Testing on {len(sessions)} most recent sessions...")

    # Statistics
    total_slots = 0
    detected_slots = 0
    confidence_scores = defaultdict(list)
    detection_failures = []

    for session in sessions:
        screenshots = list(session.glob("screenshot_*_IN_BATTLE.png"))[:5]  # Sample 5 per session

        for screenshot_path in screenshots:
            screenshot = cv2.imread(str(screenshot_path))
            if screenshot is None:
                continue

            hand = detector.get_hand(screenshot, verbose=False)

            for slot_idx, card_info in enumerate(hand):
                total_slots += 1
                if card_info:
                    detected_slots += 1
                    confidence_scores[card_info['card_name']].append(card_info['confidence'])
                else:
                    detection_failures.append((screenshot_path.name, slot_idx))

    # Print results
    if total_slots > 0:
        detection_rate = (detected_slots / total_slots) * 100
        print(f"\nDetection Rate: {detected_slots}/{total_slots} ({detection_rate:.1f}%)")

        print("\nConfidence Scores by Card:")
        for card_name in sorted(confidence_scores.keys()):
            scores = confidence_scores[card_name]
            avg_conf = np.mean(scores)
            min_conf = np.min(scores)
            max_conf = np.max(scores)
            print(f"  {card_name:20s}: avg={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f} ({len(scores)} samples)")

        if detection_failures:
            print(f"\n⚠️  Detection Failures ({len(detection_failures)}):")
            for screenshot, slot in detection_failures[:10]:  # Show first 10
                print(f"  - {screenshot}, slot {slot}")
            if len(detection_failures) > 10:
                print(f"  ... and {len(detection_failures) - 10} more")

    # Check unclassified cards directory
    print("\n3. UNCLASSIFIED CARDS ANALYSIS")
    print("-"*80)

    unclassified_found = False
    for session in sessions:
        unclassified_dir = session / "unclassified_cards"
        if unclassified_dir.exists():
            unclassified_cards = list(unclassified_dir.glob("card_*.png"))
            if unclassified_cards:
                print(f"\nSession: {session.name}")
                print(f"Found {len(unclassified_cards)} unclassified cards")

                # Analyze confidence distribution
                low_conf_count = 0
                unknown_count = 0
                for card_path in unclassified_cards:
                    if "unknown" in card_path.name:
                        unknown_count += 1
                    elif "conf0." in card_path.name or "conf0.0" in card_path.name:
                        low_conf_count += 1

                print(f"  - Unknown cards: {unknown_count}")
                print(f"  - Low confidence (<0.7): {low_conf_count - unknown_count}")
                unclassified_found = True

    if not unclassified_found:
        print("No unclassified cards found. Run with --screenshots to collect training data.")

    # Recommendations
    print("\n4. RECOMMENDATIONS")
    print("-"*80)

    if missing_in_dict:
        print("\n✓ UPDATE card_types dictionary to include:")
        print("  Add these lines to detection/card_hand_detector.py:")
        for card in sorted(missing_in_dict):
            # Guess card type based on name
            if 'tower' in card or 'furnace' in card or 'cage' in card:
                card_type = 'building'
            elif 'dragon' in card or 'spirit' in card:
                card_type = 'air'
            elif 'ram' in card or 'giant' in card:
                card_type = 'tank'
            elif 'wizard' in card:
                card_type = 'ranged'
            else:
                card_type = 'melee'
            print(f"  '{card}': '{card_type}',")

    if missing_templates:
        print("\n✗ MISSING TEMPLATES:")
        print("  Create templates for these cards in detection/card_templates/:")
        for card in sorted(missing_templates):
            print(f"  - {card}.png (or multiple variants)")

    if total_slots > 0 and detection_rate < 80:
        print("\n⚠️  Low detection rate (<80%):")
        print("  Consider:")
        print("  1. Lowering threshold in identify_card() from 0.7 to 0.6")
        print("  2. Adding more template variants for poorly detected cards")
        print("  3. Training a YOLOv8 classifier instead of template matching")

    print("\n" + "="*80)

if __name__ == "__main__":
    evaluate_hand_detector()
