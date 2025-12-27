#!/usr/bin/env python3
"""
Test the integrated troop detection pipeline
Combines YOLO detection + card classification
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detection.troop_detector import TroopDetector
import cv2


def main():
    print("="*70)
    print("INTEGRATED TROOP DETECTION TEST")
    print("="*70)

    # Initialize detector
    detector = TroopDetector(
        yolo_model_path="models/best.pt",
        classifier_model_path="models/card_classifier.pt",
        confidence_threshold=0.25
    )

    # Find a test image
    dataset_images = Path("datasets/Clash Royale/test/images")

    if not dataset_images.exists():
        print(f"❌ Test images not found at {dataset_images}")
        print("Using training images instead...")
        dataset_images = Path("datasets/Clash Royale/train/images")

    if not dataset_images.exists():
        print("❌ No dataset images found!")
        return

    test_images = list(dataset_images.glob("*.jpg"))[:5]  # Test on first 5 images

    if not test_images:
        print("❌ No images found in dataset!")
        return

    print(f"\nTesting on {len(test_images)} images from {dataset_images.name}/\n")

    for img_path in test_images:
        print(f"\n{'='*70}")
        print(f"Processing: {img_path.name}")
        print(f"{'='*70}")

        # Load image
        screenshot = cv2.imread(str(img_path))
        if screenshot is None:
            print(f"❌ Failed to load {img_path.name}")
            continue

        print(f"Image size: {screenshot.shape[1]}x{screenshot.shape[0]}")

        # Run detection
        detections = detector.detect(screenshot, verbose=True)

        # Get battlefield summary
        state = detector.get_battlefield_state(detections)
        print(f"\n📊 Battlefield Summary:")
        print(f"  Allies: {state['total_allies']}")
        if state['ally_cards']:
            for card, count in sorted(state['ally_cards'].items()):
                print(f"    - {card.replace('_', ' ').title()}: {count}")

        print(f"  Enemies: {state['total_enemies']}")
        if state['enemy_cards']:
            for card, count in sorted(state['enemy_cards'].items()):
                print(f"    - {card.replace('_', ' ').title()}: {count}")

        # Visualize detections
        vis_img = detector.visualize_detections(screenshot, detections)

        # Save visualization
        output_dir = Path("detection_results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"detected_{img_path.name}"
        cv2.imwrite(str(output_path), vis_img)
        print(f"\n💾 Saved visualization to: {output_path}")

    print(f"\n{'='*70}")
    print("✓ Test complete! Check detection_results/ folder for visualizations")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
