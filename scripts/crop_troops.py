"""
Crop detected troops from screenshots using YOLO model.

Loads YOLO model and runs inference on all images in screenshots/ folder.
Saves each detected bounding box as a separate cropped image.
"""
import cv2
from pathlib import Path
from ultralytics import YOLO


def crop_troops(
    model_path: str = "models/best.pt",
    screenshots_dir: str = "screenshots",
    output_dir: str = "crops",
    confidence_threshold: float = 0.25
):
    """
    Detect and crop troops from screenshots.

    Args:
        model_path: Path to YOLO model
        screenshots_dir: Directory containing screenshots
        output_dir: Directory to save cropped images
        confidence_threshold: Minimum confidence for detections
    """
    # Load YOLO model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    # Setup directories
    screenshots_path = Path(screenshots_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(screenshots_path.glob(ext))

    image_files = sorted(image_files)

    if not image_files:
        print(f"No images found in {screenshots_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process each image
    crop_count = 0

    for img_idx, img_path in enumerate(image_files, 1):
        print(f"Processing {img_idx}/{len(image_files)}: {img_path.name}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Could not load {img_path.name}")
            continue

        # Run inference
        results = model(img, verbose=False)

        # Extract detections
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get confidence
                conf = float(box.conf[0])

                if conf < confidence_threshold:
                    continue

                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Crop region
                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Save crop
                crop_count += 1
                crop_filename = f"crop_{crop_count:04d}.png"
                crop_path = output_path / crop_filename
                cv2.imwrite(str(crop_path), crop)

        print(f"  Extracted {len(boxes)} crops")

    print(f"\nDone! Saved {crop_count} crops to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop detected troops from screenshots")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLO model")
    parser.add_argument("--screenshots", default="screenshots", help="Screenshots directory")
    parser.add_argument("--output", default="crops", help="Output directory for crops")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    crop_troops(
        model_path=args.model,
        screenshots_dir=args.screenshots,
        output_dir=args.output,
        confidence_threshold=args.confidence
    )
