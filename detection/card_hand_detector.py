import cv2
import numpy as np
import os
from typing import List, Optional, Dict
from pathlib import Path

# Import card information database
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from detection.card_info import CARD_TYPES, CARD_INFO, get_card_category


class CardHandDetector:
    """
    Detects cards in hand using YOLOv8 classification with template matching fallback
    Works with fixed card positions at bottom of screen
    """

    def __init__(self, template_dir: str = "detection/card_templates",
                 classifier_path: str = "models/card_hand_classifier.pt",
                 use_yolo: bool = True):
        """
        Initialize card hand detector

        Args:
            template_dir: Directory containing card template images
            classifier_path: Path to YOLOv8 classification model
            use_yolo: If True, use YOLOv8 classifier; if False, use template matching
        """
        self.template_dir = Path(template_dir)
        self.use_yolo = use_yolo

        # Card type mappings loaded from card_info.py
        # To add new cards, edit detection/card_info.py
        self.card_types = CARD_TYPES
        self.card_info = CARD_INFO  # Full card info for future features

        # Load YOLOv8 classifier if enabled
        self.classifier = None
        if self.use_yolo:
            try:
                from ultralytics import YOLO
                if Path(classifier_path).exists():
                    self.classifier = YOLO(classifier_path)
                    print(f"Loaded YOLOv8 card hand classifier: {classifier_path}")
                else:
                    print(f"Warning: YOLOv8 model not found at {classifier_path}, falling back to template matching")
                    self.use_yolo = False
            except ImportError:
                print("Warning: ultralytics not installed, falling back to template matching")
                self.use_yolo = False

        # Load card templates (for fallback or if YOLO disabled)
        self.templates = {}
        self._load_templates()

        # Card slot coordinates for 720x1280 screen
        # Format: (x1, y1, x2, y2) for cropping
        # Cards are taller rectangles, positioned at bottom-right
        self.CARD_SLOTS = [
            (156, 1040, 292, 1225),  # Slot 0 (leftmost) - moved right 32px, up 70px, taller (180px height)
            (292, 1040, 427, 1225),  # Slot 1
            (427, 1040, 562, 1225),  # Slot 2
            (562, 1040, 698, 1225),  # Slot 3 (rightmost)
        ]

    def _load_templates(self):
        """
        Load all card templates from the templates directory
        Supports multiple variants per card (e.g., cannon_1.png, cannon_2.png)
        """
        for card_name in self.card_types.keys():
            # Find all variants for this card (e.g., cannon_1.png, cannon_2.png, etc.)
            variants = self._get_template_variants(card_name)

            if not variants:
                print(f"Warning: No templates found for '{card_name}'")
                continue

            # Load all variants
            for variant_file in variants:
                template_path = self.template_dir / variant_file
                template = cv2.imread(str(template_path))
                if template is not None:
                    # Store with full variant name (e.g., "cannon_1" not just "cannon")
                    variant_name = variant_file.replace('.png', '').replace('.jpg', '')

                    # Check if it's a grayed variant
                    if '_grayed' in variant_name:
                        self.templates[variant_name] = template
                    else:
                        self.templates[variant_name] = template
                else:
                    print(f"Warning: Failed to load template: {template_path}")

            # Also load grayed out variants if they exist
            grayed_variants = self._get_template_variants(f"{card_name}_grayed")
            for grayed_file in grayed_variants:
                grayed_path = self.template_dir / grayed_file
                grayed_template = cv2.imread(str(grayed_path))
                if grayed_template is not None:
                    variant_name = grayed_file.replace('.png', '').replace('.jpg', '')
                    self.templates[variant_name] = grayed_template

        print(f"Loaded {len(self.templates)} card templates (including variants and grayed)")

    def _get_template_variants(self, card_name: str) -> List[str]:
        """
        Get all template variants for a card (e.g., cannon_1.png, cannon_2.png)

        Args:
            card_name: Base card name (e.g., 'cannon')

        Returns:
            List of template filenames found
        """
        variants = []

        # Find all files matching pattern: card_name_*.png or card_name.png
        for file in os.listdir(self.template_dir):
            if file.endswith('.png') or file.endswith('.jpg'):
                # Remove extension
                base = file.replace('.png', '').replace('.jpg', '')

                # Check if it matches card_name or card_name_<number>
                if base == card_name:
                    variants.append(file)
                elif base.startswith(card_name + '_'):
                    # Check if suffix is a number (variant) or "grayed"
                    suffix = base[len(card_name) + 1:]
                    if suffix.isdigit() or suffix == 'grayed':
                        variants.append(file)

        # Sort to ensure consistent order (1, 2, 3, etc.)
        variants.sort()

        return variants

    def identify_card(self, card_img: np.ndarray, threshold: float = 0.7) -> Optional[Dict[str, any]]:
        """
        Identify a single card image using YOLOv8 classifier or template matching

        Args:
            card_img: Cropped card image from screenshot
            threshold: Minimum confidence for match (0-1)

        Returns:
            Dictionary with card_name, card_type, confidence, and available status, or None if no match
        """
        if card_img is None or card_img.size == 0:
            return None

        # Try YOLOv8 classification first
        if self.use_yolo and self.classifier is not None:
            try:
                results = self.classifier(card_img, verbose=False)
                probs = results[0].probs

                card_name = self.classifier.names[int(probs.top1)]
                confidence = float(probs.top1conf)

                if confidence >= threshold:
                    # Detect if card is available by checking color saturation
                    # Grayed cards have low saturation
                    hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
                    avg_saturation = np.mean(hsv[:, :, 1])
                    is_available = avg_saturation > 50  # Threshold for grayed detection

                    return {
                        'card_name': card_name,
                        'card_type': self.card_types.get(card_name, 'unknown'),
                        'confidence': confidence,
                        'available': is_available
                    }
            except Exception as e:
                print(f"YOLOv8 classification failed: {e}, falling back to template matching")
                # Fall through to template matching

        # Fallback to template matching
        best_match = None
        best_score = 0
        best_card_name = None

        for template_name, template in self.templates.items():
            # Resize template to match card image size if needed
            if template.shape[:2] != card_img.shape[:2]:
                template_resized = cv2.resize(template, (card_img.shape[1], card_img.shape[0]))
            else:
                template_resized = template

            # Perform template matching
            result = cv2.matchTemplate(card_img, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_card_name = template_name

        if best_score >= threshold and best_card_name:
            # Check if it's a grayed out variant
            is_available = not '_grayed' in best_card_name

            # Extract actual card name (remove _grayed and _<number> suffixes)
            actual_card_name = best_card_name.replace('_grayed', '')
            # Remove variant number (e.g., cannon_1 -> cannon)
            if '_' in actual_card_name and actual_card_name.split('_')[-1].isdigit():
                actual_card_name = '_'.join(actual_card_name.split('_')[:-1])

            return {
                'card_name': actual_card_name,
                'card_type': self.card_types.get(actual_card_name, 'unknown'),
                'confidence': best_score,
                'available': is_available
            }

        return None

    def get_hand(self, screenshot: np.ndarray, verbose: bool = False) -> List[Optional[Dict[str, any]]]:
        """
        Detect all 4 cards in hand from screenshot

        Args:
            screenshot: Full game screenshot (720x1280)
            verbose: If True, print detection results

        Returns:
            List of 4 card dictionaries (or None for undetected slots)
        """
        hand = []

        for slot_idx, (x1, y1, x2, y2) in enumerate(self.CARD_SLOTS):
            # Crop card from screenshot
            card_img = screenshot[y1:y2, x1:x2]

            # Identify card
            card_info = self.identify_card(card_img)
            hand.append(card_info)

        if verbose:
            self._print_hand(hand)

        return hand

    def _print_hand(self, hand: List[Optional[Dict[str, any]]]):
        """Print detected hand in readable format"""
        print(f"\n{'='*60}")
        print("Current Hand:")
        print(f"{'='*60}")

        for slot_idx, card_info in enumerate(hand):
            if card_info:
                card_name = card_info['card_name'].replace('_', ' ').title()
                card_type = card_info['card_type']
                conf = card_info['confidence']
                available = card_info.get('available', True)
                status = "✓" if available else "✗"
                print(f"  Slot {slot_idx}: {card_name:20s} ({card_type:8s}) | {status} | conf: {conf:.2f}")
            else:
                print(f"  Slot {slot_idx}: {'Unknown':20s}")

        print(f"{'='*60}\n")

    def save_card_crops(self, screenshot: np.ndarray, output_dir: str = "card_crops"):
        """
        Save cropped card images for debugging/template creation

        Args:
            screenshot: Full game screenshot
            output_dir: Directory to save crops
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for slot_idx, (x1, y1, x2, y2) in enumerate(self.CARD_SLOTS):
            card_img = screenshot[y1:y2, x1:x2]
            filename = output_path / f"card_slot_{slot_idx}.png"
            cv2.imwrite(str(filename), card_img)

        print(f"Saved card crops to {output_dir}/")
