import cv2
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path


class CardHandDetector:
    """
    Detects cards in hand using template matching
    Works with fixed card positions at bottom of screen
    """

    def __init__(self, template_dir: str = "detection/card_templates"):
        """
        Initialize card hand detector

        Args:
            template_dir: Directory containing card template images
        """
        self.template_dir = Path(template_dir)

        # Card type mappings for your deck
        self.card_types = {
            'tombstone': 'building',
            'bomber': 'ranged',
            'valkyrie': 'melee',
            'goblins': 'melee',
            'spear_goblins': 'ranged',
            'cannon': 'building',
            'giant': 'tank',
            'skeletons': 'melee'
        }

        # Load card templates
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
        """Load all card templates from the templates directory"""
        for card_name in self.card_types.keys():
            # Load normal (available) template
            template_path = self.template_dir / f"{card_name}.png"
            if template_path.exists():
                template = cv2.imread(str(template_path))
                if template is not None:
                    self.templates[card_name] = template
                else:
                    print(f"Warning: Failed to load template: {template_path}")
            else:
                print(f"Warning: Template not found: {template_path}")

            # Load grayed out (unavailable) template if exists
            grayed_path = self.template_dir / f"{card_name}_grayed.png"
            if grayed_path.exists():
                grayed_template = cv2.imread(str(grayed_path))
                if grayed_template is not None:
                    self.templates[f"{card_name}_grayed"] = grayed_template

        print(f"Loaded {len(self.templates)} card templates (including grayed variants)")

    def identify_card(self, card_img: np.ndarray, threshold: float = 0.7) -> Optional[Dict[str, any]]:
        """
        Identify a single card image using template matching

        Args:
            card_img: Cropped card image from screenshot
            threshold: Minimum confidence for match (0-1)

        Returns:
            Dictionary with card_name, card_type, confidence, and available status, or None if no match
        """
        if card_img is None or card_img.size == 0:
            return None

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
            is_available = not best_card_name.endswith('_grayed')
            actual_card_name = best_card_name.replace('_grayed', '') if not is_available else best_card_name

            return {
                'card_name': actual_card_name,
                'card_type': self.card_types[actual_card_name],
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
