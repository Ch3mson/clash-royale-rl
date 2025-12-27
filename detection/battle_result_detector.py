"""
Battle result detection for Clash Royale
Detects victory, defeat, or draw at the end of battle
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Literal


class BattleResultDetector:
    """
    Detects battle end screen and determines result (victory, defeat, draw)
    Uses template matching to identify result text/icons
    """

    def __init__(self, template_dir: str = "detection/result_templates"):
        """
        Initialize battle result detector

        Args:
            template_dir: Directory containing result screen templates
        """
        self.template_dir = Path(template_dir)
        self.templates = {
            'victory': [],
            'defeat': [],
            'draw': []
        }

        # Create template directory if it doesn't exist
        self.template_dir.mkdir(exist_ok=True)

        # Load templates
        self._load_templates()

    def _load_templates(self):
        """
        Load victory, defeat, and draw templates
        Each result type can have multiple variants
        """
        for result_type in ['victory', 'defeat', 'draw']:
            # Find all variants (e.g., victory_1.png, victory_2.png)
            variants = list(self.template_dir.glob(f"{result_type}_*.png"))
            variants.extend(self.template_dir.glob(f"{result_type}.png"))

            for template_path in variants:
                template = cv2.imread(str(template_path))
                if template is not None:
                    self.templates[result_type].append(template)

            if self.templates[result_type]:
                print(f"Loaded {len(self.templates[result_type])} {result_type} templates")

    def detect_result(
        self,
        screenshot: np.ndarray,
        threshold: float = 0.8,
        verbose: bool = False
    ) -> Optional[Literal['victory', 'defeat', 'draw']]:
        """
        Detect battle result from screenshot

        Args:
            screenshot: Full game screenshot
            threshold: Minimum confidence for match (0-1), default 0.8 for full-screen matching
            verbose: If True, print detection details

        Returns:
            'victory', 'defeat', 'draw', or None if no result detected
        """
        if screenshot is None or screenshot.size == 0:
            return None

        best_score = 0
        best_result = None

        # Try matching each result type using full screenshot templates
        for result_type, templates in self.templates.items():
            for template in templates:
                # Templates should be full screenshots, so they should match dimensions
                if template.shape[:2] != screenshot.shape[:2]:
                    # Resize template to match screenshot if needed
                    template_resized = cv2.resize(template, (screenshot.shape[1], screenshot.shape[0]))
                else:
                    template_resized = template

                # Perform full-screen template matching
                # Use mean squared error for full image comparison
                diff = cv2.absdiff(screenshot, template_resized)
                mse = np.mean(diff ** 2)

                # Convert MSE to similarity score (0-1, higher is better)
                # Normalize by max possible MSE (255^2 for 8-bit images)
                max_mse = 255 ** 2
                similarity = 1.0 - (mse / max_mse)

                if similarity > best_score:
                    best_score = similarity
                    best_result = result_type

        if best_score >= threshold:
            if verbose:
                print(f"[RESULT] {best_result.upper()} detected (confidence: {best_score:.2f})")
            return best_result

        return None

    def is_battle_ended(self, screenshot: np.ndarray, threshold: float = 0.8) -> bool:
        """
        Check if battle has ended (any result detected)

        Args:
            screenshot: Full game screenshot
            threshold: Minimum confidence for match

        Returns:
            True if battle ended, False otherwise
        """
        result = self.detect_result(screenshot, threshold=threshold, verbose=False)
        return result is not None

    def save_result_template(
        self,
        screenshot: np.ndarray,
        result_type: Literal['victory', 'defeat', 'draw'],
        variant_num: Optional[int] = None
    ):
        """
        Save full screenshot as a template for result detection

        Args:
            screenshot: Full game screenshot showing result
            result_type: Type of result ('victory', 'defeat', 'draw')
            variant_num: Optional variant number for multiple templates
        """
        # Generate filename
        if variant_num is not None:
            template_filename = f"{result_type}_{variant_num}.png"
        else:
            template_filename = f"{result_type}.png"

        # Save full screenshot as template
        template_path = self.template_dir / template_filename
        cv2.imwrite(str(template_path), screenshot)
        print(f"Saved {result_type} template to {template_path}")
