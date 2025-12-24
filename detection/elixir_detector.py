"""
Elixir counter using template matching OCR
Detects current elixir count (0-10) from game screenshot
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class ElixirDetector:
    """
    Detects current elixir count using template matching
    """

    # Elixir display region coordinates (x1, y1, x2, y2)
    ELIXIR_REGION = (192, 1220, 238, 1254)

    def __init__(self, template_dir: str = "detection/elixir_templates"):
        """
        Initialize elixir detector

        Args:
            template_dir: Directory containing elixir number templates (0-10)
        """
        self.template_dir = Path(template_dir)
        self.templates = {}

        # Create template directory if it doesn't exist
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Load templates for numbers 0-10
        self._load_templates()

    def _load_templates(self):
        """Load all elixir number templates (0-10), including multiple variants"""
        import os

        # Find all template files
        if not self.template_dir.exists():
            print(f"⚠️  Template directory not found: {self.template_dir}")
            return

        template_files = sorted(self.template_dir.glob("elixir_*.png"))

        # Group templates by elixir value
        # Templates can be named: elixir_5.png, elixir_5_1.png, elixir_5_2.png, etc.
        for template_path in template_files:
            # Extract elixir value from filename (e.g., "elixir_5_2.png" -> 5)
            name = template_path.stem  # "elixir_5_2"
            parts = name.split('_')
            if len(parts) >= 2:
                try:
                    elixir_value = int(parts[1])  # Get the number after "elixir_"
                    if 0 <= elixir_value <= 10:
                        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                        if template is not None:
                            # Store multiple templates per value
                            if elixir_value not in self.templates:
                                self.templates[elixir_value] = []
                            self.templates[elixir_value].append(template)
                except (ValueError, IndexError):
                    continue

        if not self.templates:
            print(f"⚠️  No elixir templates found in {self.template_dir}")
            print(f"   Add templates named: elixir_0.png through elixir_10.png")

    def get_elixir(self, screenshot: np.ndarray, verbose: bool = False) -> Optional[int]:
        """
        Detect current elixir count from screenshot

        Args:
            screenshot: Full game screenshot (BGR from OpenCV)
            verbose: Print detection details

        Returns:
            Elixir count (0-10) or None if detection fails
        """
        if not self.templates:
            if verbose:
                print("⚠️  No elixir templates loaded")
            return None

        # Extract elixir region
        x1, y1, x2, y2 = self.ELIXIR_REGION
        region = screenshot[y1:y2, x1:x2]

        # Convert to grayscale
        if len(region.shape) == 3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        region = clahe.apply(region)

        # Find best matching template (try all variants)
        best_match = None
        best_confidence = 0.0

        for elixir_value, template_list in self.templates.items():
            # Try each template variant for this elixir value
            for template in template_list:
                # Resize template to match region size if needed (handles different template sizes)
                if template.shape != region.shape:
                    template_resized = cv2.resize(template, (region.shape[1], region.shape[0]))
                else:
                    template_resized = template

                # Apply same CLAHE enhancement to template
                template_enhanced = clahe.apply(template_resized)

                # Perform template matching
                result = cv2.matchTemplate(region, template_enhanced, cv2.TM_CCOEFF_NORMED)
                max_val = result.max()

                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = elixir_value

        if verbose:
            print(f"Elixir detected: {best_match} (confidence: {best_confidence:.2f})")

        # Return best match if confidence is reasonable
        # Lower threshold (0.5) since templates may be different sizes and get resized
        if best_confidence > 0.5:
            return best_match
        else:
            if verbose:
                print(f"⚠️  Low confidence elixir detection: {best_confidence:.2f}")
            return None

    def save_elixir_crop(self, screenshot: np.ndarray, output_path: str = "elixir_crop.png"):
        """
        Save the elixir region crop for debugging/template creation

        Args:
            screenshot: Full game screenshot
            output_path: Where to save the crop
        """
        x1, y1, x2, y2 = self.ELIXIR_REGION
        region = screenshot[y1:y2, x1:x2]
        cv2.imwrite(output_path, region)
        print(f"✓ Elixir region saved to {output_path}")

    @classmethod
    def set_elixir_region(cls, x1: int, y1: int, x2: int, y2: int):
        """
        Update the elixir region coordinates

        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
        """
        cls.ELIXIR_REGION = (x1, y1, x2, y2)
        print(f"✓ Elixir region set to: ({x1}, {y1}, {x2}, {y2})")
