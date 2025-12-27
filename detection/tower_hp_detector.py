"""
Tower HP detection for Clash Royale
Uses OCR to read HP numbers from fixed tower positions
"""

import cv2
import numpy as np
from typing import Dict, Optional
import re

try:
    import pytesseract
    # Set tesseract path for macOS with homebrew
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. Install with: pip install pytesseract")
    print("Also requires Tesseract OCR: brew install tesseract (macOS)")


class TowerHPDetector:
    """
    Detects tower HP values using OCR on fixed screen regions
    """

    def __init__(self):
        """
        Initialize tower HP detector

        Tower positions for 720x1280 screen:
        - Enemy towers are at top
        - Ally towers are at bottom
        """
        if not TESSERACT_AVAILABLE:
            print("Warning: Tower HP detection disabled (pytesseract not available)")

        # HP bar regions (x1, y1, x2, y2) for 720x1280 screen
        # These are approximate - may need adjustment
        self.TOWER_REGIONS = {
            # Enemy towers (top of screen)
            'enemy_left_princess': (145, 170, 202, 192),
            'enemy_king': (340, 19, 402, 44),
            'enemy_right_princess': (524, 170, 581, 192),

            # Ally towers (bottom of screen)
            'ally_left_princess': (145, 794, 202, 816),
            'ally_king': (338, 968, 400, 994),
            'ally_right_princess': (524, 794, 581, 816),
        }

    def _preprocess_hp_region(self, region: np.ndarray, strategy: str = 'white_text') -> np.ndarray:
        """
        Preprocess HP region for better OCR accuracy

        Args:
            region: Cropped HP region from screenshot
            strategy: Preprocessing strategy ('white_text', 'adaptive', 'simple', 'otsu')

        Returns:
            Preprocessed image for OCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        if strategy == 'white_text':
            # Extract very bright text (white/light colored HP numbers)
            # Use a high threshold to get only the brightest pixels
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            # Invert to get black text on white background (Tesseract prefers this)
            processed = cv2.bitwise_not(binary)

        elif strategy == 'adaptive':
            # Adaptive thresholding - works better with varying lighting
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            # Invert to get black text on white
            processed = cv2.bitwise_not(binary)

        elif strategy == 'otsu':
            # Otsu's thresholding - automatic threshold selection
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Invert if text is darker than background
            if np.mean(binary) < 127:
                processed = cv2.bitwise_not(binary)
            else:
                processed = binary

        else:  # 'simple'
            # Simple high threshold for bright text
            _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            processed = binary

        # Clean up noise with smaller kernel
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

        # Scale up significantly for better OCR (6x for small text)
        scale = 6
        upscaled = cv2.resize(processed, (processed.shape[1] * scale, processed.shape[0] * scale),
                             interpolation=cv2.INTER_CUBIC)

        return upscaled

    def _extract_hp_number(self, text: str) -> Optional[int]:
        """
        Extract HP number from OCR text

        The text may contain both tower level (single digit) and HP (2-4 digits).
        OCR might split digits across lines or miss leading/trailing digits.
        We want the HP value, which is the largest number.

        Args:
            text: Raw OCR output

        Returns:
            HP value as integer, or None if not found (None = tower destroyed or not activated)
        """
        # Remove extra whitespace and normalize
        text = text.strip().replace('\n', ' ').replace('\r', ' ')

        # Try to find all numbers
        numbers = re.findall(r'\d+', text)

        if not numbers:
            # No numbers found - tower might be destroyed or not activated
            return 0

        # Convert to integers
        try:
            int_numbers = [int(n) for n in numbers]

            # Strategy 1: Look for the largest multi-digit number (likely HP)
            hp_candidates = [n for n in int_numbers if n >= 10]

            if hp_candidates:
                # Take the largest number (should be HP)
                hp = max(hp_candidates)
                # Sanity check: HP should be between 10 and ~4500 (max tower HP)
                if 10 <= hp <= 5000:
                    return hp

            # Strategy 2: If only single digits, try concatenating them
            # (OCR might have split "2030" into "2", "0", "3", "0")
            if len(int_numbers) >= 2:
                # Skip first number if it's single digit AND much smaller (likely tower level)
                # Only skip if first is < 10 and second exists
                start_idx = 0
                if len(int_numbers) > 1 and int_numbers[0] < 10:
                    # Check if second number is also single digit
                    # If so, they might all be HP digits
                    if int_numbers[1] < 10:
                        # All single digits - concatenate all
                        hp_str = ''.join(str(n) for n in int_numbers)
                    else:
                        # Second number is multi-digit - skip first (tower level)
                        hp_str = ''.join(str(n) for n in int_numbers[1:])
                else:
                    # First number is multi-digit - use all
                    hp_str = ''.join(str(n) for n in int_numbers)

                hp = int(hp_str)
                if 10 <= hp <= 5000:
                    return hp

            # Strategy 3: Single number that's too small (< 10)
            # Might be destroyed tower showing artifacts
            if len(int_numbers) == 1 and int_numbers[0] < 10:
                return 0

            return None

        except ValueError:
            return None

    def get_tower_hp(
        self,
        screenshot: np.ndarray,
        verbose: bool = False
    ) -> Dict[str, Optional[int]]:
        """
        Detect HP for all towers

        Args:
            screenshot: Full game screenshot (720x1280)
            verbose: If True, print detection details

        Returns:
            Dictionary with tower names and HP values
        """
        if not TESSERACT_AVAILABLE:
            # Return None for all towers if OCR not available
            return {tower: None for tower in self.TOWER_REGIONS.keys()}

        results = {}

        for tower_name, (x1, y1, x2, y2) in self.TOWER_REGIONS.items():
            # Crop tower HP region
            region = screenshot[y1:y2, x1:x2]

            # Run OCR with multiple preprocessing and PSM strategies
            try:
                hp = None
                best_text = ""

                # Try different preprocessing strategies
                preprocess_strategies = ['white_text', 'simple', 'adaptive', 'otsu']

                # Try different PSM modes for better recognition
                # PSM 8 and 13 work better for short number sequences
                psm_modes = [
                    8,   # Single word (best for isolated numbers)
                    13,  # Raw line (no assumptions)
                    7,   # Single line
                    6,   # Uniform block of text
                ]

                for strategy in preprocess_strategies:
                    processed = self._preprocess_hp_region(region, strategy=strategy)

                    for psm in psm_modes:
                        config = f'--psm {psm} -c tessedit_char_whitelist=0123456789'
                        text = pytesseract.image_to_string(processed, config=config)

                        if verbose and text.strip():
                            best_text = text.strip()

                        hp = self._extract_hp_number(text)

                        if hp is not None:
                            break  # Found valid HP

                    if hp is not None:
                        break  # Found valid HP

                results[tower_name] = hp

                if verbose:
                    if hp is not None:
                        print(f"{tower_name}: {hp} HP (OCR: '{best_text}')")
                    else:
                        print(f"{tower_name}: Could not detect HP (OCR: '{best_text}')")

            except Exception as e:
                if verbose:
                    print(f"Failed to read {tower_name}: {e}")
                results[tower_name] = None

        return results

    def save_hp_regions(self, screenshot: np.ndarray, output_dir: str = "tower_hp_crops"):
        """
        Save cropped HP regions for debugging/calibration

        Args:
            screenshot: Full game screenshot
            output_dir: Directory to save crops
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        h, w = screenshot.shape[:2]

        for tower_name, (x1, y1, x2, y2) in self.TOWER_REGIONS.items():
            # Validate coordinates are within bounds
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                print(f"⚠️  Warning: {tower_name} region {(x1, y1, x2, y2)} is invalid for {w}x{h} image")
                continue

            region = screenshot[y1:y2, x1:x2]

            if region.size == 0:
                print(f"⚠️  Warning: {tower_name} region is empty")
                continue

            # Save raw region
            raw_path = output_path / f"{tower_name}_raw.png"
            cv2.imwrite(str(raw_path), region)

            # Save preprocessed regions with different strategies
            for strategy in ['white_text', 'adaptive', 'otsu', 'simple']:
                processed = self._preprocess_hp_region(region, strategy=strategy)
                processed_path = output_path / f"{tower_name}_processed_{strategy}.png"
                cv2.imwrite(str(processed_path), processed)

        print(f"Saved HP crops to {output_dir}/")
        print("Check these images to verify tower HP regions are correct")
        print("If numbers are cut off, adjust TOWER_REGIONS coordinates")
