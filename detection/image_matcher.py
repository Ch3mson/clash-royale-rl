# detection/image_matcher.py
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import os


class ImageMatcher:
    """
    Template matching for finding UI elements in screenshots
    Uses OpenCV's template matching to locate buttons and other UI elements
    Supports multiple template variants per button (like py-clash-bot)
    """
    
    def __init__(self, template_dir: str = "detection/images"):
        """
        Initialize image matcher with template directory
        
        Args:
            template_dir: Directory containing template images
        """
        self.template_dir = Path(template_dir)
        self.templates = {}  # Cache loaded templates {name: (image, width, height)}
        
        if not self.template_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {self.template_dir}")
    
    def load_template(self, name: str) -> Tuple[np.ndarray, int, int]:
        """
        Load and cache a template image
        
        Args:
            name: Template filename (e.g., 'battle_button.png')
            
        Returns:
            (grayscale_image, width, height)
        """
        # Return from cache if already loaded
        if name in self.templates:
            return self.templates[name]
        
        # Load from file
        template_path = self.template_dir / name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        # Load as grayscale for matching
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Failed to load template: {template_path}")
        
        h, w = template.shape
        
        # Cache it
        self.templates[name] = (template, w, h)
        
        return template, w, h
    
    def find_template(
        self, 
        screenshot: np.ndarray, 
        template_name: str, 
        threshold: float = 0.8
    ) -> Optional[Tuple[int, int, float]]:
        """
        Find template in screenshot using template matching
        Supports multiple template variants (e.g., battle_button_1.png, battle_button_2.png)
        
        Args:
            screenshot: Full game screenshot (BGR from OpenCV)
            template_name: Name of template file or base name (e.g., 'battle_button.png' or 'battle_button')
            threshold: Minimum confidence (0-1) to consider a match
            
        Returns:
            (center_x, center_y, confidence) if found, else None
        """
        templates_to_try = self._get_template_variants(template_name)
        
        if not templates_to_try:
            templates_to_try = [template_name]
        
        best_match = None
        best_confidence = 0
        
        # Try each template variant
        for template_file in templates_to_try:
            try:
                template, w, h = self.load_template(template_file)
            except (FileNotFoundError, ValueError):
                continue
            
            # Convert screenshot to grayscale (BGR from cv2.imread)
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            
            # Perform template matching
            result = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
            
            # Find the best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Update best match if this is better
            if max_val > best_confidence:
                best_confidence = max_val
                if best_confidence >= threshold:
                    top_left_x, top_left_y = max_loc
                    center_x = top_left_x + w // 2
                    center_y = top_left_y + h // 2
                    best_match = (center_x, center_y, best_confidence)
        
        return best_match
    
    def _get_template_variants(self, template_name: str) -> List[str]:
        """
        Get all variants of a template (e.g., button_1.png, button_2.png)
        
        Args:
            template_name: Base template name (e.g., 'battle_button.png' or 'battle_button')
            
        Returns:
            List of template filenames found
        """
        # Remove extension if present
        base_name = template_name.replace('.png', '').replace('.jpg', '')
        
        # Find all files matching pattern: base_name_*.png
        variants = []
        for file in os.listdir(self.template_dir):
            if file.startswith(base_name) and (file.endswith('.png') or file.endswith('.jpg')):
                variants.append(file)
        
        # Sort to ensure consistent order (1, 2, 3, etc.)
        variants.sort()
        
        return variants
    
    def find_all_templates(
        self,
        screenshot: np.ndarray,
        template_name: str,
        threshold: float = 0.8,
        max_matches: int = 10
    ) -> list[Tuple[int, int, float]]:
        """
        Find all occurrences of a template (for detecting multiple instances)
        
        Args:
            screenshot: Full game screenshot
            template_name: Name of template file
            threshold: Minimum confidence
            max_matches: Maximum number of matches to return
            
        Returns:
            List of (center_x, center_y, confidence) tuples
        """
        template, w, h = self.load_template(template_name)
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
        
        # Find all locations above threshold
        locations = np.where(result >= threshold)
        matches = []
        
        for pt in zip(*locations[::-1]):  # Switch x and y
            confidence = result[pt[1], pt[0]]
            center_x = pt[0] + w // 2
            center_y = pt[1] + h // 2
            matches.append((center_x, center_y, float(confidence)))
            
            if len(matches) >= max_matches:
                break
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[2], reverse=True)
        
        return matches
    
    def visualize_match(
        self,
        screenshot: np.ndarray,
        template_name: str,
        threshold: float = 0.8,
        output_path: str = "match_visualization.png"
    ) -> bool:
        """
        Visualize template matching result (for debugging)
        
        Args:
            screenshot: Full game screenshot
            template_name: Name of template file
            threshold: Confidence threshold
            output_path: Where to save visualization
            
        Returns:
            True if match found and saved, False otherwise
        """
        result = self.find_template(screenshot, template_name, threshold)
        
        if result is None:
            print(f"No match found for {template_name}")
            return False
        
        center_x, center_y, confidence = result
        template, w, h = self.load_template(template_name)
        
        # Draw rectangle and circle on screenshot
        viz = screenshot.copy()
        
        # Rectangle around matched region
        top_left = (center_x - w // 2, center_y - h // 2)
        bottom_right = (center_x + w // 2, center_y + h // 2)
        cv2.rectangle(viz, top_left, bottom_right, (0, 255, 0), 2)
        
        # Circle at center (click point)
        cv2.circle(viz, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add confidence text
        text = f"{confidence:.2f}"
        cv2.putText(viz, text, (center_x + 10, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save
        cv2.imwrite(output_path, viz)
        print(f"✓ Visualization saved to {output_path}")
        return True
