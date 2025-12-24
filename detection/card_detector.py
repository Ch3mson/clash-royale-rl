from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2


class CardDetector:
    """
    YOLO-based detector for Clash Royale troops on the battlefield
    Detects ally and enemy units with their positions and classes
    """

    def __init__(self, model_path: str = "models/best.pt", confidence_threshold: float = 0.5, grid_system=None):
        """
        Initialize the YOLO model for troop detection

        Args:
            model_path: Path to the trained YOLO weights file
            confidence_threshold: Minimum confidence for detections (0-1)
            grid_system: GridSystem instance for converting pixels to grid coordinates
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.grid_system = grid_system

        # Class names from your training
        self.class_names = [
            'ally_air',
            'ally_building',
            'ally_melee',
            'ally_ranged',
            'ally_tank',
            'enemy_air',
            'enemy_building',
            'enemy_melee',
            'enemy_ranged',
            'enemy_tank'
        ]

    def detect(self, screenshot: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Run YOLO inference on a screenshot to detect troops

        Args:
            screenshot: BGR image from OpenCV (720x1280)
            verbose: If True, print detection results

        Returns:
            List of detections, each containing:
                - class_name: str (e.g., "enemy_tank")
                - confidence: float (0-1)
                - bbox: tuple (x1, y1, x2, y2) - bounding box coordinates
                - center: tuple (x, y) - center point of the detection
        """
        # Run inference
        results = self.model(screenshot, conf=self.confidence_threshold, verbose=False)

        detections = []

        # Parse results
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]

                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Convert to grid coordinates if grid system available
                grid_pos = None
                if self.grid_system:
                    grid_row, grid_col = self.grid_system.pixel_to_grid(center_x, center_y)
                    grid_pos = (grid_row, grid_col)

                detection = {
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (center_x, center_y),
                    'grid': grid_pos
                }

                detections.append(detection)

        if verbose:
            self._print_detections(detections)

        return detections

    def _print_detections(self, detections: List[Dict]):
        """Print detection results in a readable format"""
        if not detections:
            print("No troops detected on battlefield")
            return

        print(f"\n{'='*60}")
        print(f"Detected {len(detections)} troops:")
        print(f"{'='*60}")

        # Group by team
        allies = [d for d in detections if d['class_name'].startswith('ally')]
        enemies = [d for d in detections if d['class_name'].startswith('enemy')]

        if allies:
            print(f"\nAllies ({len(allies)}):")
            for d in allies:
                if d['grid']:
                    grid_row, grid_col = d['grid']
                    print(f"  - {d['class_name']:20s} [Grid: {grid_row:2d},{grid_col:2d}] | conf: {d['confidence']:.2f}")
                else:
                    print(f"  - {d['class_name']:20s} at ({d['center'][0]:3d}, {d['center'][1]:3d}) | conf: {d['confidence']:.2f}")

        if enemies:
            print(f"\nEnemies ({len(enemies)}):")
            for d in enemies:
                if d['grid']:
                    grid_row, grid_col = d['grid']
                    print(f"  - {d['class_name']:20s} [Grid: {grid_row:2d},{grid_col:2d}] | conf: {d['confidence']:.2f}")
                else:
                    print(f"  - {d['class_name']:20s} at ({d['center'][0]:3d}, {d['center'][1]:3d}) | conf: {d['confidence']:.2f}")

        print(f"{'='*60}\n")

    def visualize_detections(self, screenshot: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the screenshot

        Args:
            screenshot: Original BGR image
            detections: List of detections from detect()

        Returns:
            Image with visualized detections
        """
        img = screenshot.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']

            # Choose color based on team
            if class_name.startswith('ally'):
                color = (0, 255, 0)  # Green for allies
            else:
                color = (0, 0, 255)  # Red for enemies

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw center point
            center_x, center_y = detection['center']
            cv2.circle(img, (center_x, center_y), 5, color, -1)

        return img

    def get_enemy_count_by_type(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count enemy troops by type

        Args:
            detections: List of detections from detect()

        Returns:
            Dictionary with counts: {'tank': 2, 'air': 1, ...}
        """
        counts = {
            'tank': 0,
            'air': 0,
            'melee': 0,
            'ranged': 0,
            'building': 0
        }

        for d in detections:
            if d['class_name'].startswith('enemy'):
                troop_type = d['class_name'].split('_')[1]  # Get 'tank' from 'enemy_tank'
                if troop_type in counts:
                    counts[troop_type] += 1

        return counts
