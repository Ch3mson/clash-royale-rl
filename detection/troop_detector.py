from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Optional
import cv2


class TroopDetector:
    """
    Two-stage detector for Clash Royale troops:
    1. YOLO detects WHERE troops are (bounding boxes) + ally/enemy
    2. Classifier identifies WHICH card each troop is
    """

    def __init__(
        self,
        yolo_model_path: str = "models/best.pt",
        classifier_model_path: str = "models/card_classifier.pt",
        confidence_threshold: float = 0.25,
        grid_system=None
    ):
        """
        Initialize the two-stage detection pipeline

        Args:
            yolo_model_path: Path to YOLO detection model (ally vs enemy)
            classifier_model_path: Path to card classification model
            confidence_threshold: Minimum confidence for YOLO detections (0-1)
            grid_system: GridSystem instance for converting pixels to grid coordinates
        """
        print("Loading YOLO detection model...")
        self.yolo_model = YOLO(yolo_model_path)

        print("Loading card classifier model...")
        self.classifier_model = YOLO(classifier_model_path)

        self.confidence_threshold = confidence_threshold
        self.grid_system = grid_system

        print(f"✓ Models loaded successfully!")
        print(f"  YOLO classes: {self.yolo_model.names}")
        print(f"  Classifier classes: {list(self.classifier_model.names.values())}")

    def detect(self, screenshot: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Detect and classify all troops in a screenshot

        Args:
            screenshot: BGR image from OpenCV (720x1280)
            verbose: If True, print detection results

        Returns:
            List of detections, each containing:
                - team: str ('ally' or 'enemy')
                - card_name: str (e.g., 'knight', 'musketeer')
                - yolo_confidence: float (0-1) - YOLO detection confidence
                - card_confidence: float (0-1) - Card classification confidence
                - bbox: tuple (x1, y1, x2, y2) - bounding box coordinates
                - center: tuple (x, y) - center point of the detection
                - grid: tuple (row, col) or None - grid coordinates if available
                - crop: np.ndarray - cropped troop image
        """
        # Stage 1: YOLO detects WHERE troops are and if ally/enemy
        yolo_results = self.yolo_model(screenshot, conf=self.confidence_threshold, verbose=False)

        detections = []

        # Parse YOLO results
        for result in yolo_results:
            boxes = result.boxes

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get YOLO confidence and class
                yolo_confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                team = self.yolo_model.names[class_id]  # 'ally' or 'enemy'

                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Crop the detected troop for classification
                crop = screenshot[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Stage 2: Classify WHICH card it is
                card_results = self.classifier_model.predict(crop, verbose=False)

                # Get top prediction
                top_class_id = card_results[0].probs.top1
                card_name = self.classifier_model.names[top_class_id]
                card_confidence = float(card_results[0].probs.top1conf)

                # Convert to grid coordinates if grid system available
                grid_pos = None
                if self.grid_system:
                    grid_row, grid_col = self.grid_system.pixel_to_grid(center_x, center_y)
                    grid_pos = (grid_row, grid_col)

                detection = {
                    'team': team,
                    'card_name': card_name,
                    'yolo_confidence': yolo_confidence,
                    'card_confidence': card_confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'grid': grid_pos,
                    'crop': crop
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

        print(f"\n{'='*70}")
        print(f"Detected {len(detections)} troops:")
        print(f"{'='*70}")

        # Group by team
        allies = [d for d in detections if d['team'] == 'ally']
        enemies = [d for d in detections if d['team'] == 'enemy']

        if allies:
            print(f"\nAllies ({len(allies)}):")
            for d in allies:
                card = d['card_name'].replace('_', ' ').title()
                yolo_conf = d['yolo_confidence']
                card_conf = d['card_confidence']

                if d['grid']:
                    grid_row, grid_col = d['grid']
                    print(f"  - {card:20s} [Grid: {grid_row:2d},{grid_col:2d}] | YOLO: {yolo_conf:.2f} | Card: {card_conf:.2f}")
                else:
                    cx, cy = d['center']
                    print(f"  - {card:20s} at ({cx:3d}, {cy:3d}) | YOLO: {yolo_conf:.2f} | Card: {card_conf:.2f}")

        if enemies:
            print(f"\nEnemies ({len(enemies)}):")
            for d in enemies:
                card = d['card_name'].replace('_', ' ').title()
                yolo_conf = d['yolo_confidence']
                card_conf = d['card_confidence']

                if d['grid']:
                    grid_row, grid_col = d['grid']
                    print(f"  - {card:20s} [Grid: {grid_row:2d},{grid_col:2d}] | YOLO: {yolo_conf:.2f} | Card: {card_conf:.2f}")
                else:
                    cx, cy = d['center']
                    print(f"  - {card:20s} at ({cx:3d}, {cy:3d}) | YOLO: {yolo_conf:.2f} | Card: {card_conf:.2f}")

        print(f"{'='*70}\n")

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
            team = detection['team']
            card_name = detection['card_name']
            card_conf = detection['card_confidence']

            # Choose color based on team
            if team == 'ally':
                color = (0, 255, 0)  # Green for allies
            else:
                color = (0, 0, 255)  # Red for enemies

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label with card name and confidence
            label = f"{card_name} {card_conf:.2f}"

            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)

            # Text
            cv2.putText(img, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw center point
            center_x, center_y = detection['center']
            cv2.circle(img, (center_x, center_y), 3, color, -1)

        return img

    def get_troop_counts(self, detections: List[Dict]) -> Dict[str, Dict[str, int]]:
        """
        Count troops by team and card type

        Args:
            detections: List of detections from detect()

        Returns:
            Dictionary with structure:
            {
                'ally': {'knight': 2, 'archers': 1, ...},
                'enemy': {'giant': 1, 'minions': 3, ...}
            }
        """
        counts = {
            'ally': {},
            'enemy': {}
        }

        for d in detections:
            team = d['team']
            card = d['card_name']

            if card not in counts[team]:
                counts[team][card] = 0
            counts[team][card] += 1

        return counts

    def get_battlefield_state(self, detections: List[Dict]) -> Dict:
        """
        Get a summary of the battlefield state

        Args:
            detections: List of detections from detect()

        Returns:
            Dictionary containing:
                - total_allies: int
                - total_enemies: int
                - ally_cards: dict {card_name: count}
                - enemy_cards: dict {card_name: count}
        """
        counts = self.get_troop_counts(detections)

        return {
            'total_allies': sum(counts['ally'].values()),
            'total_enemies': sum(counts['enemy'].values()),
            'ally_cards': counts['ally'],
            'enemy_cards': counts['enemy']
        }
