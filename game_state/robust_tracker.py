"""
Robust Multi-Object Tracker
Handles overlapping troops, HP bar conflicts, and ID persistence
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class TrackingFeatures:
    """Features used to track a troop across frames"""
    position: Tuple[int, int]  # Center position
    bbox_size: Tuple[int, int]  # Width, height
    card_name: str  # Card type (stable across frames)
    team: str  # ally/enemy
    appearance_histogram: np.ndarray  # Color histogram of crop
    velocity: Tuple[float, float]  # Movement vector


class RobustTroopTracker:
    """
    Tracks troops using multiple features to prevent ID confusion
    when HP bars overlap or troops are close together
    """

    def __init__(self,
                 position_weight: float = 0.4,
                 card_weight: float = 0.3,
                 appearance_weight: float = 0.2,
                 size_weight: float = 0.1):
        """
        Initialize tracker with feature weights

        Args:
            position_weight: How much to weight position similarity
            card_weight: How much to weight card name consistency
            appearance_weight: How much to weight visual appearance
            size_weight: How much to weight bbox size similarity
        """
        self.position_weight = position_weight
        self.card_weight = card_weight
        self.appearance_weight = appearance_weight
        self.size_weight = size_weight

        self.tracked_troops = {}  # {troop_id: TroopTrack}
        self.next_id = 0

    def _compute_appearance_histogram(self, crop: np.ndarray) -> np.ndarray:
        """
        Compute color histogram of troop crop
        This is unique per troop even if they're the same card
        """
        if crop is None or crop.size == 0:
            return np.zeros(32)

        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Compute histogram (8 bins per channel)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])

        # Normalize
        hist = cv2.normalize(hist, hist).flatten()

        return hist

    def _compute_similarity(self,
                           track: 'TroopTrack',
                           detection: Dict,
                           timestamp: float) -> float:
        """
        Compute similarity score between existing track and new detection
        Uses multiple features to be robust against occlusion

        Returns:
            Similarity score 0-1 (higher = more similar)
        """
        score = 0.0

        # 1. Position similarity (predict where troop should be)
        predicted_pos = track.predict_position(timestamp)
        detected_pos = detection['center']

        # Euclidean distance
        pos_dist = np.sqrt((predicted_pos[0] - detected_pos[0])**2 +
                          (predicted_pos[1] - detected_pos[1])**2)

        # Convert to similarity (closer = higher score)
        max_distance = 100  # pixels
        pos_similarity = max(0, 1 - (pos_dist / max_distance))
        score += self.position_weight * pos_similarity

        # 2. Card name consistency (should not change!)
        card_match = 1.0 if track.stable_card == detection['card_name'] else 0.0
        score += self.card_weight * card_match

        # 3. Appearance similarity (visual fingerprint)
        det_hist = self._compute_appearance_histogram(detection.get('crop'))
        if det_hist is not None and track.appearance_hist is not None:
            # Correlation between histograms
            appearance_sim = cv2.compareHist(
                track.appearance_hist,
                det_hist,
                cv2.HISTCMP_CORREL
            )
            score += self.appearance_weight * max(0, appearance_sim)

        # 4. Size similarity (troops don't change size much)
        track_w, track_h = track.bbox_size
        det_bbox = detection['bbox']
        det_w, det_h = det_bbox[2] - det_bbox[0], det_bbox[3] - det_bbox[1]

        size_diff = abs(track_w - det_w) + abs(track_h - det_h)
        size_similarity = max(0, 1 - (size_diff / 100))
        score += self.size_weight * size_similarity

        return score

    def update(self, detections: List[Dict], timestamp: float):
        """
        Update tracks with new detections using Hungarian algorithm

        Handles:
        - Multiple troops close together
        - Overlapping HP bars
        - Occlusion
        - ID consistency
        """
        # Build cost matrix: tracks × detections
        track_ids = list(self.tracked_troops.keys())
        n_tracks = len(track_ids)
        n_detections = len(detections)

        if n_tracks == 0:
            # No existing tracks, create all new
            for det in detections:
                self._create_new_track(det, timestamp)
            return

        if n_detections == 0:
            # No detections, mark all as unseen
            for track in self.tracked_troops.values():
                track.frames_unseen += 1
            return

        # Compute similarity matrix
        similarity_matrix = np.zeros((n_tracks, n_detections))

        for i, track_id in enumerate(track_ids):
            track = self.tracked_troops[track_id]
            for j, detection in enumerate(detections):
                # Only match same team
                if track.team != detection['team']:
                    similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = self._compute_similarity(
                        track, detection, timestamp
                    )

        # Solve assignment problem (greedy for now, can use Hungarian later)
        matched_tracks = set()
        matched_detections = set()

        # Match highest similarities first
        while True:
            # Find max similarity
            max_val = similarity_matrix.max()
            if max_val < 0.3:  # Threshold for valid match
                break

            # Get indices
            track_idx, det_idx = np.unravel_index(
                similarity_matrix.argmax(),
                similarity_matrix.shape
            )

            # Match them
            track_id = track_ids[track_idx]
            detection = detections[det_idx]

            self.tracked_troops[track_id].update(detection, timestamp)

            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)

            # Zero out this row and column
            similarity_matrix[track_idx, :] = 0
            similarity_matrix[:, det_idx] = 0

        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                self._create_new_track(detection, timestamp)

        # Mark unmatched tracks as unseen
        for track_idx, track_id in enumerate(track_ids):
            if track_idx not in matched_tracks:
                self.tracked_troops[track_id].frames_unseen += 1

        # Clean up old tracks
        dead_ids = [
            tid for tid, track in self.tracked_troops.items()
            if track.frames_unseen > 5  # 2.5 seconds at 2fps
        ]
        for tid in dead_ids:
            del self.tracked_troops[tid]

    def _create_new_track(self, detection: Dict, timestamp: float):
        """Create a new track for unmatched detection"""
        track_id = self.next_id
        self.next_id += 1

        self.tracked_troops[track_id] = TroopTrack(
            troop_id=track_id,
            detection=detection,
            timestamp=timestamp
        )

    def get_tracked_state(self) -> List[Dict]:
        """Get current state of all tracked troops"""
        state = []
        for track in self.tracked_troops.values():
            if track.frames_unseen < 2:  # Only include recently seen
                state.append({
                    'id': track.troop_id,
                    'team': track.team,
                    'card': track.stable_card,
                    'position': track.get_position(),
                    'confidence': track.get_confidence(),
                    'age': track.age
                })
        return state


class TroopTrack:
    """Represents a tracked troop with history"""

    def __init__(self, troop_id: int, detection: Dict, timestamp: float):
        self.troop_id = troop_id
        self.team = detection['team']
        self.stable_card = detection['card_name']

        # History
        self.position_history = deque(maxlen=10)
        self.card_history = deque(maxlen=5)

        # Appearance
        crop = detection.get('crop')
        self.appearance_hist = self._compute_appearance_histogram(crop) if crop is not None else None

        # Size
        bbox = detection['bbox']
        self.bbox_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

        # Tracking state
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.frames_unseen = 0

        # Initial update
        self.update(detection, timestamp)

    def _compute_appearance_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Compute color histogram"""
        if crop is None or crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def update(self, detection: Dict, timestamp: float):
        """Update track with new detection"""
        self.position_history.append({
            'center': detection['center'],
            'timestamp': timestamp
        })

        self.card_history.append({
            'name': detection['card_name'],
            'confidence': detection['card_confidence']
        })

        # Update stable card (majority vote)
        from collections import Counter
        votes = Counter(c['name'] for c in self.card_history)
        self.stable_card = votes.most_common(1)[0][0]

        self.last_seen = timestamp
        self.frames_unseen = 0

        # Update appearance if provided
        if 'crop' in detection and detection['crop'] is not None:
            new_hist = self._compute_appearance_histogram(detection['crop'])
            if new_hist is not None:
                # Blend with existing (80% old, 20% new for stability)
                if self.appearance_hist is not None:
                    self.appearance_hist = 0.8 * self.appearance_hist + 0.2 * new_hist
                else:
                    self.appearance_hist = new_hist

    def predict_position(self, target_time: float) -> Tuple[int, int]:
        """Predict position at target time based on velocity"""
        if len(self.position_history) < 2:
            return self.position_history[-1]['center'] if self.position_history else (0, 0)

        # Get last two positions
        pos1 = self.position_history[-2]
        pos2 = self.position_history[-1]

        dt = pos2['timestamp'] - pos1['timestamp']
        if dt < 0.001:
            return pos2['center']

        # Calculate velocity
        vx = (pos2['center'][0] - pos1['center'][0]) / dt
        vy = (pos2['center'][1] - pos1['center'][1]) / dt

        # Predict
        dt_predict = target_time - pos2['timestamp']
        pred_x = int(pos2['center'][0] + vx * dt_predict)
        pred_y = int(pos2['center'][1] + vy * dt_predict)

        return (pred_x, pred_y)

    def get_position(self) -> Tuple[int, int]:
        """Get current position"""
        if self.position_history:
            return self.position_history[-1]['center']
        return (0, 0)

    def get_confidence(self) -> float:
        """Get average confidence for this track"""
        if self.card_history:
            return sum(c['confidence'] for c in self.card_history) / len(self.card_history)
        return 0.0

    @property
    def age(self) -> float:
        """How long this troop has been tracked"""
        return self.last_seen - self.first_seen
