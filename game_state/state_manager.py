"""
State Manager with Temporal Filtering
Provides stable, context-aware game state to the RL agent
"""
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter, deque


class TroopState:
    """Tracks a single troop across multiple frames"""

    def __init__(self, troop_id: int, detection: Dict, timestamp: float):
        self.id = troop_id
        self.team = detection['team']  # 'ally' or 'enemy'

        # History buffers (keep last N frames)
        self.position_history = deque(maxlen=10)
        self.classification_history = deque(maxlen=5)

        # Timestamps
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.frames_since_seen = 0

        # Add initial detection
        self.update(detection, timestamp)

    def update(self, detection: Dict, timestamp: float):
        """Update troop state with new detection"""
        self.position_history.append({
            'center': detection['center'],
            'bbox': detection['bbox'],
            'timestamp': timestamp
        })

        self.classification_history.append({
            'card_name': detection['card_name'],
            'confidence': detection['card_confidence'],
            'timestamp': timestamp
        })

        self.last_seen = timestamp
        self.frames_since_seen = 0

    def get_stable_card(self, confidence_threshold: float = 0.8) -> str:
        """
        Get most reliable card classification using voting + confidence

        Strategy:
        1. If recent high-confidence prediction exists, use it
        2. Otherwise, use majority vote from recent history
        3. Ignore low-confidence outliers
        """
        if not self.classification_history:
            return "unknown"

        # Check for recent high-confidence prediction
        for classification in reversed(self.classification_history):
            if classification['confidence'] >= confidence_threshold:
                return classification['card_name']

        # Use majority vote (ignores outliers)
        votes = Counter(c['card_name'] for c in self.classification_history)
        most_common, count = votes.most_common(1)[0]

        # Require at least 60% agreement
        if count >= len(self.classification_history) * 0.6:
            return most_common

        # Fallback: highest confidence prediction
        return max(self.classification_history,
                  key=lambda x: x['confidence'])['card_name']

    def get_current_position(self) -> Tuple[int, int]:
        """Get most recent position"""
        if self.position_history:
            return self.position_history[-1]['center']
        return (0, 0)

    def predict_position(self, dt: float = 0.1) -> Tuple[int, int]:
        """Predict position for next frame based on velocity"""
        if len(self.position_history) < 2:
            return self.get_current_position()

        # Calculate velocity from last 2 positions
        pos1 = self.position_history[-2]['center']
        pos2 = self.position_history[-1]['center']
        t1 = self.position_history[-2]['timestamp']
        t2 = self.position_history[-1]['timestamp']

        if t2 - t1 < 0.001:  # Avoid division by zero
            return pos2

        vx = (pos2[0] - pos1[0]) / (t2 - t1)
        vy = (pos2[1] - pos1[1]) / (t2 - t1)

        # Predict
        pred_x = int(pos2[0] + vx * dt)
        pred_y = int(pos2[1] + vy * dt)

        return (pred_x, pred_y)

    def is_alive(self, max_age: float = 1.0) -> bool:
        """Check if troop should still be tracked"""
        age = time.time() - self.last_seen
        return age < max_age


class ElixirTracker:
    """Tracks elixir with physics-based validation"""

    ELIXIR_REGEN_RATE = 1.0 / 2.8  # 1 elixir per 2.8 seconds
    MAX_ELIXIR = 10

    def __init__(self):
        self.history = deque(maxlen=30)  # Keep last 30 readings
        self.validated_value = 0
        self.last_update_time = None

    def update(self, reading: int, timestamp: float) -> int:
        """
        Validate and smooth elixir reading

        Returns the validated elixir value to use
        """
        if self.last_update_time is None:
            # First reading
            self.validated_value = reading
            self.last_update_time = timestamp
            self.history.append((timestamp, reading))
            return reading

        time_delta = timestamp - self.last_update_time

        # Calculate expected range
        max_natural_gain = time_delta * self.ELIXIR_REGEN_RATE
        max_possible = min(self.MAX_ELIXIR,
                          self.validated_value + max_natural_gain)
        min_possible = 0  # Could have spent everything

        # Validate reading
        if min_possible <= reading <= max_possible:
            # Reading is plausible
            self.validated_value = reading
            self.last_update_time = timestamp
            self.history.append((timestamp, reading))
            return reading

        # Reading seems implausible
        # Check if it's a missed frame (jumped by 1-2)
        expected = self.validated_value + max_natural_gain

        if reading > max_possible and reading - expected <= 2:
            # Likely missed a frame, interpolate
            interpolated = int(round(expected))
            self.validated_value = interpolated
            self.last_update_time = timestamp
            self.history.append((timestamp, interpolated))
            return interpolated

        # Very implausible reading - stick with last validated
        # (OCR might have failed)
        return self.validated_value

    def get_elixir(self) -> int:
        """Get current validated elixir"""
        return self.validated_value


class StateManager:
    """
    Manages complete game state with temporal filtering
    Provides stable state to RL agent despite noisy CV detections
    """

    def __init__(self, match_distance: float = 50.0):
        self.troops = {}  # {troop_id: TroopState}
        self.next_troop_id = 0
        self.match_distance = match_distance  # pixels

        self.elixir_tracker = ElixirTracker()

        self.current_timestamp = 0.0

    def update(self, detections: List[Dict], elixir: int, timestamp: float = None):
        """
        Update state with new detections

        Args:
            detections: List of detections from TroopDetector
            elixir: Raw elixir reading from ElixirDetector
            timestamp: Current time (or None for auto)
        """
        if timestamp is None:
            timestamp = time.time()

        self.current_timestamp = timestamp

        # Update elixir
        validated_elixir = self.elixir_tracker.update(elixir, timestamp)

        # Match new detections to existing troops
        matched_ids = set()

        for detection in detections:
            # Find closest existing troop of same team
            closest_id = None
            closest_dist = float('inf')

            det_pos = detection['center']

            for troop_id, troop in self.troops.items():
                if troop.team != detection['team']:
                    continue  # Different team

                if not troop.is_alive():
                    continue  # Troop too old

                # Calculate distance
                troop_pos = troop.predict_position()
                dist = np.sqrt((det_pos[0] - troop_pos[0])**2 +
                             (det_pos[1] - troop_pos[1])**2)

                if dist < closest_dist and dist < self.match_distance:
                    closest_dist = dist
                    closest_id = troop_id

            if closest_id is not None:
                # Update existing troop
                self.troops[closest_id].update(detection, timestamp)
                matched_ids.add(closest_id)
            else:
                # New troop
                new_id = self.next_troop_id
                self.next_troop_id += 1
                self.troops[new_id] = TroopState(new_id, detection, timestamp)
                matched_ids.add(new_id)

        # Clean up old troops
        dead_ids = [tid for tid, troop in self.troops.items()
                   if not troop.is_alive() and tid not in matched_ids]
        for tid in dead_ids:
            del self.troops[tid]

    def get_state(self) -> Dict:
        """
        Get current stable game state for RL agent

        Returns:
            Dictionary containing:
                - elixir: int
                - ally_troops: List[Dict]
                - enemy_troops: List[Dict]
                - total_allies: int
                - total_enemies: int
        """
        ally_troops = []
        enemy_troops = []

        for troop in self.troops.values():
            if not troop.is_alive():
                continue

            troop_info = {
                'id': troop.id,
                'card': troop.get_stable_card(),
                'position': troop.get_current_position(),
                'age': self.current_timestamp - troop.first_seen
            }

            if troop.team == 'ally':
                ally_troops.append(troop_info)
            else:
                enemy_troops.append(troop_info)

        return {
            'elixir': self.elixir_tracker.get_elixir(),
            'ally_troops': ally_troops,
            'enemy_troops': enemy_troops,
            'total_allies': len(ally_troops),
            'total_enemies': len(enemy_troops),
            'timestamp': self.current_timestamp
        }

    def get_troop_counts(self) -> Dict[str, Dict[str, int]]:
        """Get troop counts by card type"""
        state = self.get_state()

        ally_counts = Counter(t['card'] for t in state['ally_troops'])
        enemy_counts = Counter(t['card'] for t in state['enemy_troops'])

        return {
            'ally': dict(ally_counts),
            'enemy': dict(enemy_counts)
        }
