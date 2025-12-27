"""
State encoding for RL agent
Converts game state into a format suitable for neural networks
"""

import numpy as np
from typing import List, Optional, Dict


class StateEncoder:
    """
    Encodes game state into a fixed-size vector for the RL agent
    """

    def __init__(self, grid_rows=32, grid_cols=18, max_cards=4):
        """
        Initialize state encoder

        Args:
            grid_rows: Number of rows in the arena grid
            grid_cols: Number of columns in the arena grid
            max_cards: Maximum number of cards in hand
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_cards = max_cards

        # Calculate state size
        self.elixir_size = 1  # Current elixir (0-10)
        self.hand_size = max_cards * 3  # 4 cards × (card_id, elixir_cost, available)
        self.enemy_grid_size = grid_rows * grid_cols  # Enemy troop presence
        self.ally_grid_size = grid_rows * grid_cols   # Ally troop presence
        self.tower_hp_size = 6  # HP for 6 towers (3 enemy + 3 ally)

        self.state_size = (
            self.elixir_size +
            self.hand_size +
            self.enemy_grid_size +
            self.ally_grid_size +
            self.tower_hp_size
        )

        print(f"State size: {self.state_size}")
        print(f"  - Elixir: {self.elixir_size}")
        print(f"  - Hand: {self.hand_size}")
        print(f"  - Enemy grid: {self.enemy_grid_size}")
        print(f"  - Ally grid: {self.ally_grid_size}")
        print(f"  - Tower HP: {self.tower_hp_size}")

    def encode_state(
        self,
        elixir: float,
        hand: List[Optional[Dict]],
        enemy_detections: List[Dict],
        ally_detections: List[Dict],
        tower_hp: Dict[str, float]
    ) -> np.ndarray:
        """
        Encode game state into a vector

        Args:
            elixir: Current elixir (0-10)
            hand: List of 4 card dicts (or None)
            enemy_detections: List of enemy troop detections
            ally_detections: List of ally troop detections
            tower_hp: Dictionary with tower HP values

        Returns:
            Encoded state vector
        """
        state = np.zeros(self.state_size, dtype=np.float32)
        idx = 0

        # 1. Encode elixir (normalized to 0-1)
        state[idx] = min(elixir, 10.0) / 10.0
        idx += 1

        # 2. Encode hand (4 cards)
        from detection.card_info import CARD_INFO

        # Create card name to ID mapping
        card_names = sorted(CARD_INFO.keys())
        card_to_id = {name: i for i, name in enumerate(card_names)}

        for i in range(self.max_cards):
            if i < len(hand) and hand[i] is not None:
                card = hand[i]
                card_name = card.get('card_name', 'unknown')

                # Card ID (normalized)
                card_id = card_to_id.get(card_name, 0)
                state[idx] = card_id / len(card_names)
                idx += 1

                # Elixir cost (normalized)
                elixir_cost = CARD_INFO.get(card_name, {}).get('elixir_cost', 0)
                state[idx] = elixir_cost / 10.0
                idx += 1

                # Available (0 or 1)
                state[idx] = 1.0 if card.get('available', True) else 0.0
                idx += 1
            else:
                idx += 3  # Skip empty slot

        # 3. Encode enemy positions (grid)
        enemy_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        for detection in enemy_detections:
            grid_pos = detection.get('grid')
            if grid_pos:
                row, col = grid_pos
                if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                    enemy_grid[row, col] = 1.0

        state[idx:idx + self.enemy_grid_size] = enemy_grid.flatten()
        idx += self.enemy_grid_size

        # 4. Encode ally positions (grid)
        ally_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        for detection in ally_detections:
            grid_pos = detection.get('grid')
            if grid_pos:
                row, col = grid_pos
                if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                    ally_grid[row, col] = 1.0

        state[idx:idx + self.ally_grid_size] = ally_grid.flatten()
        idx += self.ally_grid_size

        # 5. Encode tower HP (normalized to 0-1)
        # Max HP for normalization:
        # - Princess towers: ~2500 HP
        # - King tower: ~3500 HP
        # Using 4500 as safe upper bound to handle all towers
        max_hp = 4500.0

        # Enemy towers
        state[idx] = min(tower_hp.get('enemy_left_princess', 0.0), max_hp) / max_hp
        idx += 1
        state[idx] = min(tower_hp.get('enemy_king', 0.0), max_hp) / max_hp
        idx += 1
        state[idx] = min(tower_hp.get('enemy_right_princess', 0.0), max_hp) / max_hp
        idx += 1

        # Ally towers (same normalization as enemy)
        state[idx] = min(tower_hp.get('ally_left_princess', 0.0), max_hp) / max_hp
        idx += 1
        state[idx] = min(tower_hp.get('ally_king', 0.0), max_hp) / max_hp
        idx += 1
        state[idx] = min(tower_hp.get('ally_right_princess', 0.0), max_hp) / max_hp
        idx += 1

        return state
