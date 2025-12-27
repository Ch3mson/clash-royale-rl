"""
Reward Calculator for Clash Royale RL

Calculates rewards based on game state changes, including tower HP damage.
"""

from typing import Dict, Optional
import numpy as np


class RewardCalculator:
    """
    Calculates rewards for RL agent based on game state changes
    """

    def __init__(
        self,
        tower_damage_reward: float = 0.01,
        tower_destroy_bonus: float = 10.0,
        elixir_advantage_reward: float = 0.1,
        troop_presence_reward: float = 0.05,
        win_reward: float = 100.0,
        loss_penalty: float = -100.0,
        draw_reward: float = 0.0
    ):
        """
        Initialize reward calculator

        Args:
            tower_damage_reward: Reward per HP damage to enemy towers
            tower_destroy_bonus: Bonus for destroying enemy tower
            elixir_advantage_reward: Reward for elixir advantage
            troop_presence_reward: Reward for having troops on field
            win_reward: Reward for winning battle
            loss_penalty: Penalty for losing battle
            draw_reward: Reward for draw
        """
        self.tower_damage_reward = tower_damage_reward
        self.tower_destroy_bonus = tower_destroy_bonus
        self.elixir_advantage_reward = elixir_advantage_reward
        self.troop_presence_reward = troop_presence_reward
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.draw_reward = draw_reward

    def calculate_reward(
        self,
        prev_tower_hp: Dict[str, int],
        curr_tower_hp: Dict[str, int],
        prev_elixir: float = None,
        curr_elixir: float = None,
        enemy_troop_count: int = 0,
        ally_troop_count: int = 0,
        battle_result: Optional[str] = None
    ) -> float:
        """
        Calculate reward based on state changes

        Args:
            prev_tower_hp: Previous tower HP dict
            curr_tower_hp: Current tower HP dict
            prev_elixir: Previous elixir value
            curr_elixir: Current elixir value
            enemy_troop_count: Number of enemy troops on field
            ally_troop_count: Number of ally troops on field
            battle_result: 'victory', 'defeat', 'draw', or None

        Returns:
            Total reward
        """
        reward = 0.0

        # 1. Tower HP rewards (most important!)
        reward += self._calculate_tower_hp_reward(prev_tower_hp, curr_tower_hp)

        # 2. Elixir advantage (small bonus for elixir management)
        if prev_elixir is not None and curr_elixir is not None:
            reward += self._calculate_elixir_reward(prev_elixir, curr_elixir)

        # 3. Troop presence (encourage having units on field)
        reward += self._calculate_troop_presence_reward(ally_troop_count, enemy_troop_count)

        # 4. Battle result (huge reward/penalty)
        if battle_result is not None:
            reward += self._calculate_battle_result_reward(battle_result)

        return reward

    def _calculate_tower_hp_reward(
        self,
        prev_tower_hp: Dict[str, int],
        curr_tower_hp: Dict[str, int]
    ) -> float:
        """
        Calculate reward from tower HP changes

        Positive rewards:
        - Damage enemy towers
        - Destroy enemy towers (bonus)

        Negative rewards:
        - Ally towers take damage
        - Ally towers destroyed (penalty)
        """
        reward = 0.0

        # Enemy tower damage = positive reward
        enemy_towers = ['enemy_left_princess', 'enemy_king', 'enemy_right_princess']
        for tower in enemy_towers:
            prev_hp = prev_tower_hp.get(tower, 0)
            curr_hp = curr_tower_hp.get(tower, 0)

            # Handle None values (OCR failed)
            if prev_hp is None:
                prev_hp = 0
            if curr_hp is None:
                curr_hp = 0

            damage = prev_hp - curr_hp

            if damage > 0:
                # Reward for damage
                reward += damage * self.tower_damage_reward

                # Bonus for destroying tower
                if prev_hp > 0 and curr_hp == 0:
                    reward += self.tower_destroy_bonus
                    print(f"[REWARD] Destroyed {tower}! Bonus: +{self.tower_destroy_bonus}")

        # Ally tower damage = negative reward
        ally_towers = ['ally_left_princess', 'ally_king', 'ally_right_princess']
        for tower in ally_towers:
            prev_hp = prev_tower_hp.get(tower, 0)
            curr_hp = curr_tower_hp.get(tower, 0)

            # Handle None values
            if prev_hp is None:
                prev_hp = 0
            if curr_hp is None:
                curr_hp = 0

            damage = prev_hp - curr_hp

            if damage > 0:
                # Penalty for damage
                reward -= damage * self.tower_damage_reward

                # Penalty for losing tower
                if prev_hp > 0 and curr_hp == 0:
                    reward -= self.tower_destroy_bonus
                    print(f"[REWARD] Lost {tower}! Penalty: -{self.tower_destroy_bonus}")

        return reward

    def _calculate_elixir_reward(self, prev_elixir: float, curr_elixir: float) -> float:
        """
        Small reward for maintaining elixir advantage

        Encourages not overspending and keeping elixir ready
        """
        # Slight bonus for having more elixir (but not too much to discourage spending)
        if curr_elixir >= 8:
            return -self.elixir_advantage_reward  # Penalty for hoarding
        elif curr_elixir >= 5:
            return self.elixir_advantage_reward * 0.5  # Small bonus for having resources
        else:
            return 0.0

    def _calculate_troop_presence_reward(
        self,
        ally_troop_count: int,
        enemy_troop_count: int
    ) -> float:
        """
        Small reward for having troops on field

        Encourages aggressive play and board presence
        """
        reward = 0.0

        # Bonus for having troops attacking
        if ally_troop_count > 0:
            reward += self.troop_presence_reward * min(ally_troop_count, 5)

        # Small penalty for being overwhelmed
        if enemy_troop_count > ally_troop_count + 3:
            reward -= self.troop_presence_reward * 2

        return reward

    def _calculate_battle_result_reward(self, result: str) -> float:
        """Calculate reward based on battle outcome"""
        if result == 'victory':
            return self.win_reward
        elif result == 'defeat':
            return self.loss_penalty
        elif result == 'draw':
            return self.draw_reward
        return 0.0

    def calculate_step_reward(
        self,
        prev_state: Dict,
        curr_state: Dict,
        action_taken: int = None
    ) -> float:
        """
        Convenience method to calculate reward from full state dicts

        Args:
            prev_state: Previous state dict with keys:
                - tower_hp: Dict of tower HP values
                - elixir: float
                - enemy_troops: List of enemy troops
                - ally_troops: List of ally troops
            curr_state: Current state dict (same structure)
            action_taken: Action index that was taken (unused for now)

        Returns:
            Reward value
        """
        return self.calculate_reward(
            prev_tower_hp=prev_state.get('tower_hp', {}),
            curr_tower_hp=curr_state.get('tower_hp', {}),
            prev_elixir=prev_state.get('elixir'),
            curr_elixir=curr_state.get('elixir'),
            enemy_troop_count=len(curr_state.get('enemy_troops', [])),
            ally_troop_count=len(curr_state.get('ally_troops', [])),
            battle_result=curr_state.get('battle_result')
        )
