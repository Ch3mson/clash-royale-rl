#!/usr/bin/env python3
"""
Test state encoding with tower HP integration

This script demonstrates how to integrate TowerHPDetector with StateEncoder
for RL training.

Usage:
    python3 scripts/test_state_encoding_with_tower_hp.py
    python3 scripts/test_state_encoding_with_tower_hp.py --instance 0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
from controllers.game_controller import GameController
from detection.tower_hp_detector import TowerHPDetector
from detection.card_hand_detector import CardHandDetector
from detection.elixir_detector import ElixirDetector
from rl.state_encoder import StateEncoder


def test_state_encoding(instance_id: int = 0):
    """
    Test state encoding with tower HP detection integrated

    Args:
        instance_id: BlueStacks instance ID
    """
    print("\n" + "="*60)
    print("State Encoding with Tower HP - Integration Test")
    print("="*60)

    # Initialize components
    gc = GameController(instance_id)
    tower_hp_detector = TowerHPDetector()
    hand_detector = CardHandDetector()
    elixir_detector = ElixirDetector()
    state_encoder = StateEncoder(grid_rows=32, grid_cols=18, max_cards=4)

    # Capture screenshot
    print("\nCapturing screenshot from BlueStacks...")
    screenshot = gc.take_screenshot()
    if screenshot is None:
        print("❌ Failed to capture screenshot")
        return

    print(f"✅ Screenshot captured ({screenshot.shape[1]}x{screenshot.shape[0]})")

    # Detect all game state components
    print("\n" + "-"*60)
    print("Detecting game state components...")
    print("-"*60)

    # 1. Elixir
    elixir = elixir_detector.get_elixir(screenshot, verbose=False)
    if elixir is not None:
        print(f"✅ Elixir: {elixir}")
    else:
        print("❌ Elixir: Not detected")
        elixir = 5.0  # Default for testing

    # 2. Hand
    hand = hand_detector.get_hand(screenshot, verbose=False)
    print(f"✅ Hand: {len([c for c in hand if c is not None])}/4 cards detected")
    for i, card in enumerate(hand):
        if card:
            print(f"   Slot {i}: {card['card_name']} (conf: {card['confidence']:.2f})")

    # 3. Tower HP
    print("\nDetecting tower HP...")
    tower_hp = tower_hp_detector.get_tower_hp(screenshot, verbose=False)

    print("\n🔴 Enemy Towers:")
    for tower in ['enemy_left_princess', 'enemy_king', 'enemy_right_princess']:
        hp = tower_hp.get(tower)
        tower_display = tower.replace('enemy_', '').replace('_', ' ').title()
        if hp is not None and hp > 0:
            print(f"   {tower_display:20s}: {hp:4d} HP")
        elif hp == 0:
            print(f"   {tower_display:20s}: Destroyed/Not Activated")
        else:
            print(f"   {tower_display:20s}: ❌ Not detected")

    print("\n🔵 Ally Towers:")
    for tower in ['ally_left_princess', 'ally_king', 'ally_right_princess']:
        hp = tower_hp.get(tower)
        tower_display = tower.replace('ally_', '').replace('_', ' ').title()
        if hp is not None and hp > 0:
            print(f"   {tower_display:20s}: {hp:4d} HP")
        elif hp == 0:
            print(f"   {tower_display:20s}: Destroyed/Not Activated")
        else:
            print(f"   {tower_display:20s}: ❌ Not detected")

    # 4. Encode state (no troop detections for this test)
    print("\n" + "-"*60)
    print("Encoding state vector...")
    print("-"*60)

    # For this test, we'll use empty troop detections
    enemy_detections = []
    ally_detections = []

    state_vector = state_encoder.encode_state(
        elixir=elixir,
        hand=hand,
        enemy_detections=enemy_detections,
        ally_detections=ally_detections,
        tower_hp=tower_hp
    )

    print(f"\n✅ State vector shape: {state_vector.shape}")
    print(f"   Expected size: {state_encoder.state_size}")
    print(f"   Match: {'✅' if state_vector.shape[0] == state_encoder.state_size else '❌'}")

    # Show breakdown of state vector
    print("\n" + "-"*60)
    print("State Vector Breakdown:")
    print("-"*60)

    idx = 0

    # Elixir
    print(f"Elixir (idx {idx}): {state_vector[idx]:.3f} (raw: {elixir})")
    idx += 1

    # Hand
    print(f"\nHand (idx {idx}-{idx + state_encoder.hand_size - 1}):")
    for i in range(4):
        card_idx = idx + i * 3
        if i < len(hand) and hand[i] is not None:
            card_name = hand[i]['card_name']
            print(f"  Card {i} ({card_name}):")
            print(f"    Card ID: {state_vector[card_idx]:.3f}")
            print(f"    Elixir:  {state_vector[card_idx + 1]:.3f}")
            print(f"    Available: {state_vector[card_idx + 2]:.3f}")
        else:
            print(f"  Card {i}: Empty")
    idx += state_encoder.hand_size

    # Skip grids (too long to print)
    print(f"\nEnemy grid (idx {idx}-{idx + state_encoder.enemy_grid_size - 1}): {state_encoder.enemy_grid_size} values")
    idx += state_encoder.enemy_grid_size

    print(f"Ally grid (idx {idx}-{idx + state_encoder.ally_grid_size - 1}): {state_encoder.ally_grid_size} values")
    idx += state_encoder.ally_grid_size

    # Tower HP
    print(f"\nTower HP (idx {idx}-{idx + state_encoder.tower_hp_size - 1}):")
    tower_names = [
        'enemy_left_princess', 'enemy_king', 'enemy_right_princess',
        'ally_left_princess', 'ally_king', 'ally_right_princess'
    ]
    for i, tower_name in enumerate(tower_names):
        hp_normalized = state_vector[idx + i]
        hp_raw = tower_hp.get(tower_name, 0)
        tower_display = tower_name.replace('_', ' ').title()
        print(f"  {tower_display:25s}: {hp_normalized:.3f} (raw: {hp_raw})")

    print("\n" + "="*60)
    print("✅ Integration test complete!")
    print("="*60)
    print("\nThe tower HP detector is now integrated with the state encoder.")
    print("When you implement RL training, you can use this pattern:")
    print("\n  1. Capture screenshot")
    print("  2. Detect: elixir, hand, troops, tower_hp")
    print("  3. Encode state: state_encoder.encode_state(...)")
    print("  4. Feed state vector to RL agent")


def main():
    parser = argparse.ArgumentParser(description="Test state encoding with tower HP")
    parser.add_argument(
        "--instance",
        type=int,
        default=0,
        help="BlueStacks instance ID"
    )

    args = parser.parse_args()

    test_state_encoding(args.instance)


if __name__ == "__main__":
    main()
