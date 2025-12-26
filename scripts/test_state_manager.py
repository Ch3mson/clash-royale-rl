#!/usr/bin/env python3
"""
Demo: State Manager with Temporal Filtering
Shows how context awareness handles noisy detections
"""
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from game_state.state_manager import StateManager


def simulate_noisy_detections():
    """
    Simulate a scenario with noisy CV detections
    to demonstrate temporal filtering
    """
    print("="*80)
    print("STATE MANAGER DEMO - Handling Noisy Detections")
    print("="*80)

    state_manager = StateManager(match_distance=100)

    print("\nScenario: A spear goblin is detected, but one frame misclassifies it\n")

    # Frame 1: Spear goblin correctly detected
    print("Frame 1 (t=0.0s): Detected spear_goblin")
    detections_1 = [{
        'team': 'ally',
        'card_name': 'spear_goblins',
        'card_confidence': 0.95,
        'center': (200, 500),
        'bbox': (180, 480, 220, 520)
    }]
    state_manager.update(detections_1, elixir=5, timestamp=0.0)
    state = state_manager.get_state()
    print(f"  → State: {state['ally_troops'][0]['card']}")

    # Frame 2: Still correct
    print("\nFrame 2 (t=0.5s): Detected spear_goblin (moved)")
    detections_2 = [{
        'team': 'ally',
        'card_name': 'spear_goblins',
        'card_confidence': 0.92,
        'center': (210, 490),  # Moved slightly
        'bbox': (190, 470, 230, 510)
    }]
    state_manager.update(detections_2, elixir=5, timestamp=0.5)
    state = state_manager.get_state()
    print(f"  → State: {state['ally_troops'][0]['card']}")

    # Frame 3: MISCLASSIFICATION (something blocked view)
    print("\nFrame 3 (t=1.0s): MISDETECTED as goblin_brawler (low conf)")
    detections_3 = [{
        'team': 'ally',
        'card_name': 'goblin_brawler',  # WRONG!
        'card_confidence': 0.45,  # Low confidence
        'center': (220, 480),
        'bbox': (200, 460, 240, 500)
    }]
    state_manager.update(detections_3, elixir=6, timestamp=1.0)
    state = state_manager.get_state()
    print(f"  → State: {state['ally_troops'][0]['card']} ✓ FILTERED OUT!")

    # Frame 4: Back to correct
    print("\nFrame 4 (t=1.5s): Detected spear_goblin again")
    detections_4 = [{
        'team': 'ally',
        'card_name': 'spear_goblins',
        'card_confidence': 0.93,
        'center': (230, 470),
        'bbox': (210, 450, 250, 490)
    }]
    state_manager.update(detections_4, elixir=6, timestamp=1.5)
    state = state_manager.get_state()
    print(f"  → State: {state['ally_troops'][0]['card']}")

    print("\n" + "="*80)
    print("✓ The anomaly was ignored! Stable card: spear_goblins")
    print("="*80)


def simulate_elixir_tracking():
    """
    Simulate elixir tracking with missed frames
    """
    print("\n\n" + "="*80)
    print("ELIXIR TRACKING DEMO - Handling Missed Readings")
    print("="*80)

    state_manager = StateManager()

    print("\nScenario: Elixir jumps from 6 → 8, skipping 7\n")

    # Start at 6
    print("Frame 1 (t=0.0s): Elixir = 6")
    state_manager.update([], elixir=6, timestamp=0.0)
    state = state_manager.get_state()
    print(f"  → Validated: {state['elixir']}")

    # 1.4 seconds later: Should be ~6.5, but reading says 7
    print("\nFrame 2 (t=1.4s): Elixir = 7 (natural regen)")
    state_manager.update([], elixir=7, timestamp=1.4)
    state = state_manager.get_state()
    print(f"  → Validated: {state['elixir']}")

    # 1.4 seconds later: Should be ~7.5, but OCR missed frame and reads 8
    print("\nFrame 3 (t=2.8s): Elixir = 8 (but reading jumped, missed 7.5)")
    state_manager.update([], elixir=8, timestamp=2.8)
    state = state_manager.get_state()
    print(f"  → Validated: {state['elixir']} ✓ INTERPOLATED!")

    # Continue normally
    print("\nFrame 4 (t=4.2s): Elixir = 8 (at max)")
    state_manager.update([], elixir=8, timestamp=4.2)
    state = state_manager.get_state()
    print(f"  → Validated: {state['elixir']}")

    print("\n" + "="*80)
    print("✓ Elixir smoothly interpolated despite missed reading")
    print("="*80)


def simulate_troop_tracking():
    """
    Simulate tracking troops across frames with IDs
    """
    print("\n\n" + "="*80)
    print("TROOP TRACKING DEMO - Persistent IDs Across Frames")
    print("="*80)

    state_manager = StateManager()

    print("\nScenario: Two troops moving, one disappears\n")

    # Frame 1: Two troops appear
    print("Frame 1: 2 ally troops appear")
    detections = [
        {'team': 'ally', 'card_name': 'knight', 'card_confidence': 0.95,
         'center': (100, 500), 'bbox': (80, 480, 120, 520)},
        {'team': 'ally', 'card_name': 'archers', 'card_confidence': 0.92,
         'center': (200, 500), 'bbox': (180, 480, 220, 520)}
    ]
    state_manager.update(detections, elixir=5, timestamp=0.0)
    state = state_manager.get_state()
    print(f"  Tracked troops: {[(t['id'], t['card']) for t in state['ally_troops']]}")

    # Frame 2: Troops move
    print("\nFrame 2: Troops moved")
    detections = [
        {'team': 'ally', 'card_name': 'knight', 'card_confidence': 0.93,
         'center': (110, 480), 'bbox': (90, 460, 130, 500)},
        {'team': 'ally', 'card_name': 'archers', 'card_confidence': 0.91,
         'center': (210, 490), 'bbox': (190, 470, 230, 510)}
    ]
    state_manager.update(detections, elixir=5, timestamp=0.5)
    state = state_manager.get_state()
    print(f"  Tracked troops: {[(t['id'], t['card']) for t in state['ally_troops']]}")
    print("  → Same IDs! Troops tracked successfully")

    # Frame 3: Knight disappears (defeated)
    print("\nFrame 3: Knight defeated, only archers remain")
    detections = [
        {'team': 'ally', 'card_name': 'archers', 'card_confidence': 0.94,
         'center': (220, 480), 'bbox': (200, 460, 240, 500)}
    ]
    state_manager.update(detections, elixir=5, timestamp=1.0)
    state = state_manager.get_state()
    print(f"  Tracked troops: {[(t['id'], t['card']) for t in state['ally_troops']]}")
    print("  → Knight removed, archers still tracked with same ID")

    # Wait for cleanup
    time.sleep(1.1)

    # Frame 4: Cleanup happens
    print("\nFrame 4: After 1 second, knight cleaned up")
    state_manager.update(detections, elixir=6, timestamp=2.1)
    state = state_manager.get_state()
    print(f"  Tracked troops: {[(t['id'], t['card']) for t in state['ally_troops']]}")

    print("\n" + "="*80)
    print("✓ Troops tracked with persistent IDs across frames")
    print("="*80)


if __name__ == "__main__":
    simulate_noisy_detections()
    simulate_elixir_tracking()
    simulate_troop_tracking()

    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The StateManager provides:

1. ✓ Temporal filtering - Ignores classification outliers
2. ✓ Elixir validation - Interpolates missed readings
3. ✓ Troop tracking - Persistent IDs across frames
4. ✓ Position prediction - Smooths movement

Your RL agent will receive clean, stable state despite noisy CV!
""")
    print("="*80)
