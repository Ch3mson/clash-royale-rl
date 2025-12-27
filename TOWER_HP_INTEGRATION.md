# Tower HP Integration Guide

Tower HP detection is now integrated with the RL state encoder. This guide shows how to use it.

## Quick Test

Test that tower HP detection is working correctly:

```bash
# Test tower HP detection alone
python3 scripts/test_tower_hp.py

# Test tower HP + state encoding integration
python3 scripts/test_state_encoding_with_tower_hp.py
```

## Components

### 1. TowerHPDetector

Located in `detection/tower_hp_detector.py`

Detects HP values for all 6 towers using OCR:
- `enemy_left_princess`
- `enemy_king`
- `enemy_right_princess`
- `ally_left_princess`
- `ally_king`
- `ally_right_princess`

**Usage:**
```python
from detection.tower_hp_detector import TowerHPDetector

detector = TowerHPDetector()
screenshot = gc.take_screenshot()

# Returns dict with tower names -> HP values (or None/0 if destroyed)
tower_hp = detector.get_tower_hp(screenshot, verbose=True)

# Example output:
# {
#     'enemy_left_princess': 1890,
#     'enemy_king': 3312,
#     'enemy_right_princess': 0,  # Destroyed
#     'ally_left_princess': 2030,
#     'ally_king': 0,  # Not activated yet
#     'ally_right_princess': 1850
# }
```

### 2. StateEncoder (Updated)

Located in `rl/state_encoder.py`

Now includes tower HP in the state vector:
- State size increased by 6 (one value per tower)
- All tower HP values normalized to 0-1 range (max 4500 HP)
- Properly handles destroyed towers (0 HP)

**State vector breakdown:**
```
Index | Component           | Size
------|---------------------|-------
0     | Elixir              | 1
1-12  | Hand (4 cards × 3)  | 12
13-588| Enemy grid          | 576
589-1164| Ally grid         | 576
1165-1170| Tower HP         | 6
      | TOTAL               | 1171
```

**Usage:**
```python
from rl.state_encoder import StateEncoder

encoder = StateEncoder(grid_rows=32, grid_cols=18, max_cards=4)

state_vector = encoder.encode_state(
    elixir=current_elixir,
    hand=current_hand,
    enemy_detections=enemy_troops,
    ally_detections=ally_troops,
    tower_hp=tower_hp  # Pass tower HP dict from TowerHPDetector
)

# state_vector is now ready to feed into RL agent
```

## Integration into Game Loop

When you implement RL training, add tower HP detection to your battle loop:

```python
# In main.py or your training loop

from detection.tower_hp_detector import TowerHPDetector
from rl.state_encoder import StateEncoder

# Initialize
tower_hp_detector = TowerHPDetector()
state_encoder = StateEncoder()

# In battle loop
while in_battle:
    screenshot = gc.take_screenshot()

    # Detect all state components
    elixir = elixir_detector.get_elixir(screenshot)
    hand = hand_detector.get_hand(screenshot)
    tower_hp = tower_hp_detector.get_tower_hp(screenshot)  # ADD THIS

    # Get troop detections from YOLO
    all_detections = card_detector.detect(screenshot)
    enemy_detections = [d for d in all_detections if d['class_name'].startswith('enemy')]
    ally_detections = [d for d in all_detections if d['class_name'].startswith('ally')]

    # Encode state for RL agent
    state = state_encoder.encode_state(
        elixir=elixir,
        hand=hand,
        enemy_detections=enemy_detections,
        ally_detections=ally_detections,
        tower_hp=tower_hp  # Pass tower HP
    )

    # Use state for RL
    action = rl_agent.select_action(state)
    # ... execute action, get reward, etc.
```

## Performance Considerations

Tower HP detection uses OCR which is slower than template matching. Consider:

1. **Update frequency**: Don't detect tower HP every frame
   ```python
   # Update tower HP every 2-3 seconds instead of every frame
   if time.time() - last_tower_hp_update > 2.0:
       tower_hp = tower_hp_detector.get_tower_hp(screenshot)
       last_tower_hp_update = time.time()
   ```

2. **Caching**: Tower HP changes slowly, cache the last reading
   ```python
   # Use cached value if recent detection exists
   if cached_tower_hp is not None:
       tower_hp = cached_tower_hp
   ```

3. **Parallel detection**: If needed, run OCR in a separate thread
   ```python
   # Advanced: Use threading for OCR to avoid blocking
   import threading

   def update_tower_hp_async():
       global cached_tower_hp
       screenshot = gc.take_screenshot()
       cached_tower_hp = tower_hp_detector.get_tower_hp(screenshot)

   # Run in background
   threading.Thread(target=update_tower_hp_async, daemon=True).start()
   ```

## Calibration

If tower HP regions need adjustment (e.g., different screen resolution):

```bash
# Visualize current regions on a screenshot
python3 scripts/calibrate_tower_hp.py

# Or use a specific screenshot
python3 scripts/calibrate_tower_hp.py --screenshot path/to/screenshot.png
```

This will generate:
- `tower_hp_regions_visualized.png` - Full screenshot with colored boxes
- `tower_hp_crops/` - Individual cropped regions for each tower

Edit coordinates in `detection/tower_hp_detector.py` if needed:
```python
self.TOWER_REGIONS = {
    'enemy_left_princess': (145, 170, 202, 192),  # (x1, y1, x2, y2)
    'enemy_king': (340, 19, 402, 44),
    # ... etc
}
```

## Reward Shaping with Tower HP

Now that you have tower HP in the state, you can use it for reward calculation:

```python
def calculate_reward(prev_state, new_state):
    """
    Calculate reward based on tower HP changes

    Positive rewards:
    - Damage enemy towers
    - Destroy enemy towers (bonus)

    Negative rewards:
    - Ally towers take damage
    - Ally towers destroyed (penalty)
    """
    reward = 0

    # Enemy tower damage = positive reward
    for tower in ['enemy_left_princess', 'enemy_king', 'enemy_right_princess']:
        prev_hp = prev_state['tower_hp'].get(tower, 0)
        new_hp = new_state['tower_hp'].get(tower, 0)
        damage = prev_hp - new_hp

        if damage > 0:
            reward += damage * 0.01  # Small reward for damage

        # Bonus for destroying tower
        if prev_hp > 0 and new_hp == 0:
            reward += 10.0  # Large bonus

    # Ally tower damage = negative reward
    for tower in ['ally_left_princess', 'ally_king', 'ally_right_princess']:
        prev_hp = prev_state['tower_hp'].get(tower, 0)
        new_hp = new_state['tower_hp'].get(tower, 0)
        damage = prev_hp - new_hp

        if damage > 0:
            reward -= damage * 0.01  # Small penalty for damage

        # Large penalty for losing tower
        if prev_hp > 0 and new_hp == 0:
            reward -= 10.0

    return reward
```

## Troubleshooting

### Tower HP not detecting

1. Make sure you're in a battle (not main menu)
2. Run with `verbose=True` to see OCR output
3. Check if tesseract is installed: `brew list tesseract`
4. Verify regions are correct: `python3 scripts/calibrate_tower_hp.py`

### Inaccurate readings

1. Check if regions are capturing HP numbers correctly
2. Verify you're not including tower level in the crop
3. Try adjusting preprocessing strategy in `tower_hp_detector.py`

### Performance issues

1. Reduce update frequency (every 2-3 seconds instead of every frame)
2. Use caching for recent readings
3. Consider running OCR in background thread

## Current Status

✅ **Completed:**
- Tower HP detection working for all 6 towers
- OCR with multiple preprocessing strategies
- State encoder integration
- Test scripts created
- Calibration tool working

🔄 **Next Steps (when implementing RL):**
- Add tower HP detection to main game loop
- Implement reward shaping using tower HP
- Add caching/throttling for performance
- Train RL agent with tower HP state
