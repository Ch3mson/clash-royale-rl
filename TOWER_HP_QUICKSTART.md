# Tower HP Detection - Quick Start

## Test It Now

```bash
# Test detection on live game
python3 scripts/test_tower_hp.py

# Test full integration with state encoding
python3 scripts/test_state_encoding_with_tower_hp.py
```

## Use It (3 lines of code)

```python
from detection.tower_hp_detector import TowerHPDetector

detector = TowerHPDetector()
screenshot = gc.take_screenshot()
tower_hp = detector.get_tower_hp(screenshot)

# tower_hp = {
#     'enemy_left_princess': 1890,
#     'enemy_king': 3312,
#     'enemy_right_princess': 0,
#     'ally_left_princess': 2030,
#     'ally_king': 0,
#     'ally_right_princess': 1850
# }
```

## Integrate with RL

```python
# In your RL training loop
from detection.tower_hp_detector import TowerHPDetector
from rl.state_encoder import StateEncoder

hp_detector = TowerHPDetector()
encoder = StateEncoder()

# Detect tower HP (throttled to every 2 seconds)
if time.time() - last_update > 2.0:
    tower_hp = hp_detector.get_tower_hp(screenshot)
    last_update = time.time()

# Encode state with tower HP
state = encoder.encode_state(
    elixir=elixir,
    hand=hand,
    enemy_detections=enemies,
    ally_detections=allies,
    tower_hp=tower_hp  # <-- Added!
)

# State size increased from 1165 to 1171 (+ 6 tower HP values)
```

## Files Created

- **[detection/tower_hp_detector.py](detection/tower_hp_detector.py)** - Core OCR detector
- **[scripts/test_tower_hp.py](scripts/test_tower_hp.py)** - Test on live game
- **[scripts/calibrate_tower_hp.py](scripts/calibrate_tower_hp.py)** - Calibration tool
- **[scripts/test_state_encoding_with_tower_hp.py](scripts/test_state_encoding_with_tower_hp.py)** - Integration test
- **[tower_hp_integration_example.py](tower_hp_integration_example.py)** - Code example
- **[TOWER_HP_INTEGRATION.md](TOWER_HP_INTEGRATION.md)** - Full guide
- **[TOWER_HP_SUMMARY.md](TOWER_HP_SUMMARY.md)** - Technical summary

## Files Modified

- **[rl/state_encoder.py](rl/state_encoder.py)** - Added tower HP (6 values) to state

## What Changed

**State vector size:** 1165 → 1171 (+6 tower HP values)

**State breakdown:**
```
[0]         Elixir
[1-12]      Hand (4 cards)
[13-588]    Enemy grid
[589-1164]  Ally grid
[1165-1170] Tower HP (NEW!)
```

## Dependencies

Requires pytesseract + tesseract:
```bash
pip install pytesseract
brew install tesseract  # macOS
```

## Performance Tip

Tower HP detection uses OCR (~50-100ms). Update every 2-3 seconds, not every frame:

```python
# Cache tower HP
if time.time() - last_update > 2.0:
    cached_hp = detector.get_tower_hp(screenshot)
    last_update = time.time()

# Use cached value
tower_hp = cached_hp
```

## Next Steps

When implementing RL:
1. Add tower HP detection to your battle loop
2. Use throttled updates (every 2-3 sec)
3. Implement reward shaping based on tower HP changes
4. Train agent with tower HP in state

See **[TOWER_HP_INTEGRATION.md](TOWER_HP_INTEGRATION.md)** for detailed guide.
