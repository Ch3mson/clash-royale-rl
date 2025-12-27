# Tower HP Detection - Summary

## What Was Built

Tower HP detection system that uses OCR to read HP values for all 6 towers from fixed screen positions.

## Files Created/Modified

### New Files
1. **`detection/tower_hp_detector.py`**
   - Core OCR-based tower HP detection
   - Uses pytesseract with multiple preprocessing strategies
   - Handles 6 towers: 3 enemy + 3 ally
   - Returns 0 for destroyed/inactive towers, HP value otherwise

2. **`scripts/test_tower_hp.py`**
   - Test tower HP detection on live BlueStacks screen
   - Shows detected HP for all towers
   - Optional `--save-crops` to debug region positioning

3. **`scripts/calibrate_tower_hp.py`**
   - Interactive calibration tool
   - Draws colored boxes on full screenshot showing tower regions
   - Saves cropped regions for detailed inspection
   - Use with `--screenshot` to specify exact screenshot

4. **`scripts/test_state_encoding_with_tower_hp.py`**
   - Integration test showing tower HP + state encoding
   - Demonstrates full workflow for RL training

5. **`TOWER_HP_INTEGRATION.md`**
   - Complete integration guide
   - Performance tips (caching, update frequency)
   - Reward shaping examples
   - Troubleshooting guide

6. **`TOWER_HP_SUMMARY.md`** (this file)
   - Quick reference summary

### Modified Files
1. **`rl/state_encoder.py`**
   - Added `tower_hp_size = 6` to state size calculation
   - Fixed tower HP normalization (was hardcoded to 1.0 for ally towers)
   - Now properly normalizes all 6 tower HP values to 0-1 range

## Current Tower Coordinates (720x1280 screen)

```python
TOWER_REGIONS = {
    # Enemy towers (top of screen)
    'enemy_left_princess': (145, 170, 202, 192),
    'enemy_king': (340, 19, 402, 44),
    'enemy_right_princess': (524, 170, 581, 192),

    # Ally towers (bottom of screen)
    'ally_left_princess': (145, 794, 202, 816),
    'ally_king': (338, 968, 400, 994),
    'ally_right_princess': (524, 794, 581, 816),
}
```

These coordinates were manually calibrated to:
- Exclude tower level number (only capture HP)
- Capture the bright white HP text
- Work for both activated and non-activated king towers

## How It Works

1. **Crop regions** - Extract small regions from fixed screen positions
2. **Preprocess** - Convert to grayscale, threshold to isolate white text, upscale 6x
3. **OCR** - Run tesseract with multiple PSM modes (8, 13, 7, 6)
4. **Extract number** - Parse OCR output, validate range (10-5000 HP)
5. **Return** - Dict with tower names -> HP values

## Testing

```bash
# Test tower HP detection alone
python3 scripts/test_tower_hp.py

# Save crops for debugging
python3 scripts/test_tower_hp.py --save-crops

# Visualize regions on screenshot
python3 scripts/calibrate_tower_hp.py

# Test full integration with state encoding
python3 scripts/test_state_encoding_with_tower_hp.py
```

## State Vector Integration

Tower HP is now included in the RL state vector:

**Before**: 1165 values (elixir + hand + grids)
**After**: 1171 values (+ 6 tower HP values)

State breakdown:
```
[0]         Elixir (normalized 0-1)
[1-12]      Hand (4 cards × 3 values each)
[13-588]    Enemy troop grid (32×18)
[589-1164]  Ally troop grid (32×18)
[1165-1170] Tower HP (6 towers, normalized 0-1)
```

## Usage Example

```python
from detection.tower_hp_detector import TowerHPDetector
from rl.state_encoder import StateEncoder

# Initialize
hp_detector = TowerHPDetector()
encoder = StateEncoder()

# In battle loop
screenshot = gc.take_screenshot()
tower_hp = hp_detector.get_tower_hp(screenshot)

# tower_hp = {
#     'enemy_left_princess': 1890,
#     'enemy_king': 3312,
#     'enemy_right_princess': 0,  # Destroyed
#     'ally_left_princess': 2030,
#     'ally_king': 0,  # Not activated
#     'ally_right_princess': 1850
# }

# Encode for RL
state = encoder.encode_state(
    elixir=elixir,
    hand=hand,
    enemy_detections=enemies,
    ally_detections=allies,
    tower_hp=tower_hp  # Pass tower HP dict
)
```

## Performance Notes

OCR is slower than template matching (~50-100ms per detection). Recommendations:

1. **Update frequency**: Every 2-3 seconds (tower HP changes slowly)
2. **Caching**: Reuse last reading between updates
3. **Threading**: Run OCR in background if needed

Example throttling:
```python
last_update = 0
cached_hp = None

# In loop
if time.time() - last_update > 2.0:
    cached_hp = hp_detector.get_tower_hp(screenshot)
    last_update = time.time()

# Use cached_hp for state encoding
```

## Known Limitations

1. **OCR dependency**: Requires tesseract installed (`brew install tesseract`)
2. **Fixed resolution**: Coordinates calibrated for 720x1280 screen
3. **Language**: Assumes English (numbers are universal but UI might vary)
4. **Bright text**: Optimized for white HP numbers on dark background

## Troubleshooting

**No HP detected:**
- Check if in battle (main menu has no towers)
- Verify tesseract installed: `which tesseract`
- Run calibration: `python3 scripts/calibrate_tower_hp.py`

**Wrong HP values:**
- Regions might include tower level - re-crop to exclude level
- Check if text is white on dark (preprocessing assumes this)
- Try different screenshot with clear HP visible

**Performance issues:**
- Reduce update frequency (2-3 seconds)
- Use caching between updates
- Consider threading for OCR

## Next Steps (For RL Training)

When implementing RL training:

1. Add tower HP detection to battle loop in `main.py`
2. Use cached/throttled updates for performance
3. Implement reward shaping based on tower HP changes
4. Train agent with tower HP in state vector

See `TOWER_HP_INTEGRATION.md` for detailed integration guide.
