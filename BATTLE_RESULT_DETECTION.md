# Battle Result Detection

This guide explains how to set up and use the battle result detection system for the Clash Royale RL bot.

## Overview

The battle result detector automatically identifies when a battle ends and determines the outcome:
- **Victory** - You won the battle
- **Defeat** - You lost the battle
- **Draw** - Battle ended in a tie

This is essential for the RL reward function, as it provides the primary signal for learning.

## How It Works

The detector uses **template matching** to identify result screens:

1. During battle, it continuously checks for result text/icons in the top-center of screen
2. When a result template matches with high confidence (>70%), it returns the result type
3. The result is logged and can be used for RL rewards

## Setup Instructions

### 1. Create Result Templates

You need to create templates for each result type. This only needs to be done once.

**For Victory:**
```bash
# 1. Play a battle and WIN
# 2. When the victory screen appears, run:
python3 scripts/create_result_templates.py --type victory
```

**For Defeat:**
```bash
# 1. Play a battle and LOSE
# 2. When the defeat screen appears, run:
python3 scripts/create_result_templates.py --type defeat
```

**For Draw:**
```bash
# 1. Play a battle that ends in a DRAW
# 2. When the draw screen appears, run:
python3 scripts/create_result_templates.py --type draw
```

The templates are saved to `detection/result_templates/`.

### 2. Verify Templates

Check that templates were created correctly:
```bash
ls -la detection/result_templates/
```

You should see:
```
victory.png    # or victory_1.png, victory_2.png, etc.
defeat.png
draw.png
```

### 3. Test Detection

Test the detector on a result screen:

```bash
# Capture from current BlueStacks screen:
python3 scripts/test_result_detection.py

# Or test on a saved screenshot:
python3 scripts/test_result_detection.py --image path/to/screenshot.png
```

If detection works, you should see:
```
✅ RESULT DETECTED: VICTORY
```

## Usage in Main Bot

The result detector is automatically integrated into `main.py`:

```python
# In handle_battle():
result = self.result_detector.detect_result(screenshot, threshold=0.7, verbose=False)
if result is not None:
    self.battle_result = result  # 'victory', 'defeat', or 'draw'
    print(f"[BATTLE END] Result: {result.upper()}")
```

## Using Results for RL

The battle result is stored in `agent.battle_result` and can be used for rewards:

```python
# Example reward function
def calculate_reward(self):
    if self.battle_result == 'victory':
        return 1.0
    elif self.battle_result == 'defeat':
        return -1.0
    elif self.battle_result == 'draw':
        return 0.0
    else:
        return 0.0  # Battle ongoing
```

## Troubleshooting

### "No result detected" error

**Possible causes:**
1. **No templates created** - Run `create_result_templates.py` for each result type
2. **Wrong screen** - Make sure you're on the actual result screen (not battle or menu)
3. **Threshold too high** - Try lowering the threshold in code (default is 0.7)

### Detection is slow

The detector is optimized to only check a region of interest (top 20-50% of screen) where results typically appear. If it's still slow, you can:
1. Reduce template sizes
2. Skip detection frames (e.g., check every 5 frames instead of every frame)

### Multiple variants needed

Different trophy levels or game modes may have slightly different result screens. To handle this:

```bash
# Create multiple variants:
python3 scripts/create_result_templates.py --type victory  # Creates victory.png
# Manually rename to victory_1.png
# Repeat and save as victory_2.png, etc.
```

The detector will try all variants and use the best match.

## Implementation Details

### Template Matching Algorithm

1. Extract ROI (region of interest) from screenshot:
   - Top 20-50% of screen
   - Center 60% horizontally

2. For each template:
   - Resize template to fit ROI if needed
   - Run OpenCV template matching (TM_CCOEFF_NORMED)
   - Track best match score

3. If best score > threshold:
   - Return result type ('victory', 'defeat', 'draw')
   - Otherwise return None

### File Structure

```
detection/
  battle_result_detector.py      # Main detector class
  result_templates/               # Template images
    victory.png                   # Victory screen template
    defeat.png                    # Defeat screen template
    draw.png                      # Draw screen template

scripts/
  create_result_templates.py     # Helper to create templates
  test_result_detection.py       # Test detection

main.py                          # Integration with main bot
```

## Next Steps

Once battle result detection is working:

1. **Implement RL reward function** - Use battle results for learning
2. **Track win rate** - Log results to analyze bot performance
3. **Add tower damage tracking** - For more granular rewards (optional)

See `rl/state_encoder.py` for the state encoding implementation.
