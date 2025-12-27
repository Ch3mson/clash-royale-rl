# How to Add New Cards to CardHandDetector

## Quick Summary

**Detection Rate: 86.7%** (52/60 cards detected on test screenshots)

The CardHandDetector now automatically loads all template variants for each card.

---

## Step 1: Add Card Templates

Add your card template images to `detection/card_templates/`:

### Naming Convention:
- **Single variant**: `{card_name}.png` (e.g., `giant.png`)
- **Multiple variants**: `{card_name}_1.png`, `{card_name}_2.png`, etc.
- **Grayed out**: `{card_name}_grayed.png` or `{card_name}_grayed_1.png`

### Examples:
```
detection/card_templates/
  ├── cannon_1.png          # Variant 1
  ├── cannon_2.png          # Variant 2
  ├── cannon_3.png          # Variant 3
  ├── cannon_grayed_1.png   # Grayed out variant
  ├── giant.png             # Single template
  └── wizard_1.png          # etc.
```

---

## Step 2: Update card_types Dictionary

Open `detection/card_hand_detector.py` and add your card to the `card_types` dictionary (around **line 25**):

```python
self.card_types = {
    # Original cards
    'tombstone': 'building',
    'bomber': 'ranged',
    'valkyrie': 'melee',
    'goblins': 'melee',
    'spear_goblins': 'ranged',
    'cannon': 'building',
    'giant': 'tank',
    'skeletons': 'melee',
    # New cards - add your cards here
    'battle_ram': 'tank',
    'bomb_tower': 'building',
    'fire_spirit': 'air',
    'furnace': 'building',
    'goblin_cage': 'building',
    'skeleton_dragons': 'air',
    'wizard': 'ranged',
    # ADD YOUR NEW CARD HERE:
    'your_card_name': 'card_type',  # <-- Add this line
}
```

### Card Types:
- `'melee'` - Close-range troops (knight, valkyrie, goblins)
- `'ranged'` - Ranged troops (archers, musketeer, bomber)
- `'tank'` - High HP troops (giant, battle_ram)
- `'air'` - Flying troops (minions, skeleton_dragons, fire_spirit)
- `'building'` - Structures (cannon, tombstone, furnace)

---

## Step 3: Test Detection

Run the evaluation script to check detection accuracy:

```bash
python3 evaluate_hand_detector.py
```

This will show:
- Which cards are in the dictionary
- Which templates are loaded
- Detection success rate on training screenshots
- Confidence scores for each card

---

## How to Get Card Templates

### Method 1: Use Existing Training Data
If you ran the agent with `--screenshots`, check:
```
training_data/session_<timestamp>/unclassified_cards/
```

These are auto-saved low-confidence card crops you can manually label.

### Method 2: Extract from Screenshot
1. Take a screenshot with the card in your hand
2. Use `CardHandDetector.save_card_crops(screenshot)` to extract all 4 card slots
3. Manually crop and save the card you want

### Method 3: Run Agent and Collect
```bash
python3 main.py --screenshots --games 5
```

Check `training_data/session_<timestamp>/unclassified_cards/` for auto-saved card crops.

---

## Current Detection Performance

**Results from evaluate_hand_detector.py:**

| Card | Avg Confidence | Min | Max | Samples |
|------|----------------|-----|-----|---------|
| battle_ram | 0.964 | 0.924 | 1.000 | 7 |
| bomb_tower | 0.944 | 0.810 | 1.000 | 17 |
| cannon | 0.992 | 0.984 | 1.000 | 2 |
| fire_spirit | 0.989 | 0.983 | 1.000 | 3 |
| furnace | 0.882 | 0.833 | 1.000 | 4 |
| skeleton_dragons | 0.968 | 0.877 | 1.000 | 6 |
| valkyrie | 0.907 | 0.806 | 1.000 | 8 |
| wizard | 0.944 | 0.753 | 1.000 | 5 |

**Overall: 86.7% detection rate** (52/60 cards detected)

---

## Troubleshooting

### Card Not Detected
1. Check template naming matches dictionary key exactly
2. Try adding more template variants (different levels/skins)
3. Lower threshold in `identify_card()` from 0.7 to 0.6

### Low Confidence (<0.8)
1. Add more template variants for that card
2. Ensure templates are cropped tightly around the card
3. Consider training a YOLOv8 classifier instead

### Wrong Card Detected
1. Cards look too similar - add more specific variants
2. Check if templates are correctly named
3. YOLOv8 classifier would handle this better

---

## Next Steps: Train YOLOv8 Classifier

If template matching accuracy isn't sufficient:

1. Collect training data with `--screenshots`
2. Manually label cards in `unclassified_cards/`
3. Train YOLOv8 image classifier on Google Colab
4. Replace template matching with YOLO inference

**Performance impact:** +10-15ms per frame (~15% slower, still very fast)

**Accuracy gain:** Much better, especially with similar-looking cards
