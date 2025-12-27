# Card Info Database Guide

## Overview

All card information is now centralized in [detection/card_info.py](detection/card_info.py). This makes it easier to:
- Add new cards with detailed stats
- Share card data across multiple components
- Add future features like elixir management, counter-play logic, etc.

---

## How to Add New Cards

### Option 1: Edit card_info.py Directly

Open [detection/card_info.py](detection/card_info.py) and add your card to the `CARD_INFO` dictionary:

```python
'your_card_name': {
    'category': 'ranged',              # melee, ranged, tank, air, building, spell
    'elixir_cost': 4,                  # 1-10
    'targets': ['ground', 'air'],      # LIST: can have multiple values!
    'range': 'long',                   # melee, medium, long, very_long, spell
    'speed': 'medium',                 # slow, medium, fast, very_fast, None (for buildings)
},
```

### Option 2: Copy From Existing Card

Find a similar card and copy its structure:

```python
# Example: Adding Ice Spirit (similar to Fire Spirit)
'ice_spirit': {
    'category': 'ranged',
    'elixir_cost': 1,
    'targets': ['ground', 'air'],
    'range': 'melee',
    'speed': 'very_fast'
},
```

---

## Multiple Values in Keys

Yes! You can store **multiple values as a list**:

```python
# Single target
'targets': ['ground']          # Can only hit ground troops

# Multiple targets
'targets': ['ground', 'air']   # Can hit both ground and air

# No targets (spawner buildings)
'targets': []                  # Doesn't attack (e.g., Tombstone, Furnace)
```

**Other examples:**
```python
# Future features you might add:
'keywords': ['splash', 'spawner']           # Multiple keywords
'counters': ['swarm', 'tank']               # Good against these types
'weak_to': ['spell', 'air']                 # Weak to these types
```

---

## Using the Card Info

### Get Card Category (for Threat Calculation)

```python
from detection.card_info import get_card_category

category = get_card_category('wizard')  # Returns: 'ranged'
```

### Get Elixir Cost

```python
from detection.card_info import get_card_elixir

cost = get_card_elixir('giant')  # Returns: 5
```

### Check if Card Can Target Air

```python
from detection.card_info import can_target_air

can_hit_air = can_target_air('wizard')   # Returns: True
can_hit_air = can_target_air('knight')   # Returns: False
```

### Access Full Card Info

```python
from detection.card_info import CARD_INFO

wizard_info = CARD_INFO['wizard']
# Returns: {'category': 'ranged', 'elixir_cost': 5, 'targets': ['ground', 'air'], ...}

# Access specific fields
elixir = wizard_info['elixir_cost']      # 5
targets = wizard_info['targets']          # ['ground', 'air']
can_target_air = 'air' in targets         # True
```

---

## Current Card Database

**Total Cards: 39**

### By Category:

- **Melee**: 7 cards (Knight, Mini PEKKA, Valkyrie, Goblins, etc.)
- **Ranged**: 10 cards (Archers, Bomber, Musketeer, Wizard, etc.)
- **Tank**: 4 cards (Giant, Prince, Giant Skeleton, PEKKA, Battle Ram)
- **Air**: 4 cards (Baby Dragon, Minions, Minion Horde, Skeleton Dragons)
- **Building**: 8 cards (Cannon, Tombstone, Bomb Tower, X-Bow, etc.)
- **Spell**: 6 cards (Arrows, Fireball, Lightning, Rocket, Rage)

---

## Future Features You Can Add

With the new detailed card info, you can easily implement:

### 1. **Elixir Management**
```python
# Don't play if we don't have enough elixir
card_cost = CARD_INFO[card_name]['elixir_cost']
if current_elixir < card_cost:
    return  # Can't play this card
```

### 2. **Smart Counter-Play**
```python
# Play air troops against ground-only enemies
if not can_target_air(enemy_card):
    play_card_with_category('air')
```

### 3. **Range-Based Placement**
```python
# Place long-range cards behind tanks
if CARD_INFO[card_name]['range'] == 'long':
    place_behind_tank()
```

### 4. **Speed-Based Strategy**
```python
# Fast troops for quick pushes
if CARD_INFO[card_name]['speed'] in ['fast', 'very_fast']:
    place_at_bridge()
```

### 5. **Card Filtering**
```python
# Get all air-targeting cards
air_cards = [card for card in CARD_INFO if can_target_air(card)]
```

---

## Backward Compatibility

The old `card_types` dictionary still works:

```python
from detection.card_info import CARD_TYPES

# Old way (still supported)
category = CARD_TYPES['wizard']  # Returns: 'ranged'
```

This ensures existing code doesn't break!

---

## File Locations

- **Card Database**: [detection/card_info.py](detection/card_info.py)
- **CardHandDetector**: [detection/card_hand_detector.py](detection/card_hand_detector.py)
- **CardDetector (YOLO)**: [detection/card_detector.py](detection/card_detector.py)

Both detectors now import from the centralized `card_info.py` database.
