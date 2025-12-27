# RL Training Guide

Complete guide for training the Clash Royale RL agent locally on your Mac.

## Quick Start

```bash
# Start RL training (trains while playing)
python3 main.py --rl --games 100
```

That's it! The agent will:
- Play games in BlueStacks
- Learn from experience in real-time
- Save checkpoints automatically every 5 games
- Improve over time

## What Happens During Training

1. **Game starts** → Agent observes state (hand, elixir, troops, tower HP)
2. **Agent selects action** → Initially random (exploration), gradually learns (exploitation)
3. **Action executed** → Card played at position
4. **Reward calculated** → Based on tower damage, win/loss, etc.
5. **Agent learns** → Neural network trains on experience
6. **Repeat** → Next action, continuously improving

## Training Progress

You'll see output like this:

```
[RL PLAY] knight at row 24, col 8 (epsilon: 0.850)

[RL STATS] Steps: 100, Epsilon: 0.845, Avg Reward: 12.45, Avg Loss: 0.0234

[REWARD] Destroyed enemy_left_princess! Bonus: +10.0

[RL EPISODE END]
  Episodes: 5
  Avg Reward (100): 15.30
  Epsilon: 0.820
  Buffer: 432/10000

Checkpoint saved: checkpoints/latest.pt
```

### What the Stats Mean

- **Epsilon**: Exploration rate (1.0 = random, 0.1 = mostly learned policy)
- **Avg Reward**: Higher is better (positive = winning, negative = losing)
- **Avg Loss**: Training loss (should decrease over time)
- **Buffer**: Number of experiences stored for learning

## Command Line Options

```bash
# Basic training
python3 main.py --rl --games 100

# Training with specific BlueStacks instance
python3 main.py --rl --games 50 --instance 0

# Train + save screenshots for debugging
python3 main.py --rl --games 20 --screenshots

# Long training session (overnight)
python3 main.py --rl --games 500
```

## Checkpoints

Checkpoints are saved automatically:

- **Every 5 games**: `checkpoints/checkpoint_game_5.pt`, `checkpoint_game_10.pt`, etc.
- **Latest**: `checkpoints/latest.pt` (always the most recent)

### Resume Training

Training automatically resumes from `checkpoints/latest.pt` if it exists.

To start fresh:
```bash
rm -rf checkpoints/
python3 main.py --rl --games 100
```

### Use Best Checkpoint

```bash
# Copy your best checkpoint to latest
cp checkpoints/checkpoint_game_50.pt checkpoints/latest.pt

# Continue training from that checkpoint
python3 main.py --rl --games 100
```

## Training Strategy

### Phase 1: Exploration (Games 1-50)
- **Epsilon**: 1.0 → 0.6
- **Behavior**: Mostly random actions
- **Goal**: Fill replay buffer with diverse experiences
- **Expect**: Losses, chaotic play, learning fundamentals

### Phase 2: Learning (Games 51-200)
- **Epsilon**: 0.6 → 0.2
- **Behavior**: Mix of exploration and learned policy
- **Goal**: Learn basic strategies
- **Expect**: Gradual improvement, occasional wins

### Phase 3: Exploitation (Games 201+)
- **Epsilon**: 0.2 → 0.1
- **Behavior**: Mostly uses learned policy
- **Goal**: Refine strategy, optimize play
- **Expect**: Consistent performance, strategic play

## Reward System

The agent learns from:

### Tower Damage (+/-)
- **+0.01 per HP** damage to enemy towers
- **-0.01 per HP** damage to ally towers
- **+10.0 bonus** for destroying enemy tower
- **-10.0 penalty** for losing ally tower

### Battle Results
- **+100.0** for victory
- **-100.0** for defeat
- **0.0** for draw

### Small Bonuses
- **+0.05** per ally troop on field (encourages aggression)
- **+0.1** for maintaining good elixir (5-8 range)

## Performance

### Hardware
- **Mac M1/M2/M3**: Fast training (~10ms per step)
- **Intel Mac**: Moderate training (~30ms per step)
- Both work fine - game speed is the bottleneck, not training

### Training Speed
- **~3-5 minutes** per game (depends on game length)
- **~6-8 hours** for 100 games
- Can pause anytime - checkpoints are saved

### Memory Usage
- **~500MB RAM** for agent
- **~2GB RAM** for YOLO models
- **~10k experiences** in replay buffer

## Monitoring Training

### Watch for Good Signs
- Epsilon decreasing (0.1-0.3 is good)
- Average reward increasing (positive is good)
- Win rate improving
- Agent making strategic plays (targeting enemy towers, defending)

### Watch for Bad Signs
- Average reward stuck at -100 (always losing)
- Loss not decreasing after 50+ games
- Agent playing randomly even at low epsilon

### Solutions
- **Stuck at -100**: Let it train longer (100+ games), might need better card detection
- **High loss**: Normal early on, should stabilize after 50 games
- **Random play**: Check that `use_model=True` (YOLO is working)

## Tips for Better Training

### 1. Good Card Detection
RL relies on YOLO detecting troops correctly. If detection is poor:
```bash
# Test detection first
python3 main.py --model --games 1

# Check if troops are being detected
# If not, collect more training data for YOLO
```

### 2. Longer Sessions
Agent needs lots of games to learn:
- **Minimum**: 100 games
- **Recommended**: 200-500 games
- **Optimal**: 1000+ games

### 3. Consistent Deck
Keep using the same deck during training for best results.

### 4. Let It Lose
Early games will be terrible - that's normal! Agent learns from failures.

## Troubleshooting

### "RuntimeError: MPS backend out of memory"
Apple Silicon MPS ran out of memory. Switch to CPU:
```python
# In main.py, line ~88
device='cpu'  # Force CPU instead of MPS
```

### "Checkpoint saved: checkpoints/latest.pt" not appearing
Check permissions:
```bash
mkdir -p checkpoints
chmod 755 checkpoints
```

### Agent plays same action repeatedly
Check `_get_valid_actions()` - might not have enough elixir or cards on cooldown.

### Training is very slow
Normal! Each game takes 3-5 minutes. This is real-time learning.

## Advanced: Hyperparameter Tuning

Edit values in `main.py` (lines 88-107):

```python
# Learning rate (how fast it learns)
learning_rate=0.0001  # Decrease if training is unstable

# Discount factor (how much it values future rewards)
gamma=0.99  # Lower = more short-term focused

# Exploration decay
epsilon_decay=0.995  # Higher = explores longer

# Batch size
batch_size=64  # Larger = more stable but slower

# Buffer size
buffer_size=10000  # Larger = more diverse experiences
```

## Next Steps

After training:

1. **Evaluate performance**:
   ```bash
   # Test trained agent
   python3 main.py --rl --games 10
   # Watch win rate
   ```

2. **Save best model**:
   ```bash
   cp checkpoints/latest.pt checkpoints/best_model.pt
   ```

3. **Continue training**:
   ```bash
   python3 main.py --rl --games 100
   # Automatically resumes from latest.pt
   ```

4. **Experiment with rewards**:
   Edit `rl/reward_calculator.py` to adjust reward values

## FAQ

**Q: How long until it's good?**
A: 100-200 games minimum. Expect noticeable improvement after 50 games.

**Q: Can I pause training?**
A: Yes! Just stop (Ctrl+C). Restart with same command - it resumes from latest.pt.

**Q: Does it work without YOLO?**
A: No - RL needs troop detection to see enemy threats and make decisions.

**Q: Will it beat human players?**
A: Eventually! But needs 500+ games and good card detection.

**Q: Can I train on Colab?**
A: No - needs live BlueStacks connection. Must train locally.

**Q: What if tower HP detection fails?**
A: Agent uses cached value (last successful reading). Run `python3 scripts/test_tower_hp.py` to verify detection works.

## Files Created

- **rl/dqn_agent.py** - DQN neural network and training logic
- **rl/reward_calculator.py** - Reward calculation with tower HP
- **rl/state_encoder.py** - State encoding (already updated)
- **main.py** - Updated with RL integration
- **checkpoints/** - Saved models (auto-created)

Everything is ready to go! Just run:

```bash
python3 main.py --rl --games 100
```

Happy training! 🚀
