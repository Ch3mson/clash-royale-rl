# Clash Royale RL Bot - Project Plan

## Overview
Build a reinforcement learning bot for Clash Royale that learns from gameplay, uses Roboflow CV models for game state detection, and trains via self-play with a DQN/PPO agent. Uses hybrid manual+bot play to reduce ban risk.

## Hardware Setup
- **GPU**: RTX 3060 Ti (8GB VRAM)
- **CPU**: Ryzen 5600X (6 cores/12 threads)
- **Realistic capacity**: 3-4 parallel BlueStacks instances max
- **No proxies needed** (using hybrid manual+bot play strategy)

## Project Structure

```
clashroyalerl/
├── scripts/
│   ├── 1_setup_instances.py          # Create BlueStacks instances
│   ├── 2_calibrate_grid.py           # Calibrate arena boundaries
│   ├── 3_collect_cv_data.py          # Screenshot collection for Roboflow
│   ├── 4_record_manual_play.py       # Record your gameplay with actions
│   ├── 5_pretrain_bc.py              # Behavioral cloning (imitation learning)
│   └── 6_train_rl_multi.py           # Main RL training loop
│
├── src/
│   ├── environment/
│   │   ├── clash_env.py              # Gymnasium environment wrapper
│   │   ├── game_controller.py        # PyAutoGUI card placement & control
│   │   └── instance_manager.py       # BlueStacks instance management
│   │
│   ├── cv/
│   │   ├── roboflow_detector.py      # Roboflow inference integration
│   │   ├── state_extractor.py        # Convert detections to GameState
│   │   └── grid_system.py            # Pixel ↔ Grid conversion
│   │
│   ├── agent/
│   │   ├── dqn.py                    # DQN neural network
│   │   ├── replay_buffer.py          # Experience replay buffer
│   │   └── worker.py                 # Worker process for parallel training
│   │
│   └── utils/
│       ├── rewards.py                # Reward function definitions
│       ├── humanization.py           # Anti-detection (timing, randomization)
│       └── config.py                 # Configuration management
│
├── config/
│   ├── config.yaml                   # Hyperparameters & settings
│   └── arena_calibration_*.json      # Per-instance grid calibration
│
├── data/
│   ├── screenshots/                  # Raw game captures for Roboflow
│   ├── manual_games/                 # Recorded manual gameplay
│   └── models/                       # Saved RL checkpoints
│
├── .env                              # Roboflow API keys
├── requirements.txt
└── README.md
```

---

## Phase 1: Environment Setup (Week 1)

### ✅ Todo: Development Environment
- [ ] Install Python 3.11+, PyTorch, requirements
- [ ] Install BlueStacks 5 (Pie 64-bit)
- [ ] Create Roboflow account and workspace
- [ ] Set up Docker for Roboflow inference server
- [ ] Configure .env with Roboflow API key
- [ ] Initialize git repository

### ✅ Todo: Multi-Instance Setup
**Script**: `scripts/1_setup_instances.py`
- [ ] Create 3 BlueStacks instances programmatically
- [ ] Name them: ClashRL_0, ClashRL_1, ClashRL_2
- [ ] Set consistent resolution (1280x720) for all instances
- [ ] Configure Android 9 (Pie 64-bit) for each
- [ ] Manually install Clash Royale on each instance
- [ ] Complete tutorial and create throwaway Supercell accounts
- [ ] Test launching all instances simultaneously

**What this script does**: Automates creation of multiple BlueStacks instances, ensures consistent configuration

---

## Phase 2: Computer Vision Foundation (Week 1-2)

### ✅ Todo: Data Collection for Roboflow
**Script**: `scripts/3_collect_cv_data.py`
- [ ] Play 20-30 games manually while recording
- [ ] Capture screenshots every 0.5s (~50-100 per game)
- [ ] Focus on diverse scenarios: different elixir counts, troop positions
- [ ] Organize screenshots by game phase (early/mid/late/overtime/victory/defeat)
- [ ] Aim for 300-500 total screenshots

**What this script does**: Automates screenshot capture during manual gameplay for CV training data

### ✅ Todo: Roboflow CV Model Training
- [ ] Upload screenshots to Roboflow workspace
- [ ] Create 4 separate detection projects:
  - **Card Detection**: Annotate 4 cards in hand (300+ images)
  - **Troop Detection**: Annotate battlefield troops (300+ images)
  - **Elixir Counter**: OCR or digit recognition for elixir bar
  - **Game State**: Detect win/loss screens, "Play Again" button
- [ ] Use Roboflow augmentation to 3x dataset size
- [ ] Train models (YOLOv8 format)
- [ ] Deploy to local inference server
- [ ] Test detection accuracy (aim for >85% confidence)

**What you need**: Only annotate YOUR 8 deck cards initially (not all 109 cards)

### ✅ Todo: Recommended Simple Deck
Use **Giant Beatdown** for fastest learning:
- Giant (win condition)
- Musketeer (support)
- Mini P.E.K.K.A (defense)
- Valkyrie (splash)
- Fireball (spell)
- Arrows (spell)
- Ice Spirit (cycle)
- Cannon (building)

**Why**: Simple strategy (Giant at bridge when elixir full), easier for RL to learn

### ✅ Todo: Grid System Calibration
**Script**: `scripts/2_calibrate_grid.py`
- [ ] Run calibration for each instance (ClashRL_0, _1, _2)
- [ ] Interactive tool: click 4 arena corners
- [ ] Calculate pixel boundaries automatically
- [ ] Define strategic grid (start with 6x5 = 30 positions)
- [ ] Save calibration to JSON per instance
- [ ] Visualize grid overlay for verification

**What this script does**: Converts pixel coordinates to discrete grid positions for RL action space

---

## Phase 3: State Representation & Environment (Week 2)

### ✅ Todo: CV Pipeline Integration
**File**: `src/cv/roboflow_detector.py`
- [ ] Create CVProcessor class wrapping Roboflow inference
- [ ] Implement frame-by-frame detection with caching
- [ ] Handle detection failures gracefully (use previous state if confidence < 0.6)
- [ ] Performance target: <100ms per frame on GPU

**File**: `src/cv/state_extractor.py`
- [ ] Define GameState dataclass (elixir, cards, towers, troops)
- [ ] Convert Roboflow detections to structured state
- [ ] Implement grid-based troop position tracking
- [ ] Add state validation and error handling

**File**: `src/cv/grid_system.py`
- [ ] Implement ArenaConfig dataclass
- [ ] Create GridSystem class with pixel_to_grid() and grid_to_pixel()
- [ ] Define ActionSpace with valid placement cells (30-40 strategic positions)
- [ ] Action masking (can't play cards without elixir)

### ✅ Todo: Gymnasium Environment
**File**: `src/environment/clash_env.py`
- [ ] Create ClashRoyaleEnv extending gymnasium.Env
- [ ] Define observation_space (Box with state vector)
- [ ] Define action_space (Discrete with 120-150 actions)
- [ ] Implement reset() - click "Play Again", wait for game start
- [ ] Implement step(action) - execute action, capture state, calculate reward
- [ ] Handle game end detection (win/loss screens)

**File**: `src/environment/game_controller.py`
- [ ] Implement card dragging with PyAutoGUI
- [ ] Add "Play Again" button detection and clicking
- [ ] Implement retry logic for failed clicks
- [ ] Add safety checks (detect disconnects, crashes)
- [ ] Fix "Play Again" bug from CRBot (detect both victory and defeat screens)

---

## Phase 4: Reward Function & Anti-Detection (Week 2)

### ✅ Todo: Reward Function
**File**: `src/utils/rewards.py`

**Version 1 (Sparse - start here)**:
- [ ] +100 for game won
- [ ] -100 for game lost
- [ ] Nothing during gameplay

**Version 2 (Dense - add later)**:
- [ ] +(enemy_tower_damage - my_tower_damage) × 10
- [ ] -1 when elixir at max (encourage spending)
- [ ] +500 for win, +200 bonus for 3-crown
- [ ] Track and log which rewards fire most often

### ✅ Todo: Anti-Detection / Humanization
**File**: `src/utils/humanization.py`
- [ ] Random timing delays (±30% variance on all actions)
- [ ] Click position variance (±3-5 pixels)
- [ ] "Think pauses" (15% chance of 1-3s hesitation)
- [ ] Session management (play 2-4 hour sessions, then break)
- [ ] Random breaks (short: 2-8 min, medium: 15-45 min, long: 1-4 hrs)
- [ ] Bezier curve mouse movements (not straight lines)
- [ ] Intentional "mistakes" (2% chance of suboptimal play)

**Critical**: This is KEY to avoiding bans with hybrid manual+bot approach

---

## Phase 5: Imitation Learning (Week 2-3, Optional but Recommended)

### ✅ Todo: Record Manual Gameplay
**Script**: `scripts/4_record_manual_play.py`
- [ ] Play 50-100 games manually
- [ ] Record state + action pairs for each card placement
- [ ] Save as structured dataset (states, actions, rewards)
- [ ] Aim for diverse scenarios and strategies
- [ ] Store in data/manual_games/

**What this script does**: Captures your gameplay as supervised learning data

### ✅ Todo: Behavioral Cloning Pre-training
**Script**: `scripts/5_pretrain_bc.py`
- [ ] Load manual gameplay dataset
- [ ] Train DQN network to predict your actions from states
- [ ] Use supervised learning (cross-entropy loss)
- [ ] Train for 200-500 episodes
- [ ] Save pre-trained weights as initialization for RL

**Why do this**: Gives RL agent a "warm start", reduces learning time by 60%

---

## Phase 6: DQN Agent Implementation (Week 3)

### ✅ Todo: DQN Network
**File**: `src/agent/dqn.py`
- [ ] Implement DQN neural network (state_dim → 256 → 256 → 128 → action_dim)
- [ ] Add dropout layers (0.2) for regularization
- [ ] Consider Dueling DQN architecture (separate value/advantage streams)
- [ ] Implement forward pass for Q-value prediction

### ✅ Todo: Experience Replay
**File**: `src/agent/replay_buffer.py`
- [ ] Create ReplayBuffer class (capacity: 100,000 transitions)
- [ ] Implement add(), sample(), __len__()
- [ ] Use random sampling for training batches
- [ ] Handle edge cases (buffer not full yet)

### ✅ Todo: Central Learner
**File**: `src/agent/dqn.py` (CentralLearner class)
- [ ] Implement Double DQN with target network
- [ ] Training hyperparameters:
  - Batch size: 64
  - Gamma (discount): 0.99
  - Learning rate: 3e-4 (Adam optimizer)
  - Epsilon decay: 1.0 → 0.05 over 500 episodes
  - Target network update: every 100 steps
- [ ] Implement train_step() method
- [ ] Add checkpoint saving/loading
- [ ] Track metrics (loss, Q-values, epsilon)

---

## Phase 7: Parallel Training Setup (Week 3-4)

### ✅ Todo: Worker Process
**File**: `src/agent/worker.py`
- [ ] Create ClashRoyaleWorker class
- [ ] Each worker controls ONE BlueStacks instance
- [ ] Implement collect_episodes() - play games, store transitions
- [ ] Implement select_action() with epsilon-greedy
- [ ] Add weight syncing from central learner
- [ ] Handle instance crashes gracefully

### ✅ Todo: Multi-Instance Training Orchestration
**Script**: `scripts/6_train_rl_multi.py` (MAIN TRAINING SCRIPT)
- [ ] Create MultiInstanceTrainer class
- [ ] Initialize 3 workers (one per BlueStacks instance)
- [ ] Initialize central learner
- [ ] Implement main training loop:
  1. All workers collect episodes in parallel (multiprocessing)
  2. Aggregate experiences from all workers
  3. Add to central replay buffer
  4. Train network on batches
  5. Sync updated weights to all workers
  6. Repeat
- [ ] Add TensorBoard logging
- [ ] Save checkpoints every 50 episodes
- [ ] Implement resume from checkpoint

**What this script does**: Orchestrates all 3 instances, coordinates parallel data collection and centralized learning

---

## Phase 8: Initial Training (Week 4)

### ✅ Todo: Baseline Performance
- [ ] Test random agent (should lose 100%)
- [ ] Test rule-based agent (place cards when elixir full)
- [ ] Record YOUR manual win rate for comparison

### ✅ Todo: Training Protocol

**Stage 1: Sparse Rewards (Episodes 0-200)**
- [ ] Train with win/loss rewards only
- [ ] Monitor if agent learns to:
  - Play cards when it has elixir
  - Not waste elixir at max
  - Basic defensive plays
- [ ] Target: Agent stops making illegal moves

**Stage 2: Dense Rewards (Episodes 200-500)**
- [ ] Add tower damage rewards
- [ ] Add elixir efficiency penalties
- [ ] Monitor win rate increase
- [ ] Target: 30-40% win rate at low trophies

**Stage 3: Curriculum Learning (Episodes 500+)**
- [ ] Start at low trophy range (1000-2000)
- [ ] Gradually increase difficulty
- [ ] Continue training until 40%+ win rate

### ✅ Todo: Monitoring
- [ ] Use TensorBoard for real-time metrics:
  - Episode reward
  - Win rate (rolling average over 50 games)
  - Average Q-values
  - Loss curves
  - Epsilon decay
- [ ] Log game replays for debugging
- [ ] Save best-performing checkpoint

---

## Phase 9: Hybrid Manual+Bot Play Schedule (Ongoing)

### ✅ Todo: Daily Routine (Per Account)
**This is your anti-ban strategy**

**Weekdays**:
- Morning (7-8am): YOU play manually (2-3 games)
- Afternoon (2-5pm): BOT plays (20-30 games while you're busy)
- Evening (8-9pm): YOU play manually (2-3 games)

**Weekends**:
- Morning: Manual play session (10-15 games)
- Afternoon: Bot training (30-40 games)
- Evening: Manual play (5-10 games)

**Expected**: ~150 bot games/week + ~50 manual games/week per account

### ✅ Todo: Session Manager Implementation
- [ ] Implement scheduler in humanization.py
- [ ] Auto-pause bot during manual play times
- [ ] Add random breaks between sessions
- [ ] Never play 24/7 (simulate human sleep 11pm-7am)
- [ ] Stagger bot sessions across instances (Instance 1: 2-5pm, Instance 2: 3-6pm, Instance 3: 4-7pm)

**Why this works**: Accounts look legitimate, no proxies needed, much lower ban risk

---

## Phase 10: Advanced Improvements (Week 5+)

### ✅ Todo: Upgrade to PPO (Optional)
- [ ] If DQN plateaus, implement PPO using Stable-Baselines3
- [ ] PPO is more sample-efficient for continuous improvements
- [ ] Use Ray RLlib for easier distributed PPO

### ✅ Todo: Advanced Features
- [ ] Elixir prediction: Estimate enemy elixir from troop placements
- [ ] Card counting: Track which cards enemy has used
- [ ] Combo recognition: Learn synergies (Giant + Musketeer)
- [ ] Multi-deck training: Train agent on multiple decks

### ✅ Todo: Multiple Deck Support
- [ ] Once Giant Beatdown works well (40%+ win rate), expand
- [ ] Train second model on Hog Cycle deck
- [ ] Train third model on your main deck
- [ ] Compare performance across deck archetypes

---

## Success Metrics

### Week 2
- [ ] Environment runs, random actions execute correctly
- [ ] CV models detect cards/troops with >85% accuracy
- [ ] Grid system correctly converts positions

### Week 3
- [ ] DQN training starts without errors
- [ ] Agent learns to play valid moves (no illegal actions)
- [ ] Replay buffer fills with diverse experiences

### Week 4
- [ ] Win rate > 20% at trophy 1000-1500
- [ ] Agent demonstrates basic strategy (e.g., places Giant, supports it)

### Week 6
- [ ] Win rate > 40% at trophy 2000-3000
- [ ] Agent handles most game situations

### Week 8+
- [ ] Competitive play at trophy 4000+
- [ ] Optimize for trophy climbing
- [ ] Consider advanced RL algorithms (PPO, A3C)

---

## Key Learning Resources

### Computer Vision
- Roboflow Documentation: https://docs.roboflow.com/
- YOLOv8 Guide: https://docs.ultralytics.com/
- Roboflow YouTube tutorials for annotation and training

### Reinforcement Learning
- **Spinning Up in Deep RL** (OpenAI): https://spinningup.openai.com/
  - Best RL introduction, covers DQN, PPO, policy gradients
  - Read DQN section (Chapters 3-4) first
- **Sutton & Barto - RL Book** (free): http://incompleteideas.net/book/the-book-2nd.html
  - The RL bible, chapters 6-7 on temporal difference learning
- **Stable-Baselines3 Docs**: https://stable-baselines3.readthedocs.io/
  - Production-ready RL algorithms

### Game AI
- DeepMind AlphaStar (StarCraft II): Similar real-time strategy challenges
- OpenAI Five (Dota 2): Multi-agent RL in real-time games
- PPO Paper: https://arxiv.org/abs/1707.06347

### Implementation
- PyTorch RL Tutorial: DQN implementation guide
- Gymnasium Docs: Custom environment creation
- Ray RLlib: Distributed RL training

---

## Risk Mitigation

### Account Bans
- **Strategy**: Use throwaway Supercell accounts (NOT your main account)
- **Expected lifetime**: 1-4 weeks per account with hybrid play
- **Solution**: Keep creating new accounts, treat bans as "cost of doing business"
- **Protection**: NEVER bot on same network as your main account

### Training Time
- **With 3 instances**: 3x faster data collection
- **Expected**: 400 games/day (vs 40 with single instance)
- **Timeline**: Reach competitive play in ~3 weeks instead of 8

### CV Detection Failures
- **Strategy**: Log all detection errors, retrain models iteratively
- **Fallback**: Use previous frame's state if current detection fails
- **Monitoring**: Track detection confidence scores

### Reward Hacking
- **Watch for**: Agent learning to exploit reward function (e.g., learns to draw games)
- **Solution**: Continuously monitor gameplay, adjust reward function
- **Testing**: Manually review bot games every 100 episodes

### Hardware Limitations
- **3 instances max** on RTX 3060 Ti + Ryzen 5600X
- **Monitor**: GPU/CPU usage, adjust if system becomes unstable
- **Fallback**: Reduce to 2 instances if performance degrades

---

## Testing Checklist

### Per-Script Testing
```bash
# Test each script individually with --test flag

python scripts/1_setup_instances.py --test           # Creates 1 instance
python scripts/2_calibrate_grid.py --instance_id 0   # Calibrate instance 0
python scripts/3_collect_cv_data.py --test           # Collect 10 screenshots
python scripts/4_record_manual_play.py --test        # Record 5 games
python scripts/5_pretrain_bc.py --test              # Train 10 epochs
python scripts/6_train_rl_multi.py --test           # Train 10 episodes, 1 worker
```

### Integration Testing
- [ ] All 3 instances launch correctly
- [ ] CV detection runs on all instances simultaneously
- [ ] Workers collect experiences without crashes
- [ ] Central learner updates network correctly
- [ ] Weights sync to all workers
- [ ] Checkpoints save and load properly

---

## Final Notes

### What Makes This Project Unique
1. **Hybrid manual+bot play** - Much safer than pure botting
2. **Multi-instance parallel training** - 3x faster learning
3. **Grid-based action space** - Simpler than pixel-based approaches
4. **Roboflow CV** - No need to train detection models from scratch
5. **Anti-detection from day 1** - Humanization built into the system

### Expected Timeline
- **Week 1-2**: Setup, CV training, environment implementation
- **Week 3**: DQN agent implementation and initial training
- **Week 4**: First successful bot gameplay, basic strategy emerges
- **Week 5-6**: Performance optimization, win rate improvement
- **Week 7-8**: Competitive play, advanced features

### When to Pivot
- If DQN isn't learning after 500 episodes → Check reward function
- If CV detection is unreliable → Collect more training data
- If instances keep crashing → Reduce to 2 workers
- If accounts getting banned quickly → Increase manual play ratio

Good luck! 🎮🤖

