import time
import random
import argparse
import os
from datetime import datetime
import cv2
from controllers.game_controller import GameController
from detection.state_detector import StateDetector, GameState
from detection.card_detector import CardDetector
from detection.card_hand_detector import CardHandDetector
from detection.elixir_detector import ElixirDetector
from detection.battle_result_detector import BattleResultDetector
from detection.tower_hp_detector import TowerHPDetector
from rl.dqn_agent import DQNAgent
from rl.state_encoder import StateEncoder
from rl.reward_calculator import RewardCalculator

# Main agent who runs the game
class Agent:
    def __init__(self, instance_id=0, save_screenshots=False, use_model=False, use_rl=False):
        self.gc = GameController(instance_id)
        self.detector = StateDetector(self.gc.image_matcher)
        self.games_played = 0

        # Screenshot saving setup
        self.save_screenshots = save_screenshots
        self.screenshot_interval = 5.0  # Fixed at 5 seconds
        self.last_save_time = 0
        self.screenshot_count = 0
        self.output_dir = None

        # Card hand detection
        self.hand_detector = CardHandDetector()
        self.current_hand = []

        # Elixir detection
        self.elixir_detector = ElixirDetector()
        self.current_elixir = None

        # Battle result detection
        self.result_detector = BattleResultDetector()
        self.battle_result = None  # 'victory', 'defeat', 'draw', or None

        # Battle state
        self.last_play_time = 0
        self.play_interval = random.uniform(2.0, 3.0)  # 2-3 seconds between plays

        # YOLO model setup
        self.use_model = use_model
        self.card_detector = None
        if self.use_model:
            print("Loading YOLO models...")
            self.card_detector = CardDetector(
                model_path="models/best.pt",  # Detection model (WHERE + ally/enemy)
                classifier_path="models/card_classifier.pt",  # Classification model (WHICH card)
                confidence_threshold=0.25,
                grid_system=self.gc.grid
            )
            print("Models loaded successfully!")

        # RL training setup
        self.use_rl = use_rl
        self.rl_agent = None
        self.state_encoder = None
        self.reward_calculator = None
        self.prev_state = None
        self.prev_action = None
        self.tower_hp_detector = None
        self.cached_tower_hp = None
        self.last_tower_hp_update = 0
        self.TOWER_HP_UPDATE_INTERVAL = 2.0  # Update every 2 seconds

        if self.use_rl:
            print("\n" + "="*60)
            print("Initializing RL Training System")
            print("="*60)

            # Tower HP detector
            self.tower_hp_detector = TowerHPDetector()

            # State encoder
            self.state_encoder = StateEncoder(grid_rows=32, grid_cols=18, max_cards=4)

            # Action space: 4 cards × 32 rows × 18 cols = 2304 actions
            self.action_size = 4 * 32 * 18

            # RL agent
            self.rl_agent = DQNAgent(
                state_size=self.state_encoder.state_size,
                action_size=self.action_size,
                learning_rate=0.0001,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.1,
                epsilon_decay=0.995,
                buffer_size=10000,
                batch_size=64
            )

            # Reward calculator
            self.reward_calculator = RewardCalculator(
                tower_damage_reward=0.01,
                tower_destroy_bonus=10.0,
                elixir_advantage_reward=0.1,
                win_reward=100.0,
                loss_penalty=-100.0
            )

            # Try to load checkpoint
            checkpoint_path = "checkpoints/latest.pt"
            if os.path.exists(checkpoint_path):
                self.rl_agent.load_checkpoint(checkpoint_path)
            else:
                print("No checkpoint found - starting fresh")

            print("="*60)
            print("RL Training Enabled!")
            print(f"State size: {self.state_encoder.state_size}")
            print(f"Action size: {self.action_size}")
            print("="*60 + "\n")

        if self.save_screenshots:
            self.output_dir = self._create_output_dir()

    def _create_output_dir(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"training_data/session_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectory for unclassified card crops
        self.unclassified_cards_dir = os.path.join(output_dir, "unclassified_cards")
        os.makedirs(self.unclassified_cards_dir, exist_ok=True)

        print(f"Saving screenshots to: {output_dir}")
        print(f"Unclassified cards will be saved to: {self.unclassified_cards_dir}")
        # Initialize last_save_time to allow immediate first screenshot
        self.last_save_time = 0
        self.card_save_count = 0
        return output_dir

    def _save_screenshot(self, screenshot, state):
        self.screenshot_count += 1
        filename = f"screenshot_{self.screenshot_count:04d}_{state.name}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, screenshot)
        # print(f"Saved: {filename}")
        # ^ use this only for debugging

    def _save_unclassified_cards(self, screenshot, hand, confidence_threshold=0.7):
        """
        Save card crops that couldn't be classified with high confidence for manual labeling

        Args:
            screenshot: Full game screenshot
            hand: List of card detection results from CardHandDetector
            confidence_threshold: Save cards below this confidence (default: 0.7)
        """
        for slot_idx, card_info in enumerate(hand):
            # Save if card couldn't be detected or confidence is low
            if card_info is None or card_info['confidence'] < confidence_threshold:
                # Extract card crop from screenshot
                x1, y1, x2, y2 = self.hand_detector.CARD_SLOTS[slot_idx]
                card_crop = screenshot[y1:y2, x1:x2]

                # Generate filename
                self.card_save_count += 1
                card_name = card_info['card_name'] if card_info else 'unknown'
                conf = card_info['confidence'] if card_info else 0.0
                filename = f"card_{self.card_save_count:04d}_slot{slot_idx}_{card_name}_conf{conf:.2f}.png"
                filepath = os.path.join(self.unclassified_cards_dir, filename)

                # Save crop
                cv2.imwrite(filepath, card_crop)
                print(f"[CARD] Saved low-confidence card: {filename}")

    def play_games(self, num_games=1):
        print(f"Agent is now playing {num_games} games...")

        while self.games_played < num_games:
            screenshot = self.gc.take_screenshot()

            if screenshot is None:
                continue

            state = self.detector.detect_state(screenshot, verbose=False)

            # Run YOLO detection if enabled and in battle
            if self.use_model and state == GameState.IN_BATTLE:
                # Debug: Save a test screenshot to check what model sees
                import os
                debug_dir = "debug_screenshots"
                os.makedirs(debug_dir, exist_ok=True)
                if not hasattr(self, '_debug_screenshot_saved'):
                    cv2.imwrite(f"{debug_dir}/test_detection.png", screenshot)
                    print(f"[DEBUG] Saved screenshot: {screenshot.shape}")
                    self._debug_screenshot_saved = True

                detections = self.card_detector.detect(screenshot, verbose=True)

            # Save screenshot if enabled and in battle
            if self.save_screenshots and state == GameState.IN_BATTLE:
                current_time = time.time()
                if current_time - self.last_save_time >= self.screenshot_interval:
                    self._save_screenshot(screenshot, state)

                    # Detect hand and save unclassified cards from this screenshot
                    hand = self.hand_detector.get_hand(screenshot, verbose=False)
                    self._save_unclassified_cards(screenshot, hand, confidence_threshold=0.7)

                    self.last_save_time = current_time

            if state == GameState.MAIN_MENU:
                self.handle_main_menu()
            elif state == GameState.IN_BATTLE:
                self.handle_battle(screenshot)
            elif state == GameState.BATTLE_END:
                self.handle_battle_end()
            elif state == GameState.QUEUEING:
                print("Queueing for battle...")
                time.sleep(2)
            else:
                # UNKNOWN or LOADING state
                time.sleep(1)
    
    def handle_main_menu(self):
        print("Handling main menu:")
        self.gc.click_battle_button()
        time.sleep(5)
    
    def handle_battle(self, screenshot):
        # Check for battle result first
        result = self.result_detector.detect_result(screenshot, threshold=0.8, verbose=False)
        if result is not None:
            self.battle_result = result
            print(f"\n{'='*60}")
            print(f"[BATTLE END] Result: {result.upper()}")
            print(f"{'='*60}\n")
            return

        # If RL mode is enabled, use RL agent
        if self.use_rl:
            self._handle_battle_rl(screenshot)
            return

        # If model is not enabled, just wait (let user play manually)
        if not self.use_model:
            time.sleep(0.5)  # Just monitor, don't play
            return

        # Check elixir on every iteration (as fast as possible)
        new_elixir = self.elixir_detector.get_elixir(screenshot, verbose=False)
        if new_elixir is not None:
            self.current_elixir = new_elixir

        # Detect cards in hand (fast, no delay)
        self.current_hand = self.hand_detector.get_hand(screenshot, verbose=False)

        # Get all detections if YOLO model is enabled (fast threat detection)
        all_detections = []
        enemy_detections = []
        if self.card_detector:
            all_detections = self.card_detector.detect(screenshot, verbose=False)

            # Separate enemy detections for placement logic
            enemy_detections = [d for d in all_detections if d['class_name'].startswith('enemy')]

            # Only place cards if enemies are detected
            if len(enemy_detections) == 0:
                return

        # Select best card to play based on situation
        card_slot, card_info = self._select_best_card(enemy_detections)

        if card_slot is None:
            # No playable cards - continue checking without delay
            return

        # Determine placement based on card type and enemy positions
        row, col = self._get_smart_placement(card_info, enemy_detections)

        # If no valid placement, don't play
        if row is None or col is None:
            return

        # Play the selected card immediately
        card_name = card_info['card_name']
        card_type = card_info['card_type']
        elixir_cost = card_info.get('elixir_cost', '?')
        print(f"[PLAY] {card_name} ({card_type}, {elixir_cost} elixir) at row {row}, col {col}")

        self.gc.play_card(card_slot, row, col)

        # Small delay after playing to prevent duplicate plays (card animation time)
        time.sleep(0.5)

    def _handle_battle_rl(self, screenshot):
        """
        RL-enabled battle handler with tower HP tracking and learning
        """
        # Detect all game state components
        current_time = time.time()

        # 1. Elixir (fast)
        elixir = self.elixir_detector.get_elixir(screenshot, verbose=False)
        if elixir is None:
            elixir = 5.0  # Default if detection fails

        # 2. Hand (fast)
        hand = self.hand_detector.get_hand(screenshot, verbose=False)

        # 3. Tower HP (throttled - OCR is slower)
        if current_time - self.last_tower_hp_update > self.TOWER_HP_UPDATE_INTERVAL:
            self.cached_tower_hp = self.tower_hp_detector.get_tower_hp(screenshot, verbose=False)
            self.last_tower_hp_update = current_time

        tower_hp = self.cached_tower_hp if self.cached_tower_hp is not None else {
            'enemy_left_princess': 0, 'enemy_king': 0, 'enemy_right_princess': 0,
            'ally_left_princess': 0, 'ally_king': 0, 'ally_right_princess': 0,
        }

        # 4. Troop detections (if YOLO enabled)
        all_detections = []
        enemy_detections = []
        ally_detections = []

        if self.card_detector:
            all_detections = self.card_detector.detect(screenshot, verbose=False)
            enemy_detections = [d for d in all_detections if d['class_name'].startswith('enemy')]
            ally_detections = [d for d in all_detections if d['class_name'].startswith('ally')]

        # 5. Encode current state
        current_state_vector = self.state_encoder.encode_state(
            elixir=elixir,
            hand=hand,
            enemy_detections=enemy_detections,
            ally_detections=ally_detections,
            tower_hp=tower_hp
        )

        # Build state dict for reward calculation
        current_state_dict = {
            'tower_hp': tower_hp,
            'elixir': elixir,
            'enemy_troops': enemy_detections,
            'ally_troops': ally_detections,
            'battle_result': self.battle_result
        }

        # 6. Calculate reward from previous step
        if self.prev_state is not None and self.prev_action is not None:
            reward = self.reward_calculator.calculate_step_reward(
                prev_state=self.prev_state,
                curr_state=current_state_dict,
                action_taken=self.prev_action
            )

            # Store transition in replay buffer
            done = self.battle_result is not None
            self.rl_agent.store_transition(
                state=self.prev_state_vector,
                action=self.prev_action,
                reward=reward,
                next_state=current_state_vector,
                done=done
            )

            # Train agent
            loss = self.rl_agent.train_step()

            # Print stats periodically
            if self.rl_agent.steps % 100 == 0:
                stats = self.rl_agent.get_stats()
                print(f"\n[RL STATS] Steps: {stats['steps']}, "
                      f"Epsilon: {stats['epsilon']:.3f}, "
                      f"Avg Reward: {stats['avg_reward_100']:.2f}, "
                      f"Avg Loss: {stats['avg_loss_100']:.4f}")

        # 7. Get valid actions
        valid_actions = self._get_valid_actions(hand, elixir)

        if len(valid_actions) == 0:
            # No valid actions - wait
            self.prev_state = None
            self.prev_action = None
            self.prev_state_vector = None
            time.sleep(0.1)
            return

        # 8. Select action using RL agent
        action = self.rl_agent.select_action(current_state_vector, valid_actions)

        # 9. Decode action to (card_slot, row, col)
        card_slot, row, col = self._decode_action(action)

        # 10. Execute action
        card_info = hand[card_slot] if card_slot < len(hand) else None
        if card_info:
            card_name = card_info.get('card_name', 'unknown')
            print(f"[RL PLAY] {card_name} at row {row}, col {col} (epsilon: {self.rl_agent.epsilon:.3f})")

        self.gc.play_card(card_slot, row, col)

        # 11. Save state for next step
        self.prev_state = current_state_dict
        self.prev_state_vector = current_state_vector
        self.prev_action = action

        # Small delay to prevent duplicate plays
        time.sleep(0.5)

    def _get_valid_actions(self, hand, elixir):
        """
        Get list of valid action indices based on available cards and elixir
        """
        from detection.card_info import get_card_elixir

        valid_actions = []

        for card_slot in range(4):
            if card_slot >= len(hand) or hand[card_slot] is None:
                continue

            card = hand[card_slot]
            card_name = card.get('card_name', 'unknown')
            available = card.get('available', True)

            if not available:
                continue

            elixir_cost = get_card_elixir(card_name)
            if elixir < elixir_cost:
                continue

            # Card is playable - add all positions as valid actions
            for row in range(16, 32):  # Only our territory
                for col in range(18):
                    action = self._encode_action(card_slot, row, col)
                    valid_actions.append(action)

        return valid_actions

    def _encode_action(self, card_slot, row, col):
        """Encode (card_slot, row, col) to single action index"""
        return card_slot * (32 * 18) + row * 18 + col

    def _decode_action(self, action):
        """Decode action index to (card_slot, row, col)"""
        card_slot = action // (32 * 18)
        remainder = action % (32 * 18)
        row = remainder // 18
        col = remainder % 18
        return card_slot, row, col

    def _select_best_card(self, enemy_detections):
        """
        Select the best card to play based on:
        1. Elixir availability
        2. Card availability (not on cooldown)
        3. Enemy threats (air vs ground, type counters)

        Returns:
            (card_slot, card_info) tuple, or (None, None) if no playable card
        """
        from detection.card_info import get_card_elixir, can_target_air, CARD_INFO

        # Check if we have any enemies
        has_air_enemies = any('air' in d.get('class_name', '') for d in enemy_detections)

        # Build list of playable cards
        playable_cards = []

        for slot_idx, card in enumerate(self.current_hand):
            if card is None:
                continue

            card_name = card['card_name']
            available = card.get('available', True)

            # Skip if card is on cooldown
            if not available:
                continue

            # Get card stats from CARD_INFO
            card_stats = CARD_INFO.get(card_name, {})
            elixir_cost = card_stats.get('elixir_cost', 10)  # Default high cost if unknown

            # Skip if not enough elixir
            if self.current_elixir is not None and self.current_elixir < elixir_cost:
                continue

            # Add elixir cost and stats to card info
            card_with_stats = card.copy()
            card_with_stats['elixir_cost'] = elixir_cost
            card_with_stats['can_target_air'] = can_target_air(card_name)
            card_with_stats['targets'] = card_stats.get('targets', [])

            # Calculate priority score
            priority = 0

            # Prioritize air-targeting cards if there are air enemies
            if has_air_enemies and card_with_stats['can_target_air']:
                priority += 10

            # Prioritize cheaper cards (elixir efficiency)
            priority += (10 - elixir_cost) * 0.5

            # Prioritize by card type for defense
            card_type = card.get('card_type', 'unknown')
            if card_type == 'building':
                priority += 5  # Buildings are good defensive structures
            elif card_type == 'ranged':
                priority += 3  # Ranged units are versatile
            elif card_type == 'tank':
                priority += 2  # Tanks are strong but slow

            playable_cards.append((slot_idx, card_with_stats, priority))

        if not playable_cards:
            return None, None

        # Sort by priority (highest first) and pick best card
        playable_cards.sort(key=lambda x: x[2], reverse=True)
        best_slot, best_card, _ = playable_cards[0]

        return best_slot, best_card

    def _get_smart_placement(self, card_info, enemy_detections):
        """
        Determine smart placement based on card type and enemy positions

        Buildings: Place in front of towers (defensive position)
        Air-counters: Place near air enemies
        Other cards: Use weighted threat placement

        Returns:
            (row, col) tuple, or (None, None) if no valid placement
        """
        card_type = card_info.get('card_type', 'unknown')
        card_name = card_info.get('card_name', 'unknown')

        print(f"[DEBUG] Card: {card_name}, Type: {card_type}")

        # Buildings always go in defensive positions (front of towers)
        if card_type == 'building':
            # Place in our territory, centered
            row = random.randint(22, 28)  # Closer to our towers
            col = random.randint(6, 11)   # Center column
            print(f"[PLACEMENT] Building placement (defensive)")
            return row, col

        # Check if there are air enemies and we can target air
        has_air_enemies = any('air' in d.get('class_name', '') for d in enemy_detections)
        can_hit_air = card_info.get('can_target_air', False)

        if has_air_enemies and can_hit_air:
            # Place near air enemies
            air_enemies = [d for d in enemy_detections if 'air' in d.get('class_name', '')]
            if air_enemies:
                # Get average position of air enemies
                avg_col = sum(d.get('grid', (0, 9))[1] for d in air_enemies) / len(air_enemies)

                # Place on same side as air enemies (avoid center columns)
                if avg_col < 9:
                    col = random.randint(0, 7)  # Left side, avoid column 8
                else:
                    col = random.randint(10, 17)  # Right side, avoid column 9

                row = random.randint(20, 28)  # Defensive position
                print(f"[PLACEMENT] Anti-air placement (targeting air enemies)")
                return row, col

        # Use weighted threat placement for everything else
        return self._get_weighted_placement(enemy_detections)

    def _get_weighted_placement(self, enemy_detections):
        """
        Determine where to place card based on enemy positions minus ally presence.

        Threat = Raw Enemy Threat - (Ally Threat / 2)

        For example:
        - 3 enemy melee = 3 threat
        - 4 ally melee on same side = 4/2 = 2 defense
        - Net threat = 3 - 2 = 1

        Places on the side with higher net threat.
        """
        # If model is not enabled, just place randomly
        if not self.use_model or not self.card_detector:
            row = random.randint(16, 31)
            col = random.randint(0, 17)
            return row, col

        # Threat values based on actual troop strength
        THREAT_VALUES = {
            # Troops - Low threat
            'skeletons': 1,
            'spear_goblins': 2,
            'goblins': 3,
            'fire_spirit': 3,
            'electro_spirit': 2,
            'bomber': 4,
            # Troops - Medium threat
            'archers': 4,
            'minions': 5,
            'furnace': 5,  # Reworked - now a troop
            # Troops - Medium-high threat
            'barbarians': 6,
            'skeleton_dragons': 7,
            # Troops - High threat
            'knight': 9,
            'mega_minion': 9,
            # Troops - Very high threat
            'musketeer': 10,
            'valkyrie': 11,
            'wizard': 12,
            'mini_pekka': 14,
            # Troops - Extreme threat
            'battle_ram': 16,
            'giant': 18,
            # Spells
            'arrows': 5,
            'fireball': 8,
            # Buildings
            'cannon': 0,
            'bomb_tower': 0,
            'inferno_tower': 0,
            'tombstone': 0,
            'goblin_hut': 4,
            'goblin_cage': 0,
        }

        def get_threat_value(card_name, card_type_fallback):
            """Get threat value with fallback to category-based estimate"""
            # Try exact match first
            if card_name in THREAT_VALUES:
                return THREAT_VALUES[card_name]
            # Category-based fallback
            category_defaults = {
                'melee': 5,
                'ranged': 4,
                'tank': 10,
                'air': 5,
                'building': 0
            }
            return category_defaults.get(card_type_fallback, 5)

        # Get all detections (both ally and enemy)
        all_detections = []
        if self.card_detector:
            all_detections = self.card_detector.detect(self.gc.take_screenshot(), verbose=False)

        # Separate ally and enemy detections
        ally_detections = [d for d in all_detections if d['class_name'].startswith('ally')]

        # Calculate raw enemy threat for left (col 0-8) and right (col 9-17)
        left_enemy_threat = 0
        right_enemy_threat = 0

        left_ally_defense = 0
        right_ally_defense = 0

        # Calculate enemy threats
        for detection in enemy_detections:
            class_name = detection.get('class_name', 'enemy')

            # Extract card name and type
            # Format can be: "enemy", "enemy_tank", or potentially "enemy_wizard"
            if '_' in class_name:
                parts = class_name.split('_', 1)
                card_identifier = parts[1]  # Remove 'enemy_' prefix
            else:
                card_identifier = 'melee'  # Default

            # Try to get specific threat value
            threat = get_threat_value(card_identifier, card_identifier)

            # Get column position from grid coordinates
            grid = detection.get('grid')
            if grid:
                grid_row, grid_col = grid
                col = grid_col
            else:
                col = 9  # Default to middle if no grid data

            if col < 9:
                left_enemy_threat += threat
            else:
                right_enemy_threat += threat

        # Calculate ally defense
        for detection in ally_detections:
            class_name = detection.get('class_name', 'ally')

            # Extract card name and type
            if '_' in class_name:
                parts = class_name.split('_', 1)
                card_identifier = parts[1]  # Remove 'ally_' prefix
            else:
                card_identifier = 'melee'  # Default

            # Try to get specific threat value
            threat = get_threat_value(card_identifier, card_identifier)

            # Get column position from grid coordinates
            grid = detection.get('grid')
            if grid:
                grid_row, grid_col = grid
                col = grid_col
            else:
                col = 9  # Default to middle if no grid data

            if col < 9:
                left_ally_defense += threat
            else:
                right_ally_defense += threat

        # Calculate net threat: Raw Enemy Threat - (Ally Defense / 2)
        left_net_threat = left_enemy_threat - (left_ally_defense / 2.0)
        right_net_threat = right_enemy_threat - (right_ally_defense / 2.0)

        # Debug output
        print(f"Left: Enemy={left_enemy_threat}, Ally={left_ally_defense}, Net Threat={left_net_threat:.1f}")
        print(f"Right: Enemy={right_enemy_threat}, Ally={right_ally_defense}, Net Threat={right_net_threat:.1f}")

        # Decide which side to play on (place where net threat is higher)
        # NOTE: Grid coordinates are inverted - swap left/right

        # If no net threat on either side, don't place
        if left_net_threat <= 0 and right_net_threat <= 0:
            print(f"No net threat - not placing")
            return None, None

        # If equal net threat (both > 0), place randomly (but avoid center)
        if left_net_threat == right_net_threat and left_net_threat > 0:
            if random.random() < 0.5:
                col = random.randint(0, 7)  # Left side
            else:
                col = random.randint(10, 17)  # Right side
            print(f"Placing RANDOM (equal net threat)")
        # More net threat on left, play on RIGHT side (inverted) - avoid center
        elif left_net_threat > right_net_threat:
            col = random.randint(0, 7)  # Left side, avoid columns 8-9
            print(f"Placing LEFT (higher threat on left)")
        # More net threat on right, play on LEFT side (inverted) - avoid center
        else:
            col = random.randint(10, 17)  # Right side, avoid columns 8-9
            print(f"Placing RIGHT (higher threat on right)")

        # Always play in our territory (rows 16-31)
        row = random.randint(16, 31)

        return row, col

    def handle_battle_end(self):
        # Log battle result if detected
        if self.battle_result is not None:
            print(f"[BATTLE RESULT] {self.battle_result.upper()}")

        # RL: End episode and save checkpoint
        if self.use_rl and self.rl_agent is not None:
            # Mark episode as done
            self.rl_agent.end_episode()

            # Save checkpoint every 5 games
            if self.games_played % 5 == 0:
                checkpoint_path = f"checkpoints/checkpoint_game_{self.games_played}.pt"
                self.rl_agent.save_checkpoint(checkpoint_path)

                # Also save as "latest"
                self.rl_agent.save_checkpoint("checkpoints/latest.pt")

            # Print episode stats
            stats = self.rl_agent.get_stats()
            print(f"\n[RL EPISODE END]")
            print(f"  Episodes: {stats['episodes']}")
            print(f"  Avg Reward (100): {stats['avg_reward_100']:.2f}")
            print(f"  Epsilon: {stats['epsilon']:.3f}")
            print(f"  Buffer: {stats['buffer_size']}/{10000}")

            # Reset state for next episode
            self.prev_state = None
            self.prev_action = None
            self.prev_state_vector = None
            self.cached_tower_hp = None

        print("finishing game ok button")
        self.gc.click_ok_button()
        self.games_played += 1

        # Reset battle result for next game
        self.battle_result = None

        time.sleep(3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clash Royale RL Agent")
    parser.add_argument("--instance", type=int, default=0, help="Bluestacks Instance ID")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--screenshots", action="store_true", help="Save screenshots every 3 seconds for training")
    parser.add_argument("--model", action="store_true", help="Use YOLO model to detect troops and print detections")
    parser.add_argument("--rl", action="store_true", help="Enable RL training mode with DQN agent")

    args = parser.parse_args()

    agent = Agent(
        instance_id=args.instance,
        save_screenshots=args.screenshots,
        use_model=args.model or args.rl,  # RL requires model for troop detection
        use_rl=args.rl
    )
    agent.play_games(num_games=args.games)
