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

# Main agent who runs the game
class Agent:
    def __init__(self, instance_id=0, save_screenshots=False, use_model=False):
        self.gc = GameController(instance_id)
        self.detector = StateDetector(self.gc.image_matcher)
        self.games_played = 0

        # Screenshot saving setup
        self.save_screenshots = save_screenshots
        self.screenshot_interval = 3.0  # Fixed at 3 seconds
        self.last_save_time = 0
        self.screenshot_count = 0
        self.output_dir = None

        # Card hand detection
        self.hand_detector = CardHandDetector()
        self.current_hand = []

        # Elixir detection
        self.elixir_detector = ElixirDetector()
        self.current_elixir = None

        # Battle state
        self.last_play_time = 0
        self.play_interval = random.uniform(2.0, 3.0)  # 2-3 seconds between plays

        # YOLO model setup
        self.use_model = use_model
        self.card_detector = None
        if self.use_model:
            print("Loading YOLO model...")
            self.card_detector = CardDetector(
                model_path="models/best-2.pt",
                confidence_threshold=0.5,
                grid_system=self.gc.grid
            )
            print("Model loaded successfully!")

        if self.save_screenshots:
            self.output_dir = self._create_output_dir()

    def _create_output_dir(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"training_data/session_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving screenshots to: {output_dir}")
        # Initialize last_save_time to allow immediate first screenshot
        self.last_save_time = 0
        return output_dir

    def _save_screenshot(self, screenshot, state):
        self.screenshot_count += 1
        filename = f"screenshot_{self.screenshot_count:04d}_{state.name}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, screenshot)
        # print(f"Saved: {filename}")
        # ^ use this only for debugging

    def play_games(self, num_games=1):
        print(f"Agent is now playing {num_games} games...")

        while self.games_played < num_games:
            screenshot = self.gc.take_screenshot()

            if screenshot is None:
                continue

            state = self.detector.detect_state(screenshot)

            # Run YOLO detection if enabled and in battle
            if self.use_model and state == GameState.IN_BATTLE:
                detections = self.card_detector.detect(screenshot, verbose=False)

            # Save screenshot if enabled and in battle
            if self.save_screenshots and state == GameState.IN_BATTLE:
                current_time = time.time()
                if current_time - self.last_save_time >= self.screenshot_interval:
                    self._save_screenshot(screenshot, state)
                    self.last_save_time = current_time

            if state == GameState.MAIN_MENU:
                self.handle_main_menu()
            elif state == GameState.IN_BATTLE:
                self.handle_battle(screenshot)
            elif state == GameState.BATTLE_END:
                self.handle_battle_end()
            else:
                time.sleep(1)
    
    def handle_main_menu(self):
        print("Handling main menu:")
        self.gc.click_battle_button()
        time.sleep(5)
    
    def handle_battle(self, screenshot):
        current_time = time.time()

        # Check elixir frequently (4 times per second)
        # This runs every loop iteration to catch elixir changes
        new_elixir = self.elixir_detector.get_elixir(screenshot, verbose=False)
        if new_elixir is not None and new_elixir != self.current_elixir:
            self.current_elixir = new_elixir
            print(f"Elixir: {self.current_elixir}")

        # Only play every 2-3 seconds
        if current_time - self.last_play_time < self.play_interval:
            time.sleep(0.25)  # Check elixir 4 times per second
            return

        # Detect cards in hand
        self.current_hand = self.hand_detector.get_hand(screenshot, verbose=False)

        # Get all detections if YOLO model is enabled
        all_detections = []
        enemy_detections = []
        if self.use_model and self.card_detector:
            all_detections = self.card_detector.detect(screenshot, verbose=False)

            # Separate enemy detections for placement logic
            enemy_detections = [d for d in all_detections if d['class_name'].startswith('enemy_')]

            # If no enemies detected, don't place cards
            if len(enemy_detections) == 0:
                return

        # Determine placement based on enemy positions
        row, col = self._get_weighted_placement(enemy_detections)

        # Pick a random card to play
        card = random.randint(0, 3)

        # Print info about the play
        card_info = self.current_hand[card] if card < len(self.current_hand) and self.current_hand[card] else None
        if card_info:
            card_name = card_info['card_name']
            card_type = card_info['card_type']
            available = card_info['available']
            status = "✓" if available else "✗"
            # print(f"Playing {card_name} ({card_type}) {status} at row {row}, col {col}")
        else:
            pass
            # print(f"Playing card {card} at row {row}, col {col}")
        self.gc.play_card(card, row, col)

        # Update timing
        self.last_play_time = current_time
        self.play_interval = random.uniform(2.0, 3.0)

    def _get_weighted_placement(self, enemy_detections):
        """
        Determine where to place card based on enemy positions.
        Places on the side with more enemy weight.

        Temporary weights:
        - enemy_ranged: 1
        - enemy_melee: 1
        - enemy_tank: 3
        - enemy_building: 0
        - enemy_air: 1
        """
        # Weight mapping
        weights = {
            'ranged': 1,
            'melee': 1,
            'tank': 3,
            'building': 0,
            'air': 1
        }

        # Calculate weighted positions for left (col 0-8) and right (col 9-17)
        left_weight = 0
        right_weight = 0

        for detection in enemy_detections:
            # Extract card type from class_name (e.g., "enemy_tank" -> "tank")
            class_name = detection.get('class_name', 'enemy_melee')
            card_type = class_name.replace('enemy_', '')  # Remove 'enemy_' prefix
            weight = weights.get(card_type, 1)

            # Get column position from grid coordinates
            grid = detection.get('grid')
            if grid:
                grid_row, grid_col = grid
                col = grid_col
            else:
                col = 9  # Default to middle if no grid data

            if col < 9:
                left_weight += weight
            else:
                right_weight += weight

        # Debug output
        print(f"Enemy weights: Left={left_weight}, Right={right_weight}")

        # Decide which side to play on (place where enemies are)
        # NOTE: Grid coordinates are inverted - swap left/right
        if left_weight > right_weight:
            # More enemies on left, play on RIGHT side (inverted)
            col = random.randint(0, 8)
            print(f"Placing LEFT (enemies on left)")
        elif right_weight > left_weight:
            # More enemies on right, play on LEFT side (inverted)
            col = random.randint(9, 17)
            print(f"Placing RIGHT (enemies on right)")
        elif right_weight == left_weight and right_weight != 0:
            # Equal enemies, play randomly anywhere
            col = random.randint(0, 17)
            print(f"Placing RANDOM (equal/no enemies)")

        # Always play in our territory (rows 16-31)
        row = random.randint(16, 31)

        return row, col

    def handle_battle_end(self):
        print("finishing game ok button")
        self.gc.click_ok_button()
        self.games_played += 1
        time.sleep(3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clash Royale RL Agent")
    parser.add_argument("--instance", type=int, default=0, help="Bluestacks Instance ID")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--screenshots", action="store_true", help="Save screenshots every 3 seconds for training")
    parser.add_argument("--model", action="store_true", help="Use YOLO model to detect troops and print detections")

    args = parser.parse_args()

    agent = Agent(instance_id=args.instance, save_screenshots=args.screenshots, use_model=args.model)
    agent.play_games(num_games=args.games)
