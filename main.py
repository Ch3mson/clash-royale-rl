# $env:Path += ";C:\Program Files\BlueStacks_nxt"
# HD-Adb devices
import time
import random
import sys
import argparse
from controllers.game_controller import GameController
from detection.state_detector import StateDetector, GameState

# Main agent who runs the game
class Agent:
    def __init__(self, instance_id=0):
        self.gc = GameController(instance_id)
        self.detector = StateDetector(self.gc.image_matcher)
        self.games_played = 0

    def play_games(self, num_games=1):
        print(f"Agent is now playing {num_games} games...")

        while self.games_played < num_games:
            screenshot = self.gc.take_screenshot()

            if screenshot is None:
                continue

            state = self.detector.detect_state(screenshot)

            if state == GameState.MAIN_MENU:
                self.handle_main_menu()
            elif state == GameState.IN_BATTLE:
                self.handle_battle()
            elif state == GameState.BATTLE_END:
                self.handle_battle_end()
            else:
                time.sleep(1)
    
    def handle_main_menu(self):
        print("Handling main menu:")
        self.gc.click_battle_button()
        time.sleep(5)
    
    def handle_battle(self):
        card = random.randint(0, 3)
        row = random.randint(16, 31)
        col = random.randint(0, 17)

        print(f"playing {card} at {row}, {col}")
        self.gc.play_card(card, row, col)
        time.sleep(random.uniform(1.5, 2.5))

    def handle_battle_end(self):
        print("finishing game ok button")
        self.gc.click_ok_button()
        self.games_played += 1
        time.sleep(3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clash Royale RL Agent")
    parser.add_argument("--instance", type=int, default=0, help="Bluestacks Instance ID")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")

    args = parser.parse_args()

    agent = Agent(instance_id=args.instance)
    agent.play_games(num_games=args.games)
