# controllers/game_controller.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now the imports will work
import json
import time
import random
from typing import Tuple, Optional
from controllers.adb_controller import ADBController
from detection.grid_system import ArenaConfig, GridSystem
from detection.image_matcher import ImageMatcher
import cv2


class GameController:
    def __init__(self, instance_id: int):
        """
        Initialize game controller for a specific BlueStacks instance
        
        Args:
            instance_id: Instance ID (matches config file)
        """
        self.instance_id = instance_id
        
        config_path = Path("config") / f"instance_{instance_id}.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {config_path}\n"
                f"Run calibrate_arena.py first!"
            )
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        self.adb_port = data["adb_port"]
        self.arena_config = ArenaConfig(**data["arena_config"])
        
        self.adb = ADBController(self.adb_port)
        self.grid = GridSystem(self.arena_config)
        self.image_matcher = ImageMatcher()
        
        screen_width = self.arena_config.screen_width
        screen_height = self.arena_config.screen_height
        
        card_y = int(screen_height * 0.89)

        margin = int(screen_width * 0.20)
        card_width = (screen_width - 2 * margin) / 4

        self.card_positions = [
            (int(margin + card_width * 0.5 + i * card_width) + 50, card_y)
            for i in range(4)
        ]
        
        print(f"GameController initialized for instance {instance_id}")
        print(f"Screen: {screen_width}x{screen_height}")
        print(f"Card positions: {self.card_positions}")
    
    def play_card(self, card_slot: int, row: int, col: int) -> bool:
        """
        Play a card at the specified grid position
        
        Args:
            card_slot: Card index (0-3, left to right)
            row: Grid row (0-31, top to bottom)
            col: Grid column (0-17, left to right)
            
        Returns:
            True if action executed successfully, False otherwise
        """
        if card_slot not in range(4):
            print(f"❌ Invalid card slot: {card_slot} (must be 0-3)")
            return False
        
        start_x, start_y = self.card_positions[card_slot]
        
        target_x, target_y = self.grid.grid_to_pixel(row, col)
        
        target_x += random.randint(-3, 3)
        target_y += random.randint(-3, 3)
        
        print(f"Playing card {card_slot}: ({start_x},{start_y}) → ({target_x},{target_y}) [Grid: {row},{col}]")
        
        success = self.adb.swipe(start_x, start_y, target_x, target_y, duration=200)
        
        if not success:
            print(f"❌ Swipe failed")
            return False
        
        time.sleep(random.uniform(0.15, 0.35))
        
        return True
    
    def test_connection(self) -> bool:
        """Test if ADB connection is working"""
        return self.adb.test_connection()
    
    def take_screenshot(self):
        """Take a screenshot of the game"""
        return self.adb.screenshot()
    
    def click_position(self, x: int, y: int) -> bool:
        """
        Click at specific pixel coordinates
        
        Args:
            x, y: Pixel coordinates
            
        Returns:
            True if click succeeded
        """
        return self.adb.click(x, y)
    
    def get_screen_size(self) -> Optional[Tuple[int, int]]:
        """Get screen dimensions (width, height)"""
        return self.adb.get_screen_size()
    
    def get_grid_info(self) -> dict:
        """Get information about the grid system"""
        return {
            "rows": self.grid.grid_rows,
            "cols": self.grid.grid_cols,
            "arena_bounds": {
                "left": self.arena_config.arena_left,
                "right": self.arena_config.arena_right,
                "top": self.arena_config.arena_top,
                "bottom": self.arena_config.arena_bottom
            },
            "cell_size": {
                "width": self.grid.cell_width,
                "height": self.grid.cell_height
            }
        }

    def get_elixir(self) -> int:
        """
        Get current elixir count using fast pixel detection
        
        The elixir bar in Clash Royale shows 10 segments that fill with pink color.
        We count how many segments are filled by sampling pixel colors.
        
        Returns:
            int: Current elixir (0-10)
        """
        screenshot = self.adb.screenshot()
        if screenshot is None:
            return 0
        
        elixir_count = 0
        
        target_color = [198, 30, 193]
        tolerance = 50
        
        try:
            bar_start_x = 202
            bar_end_x = 231
            bar_y = 1238  # Middle of the bar vertically
            
            # Sample 10 evenly-spaced points
            for i in range(10):
                # Calculate x position for this segment
                x = bar_start_x + int((bar_end_x - bar_start_x) * (i + 0.5) / 10)
                
                # Get pixel color (BGR)
                pixel = screenshot[bar_y, x]
                
                # Check if pixel matches elixir color (within tolerance)
                color_match = all(
                    abs(int(pixel[c]) - target_color[c]) <= tolerance 
                    for c in range(3)
                )
                
                if color_match:
                    elixir_count += 1
            
            return elixir_count
            
        except Exception as e:
            print(f"Error detecting elixir: {e}")
            return 0
    
    def find_button(self, button_name: str, threshold: float = 0.8) -> Optional[Tuple[int, int, float]]:
        """
        Find a UI button in the current screen
        
        Args:
            button_name: Button template filename (e.g., 'battle_button.png')
            threshold: Confidence threshold (0.0-1.0)
            
        Returns:
            (x, y, confidence) if found, else None
        """
        screenshot = self.adb.screenshot()
        if screenshot is None:
            return None
        
        return self.image_matcher.find_template(screenshot, button_name, threshold)
    
    def click_button(self, button_name: str, threshold: float = 0.8) -> bool:
        """
        Find and click a UI button
        
        Args:
            button_name: Button template filename
            threshold: Confidence threshold
            
        Returns:
            True if button found and clicked, False otherwise
        """
        result = self.find_button(button_name, threshold)
        
        if result:
            x, y, confidence = result
            print(f"Found {button_name} at ({x}, {y}) [confidence: {confidence:.2f}]")
            
            x += random.randint(-2, 2) # avoid bot detection with random
            y += random.randint(-2, 2)
            
            time.sleep(random.uniform(0.1, 0.2))
            
            success = self.adb.click(x, y)
            if success:
                time.sleep(random.uniform(0.3, 0.5))
            return success
        else:
            print(f"❌ {button_name} not found")
            return False
    
    def click_battle_button(self) -> bool:
        """
        Click the battle button at fixed position (to start a match)

        Returns:
            True if clicked successfully, False otherwise
        """
        # Battle button is at ~50% x, ~78% y of screen
        x = int(self.arena_config.screen_width * 0.50)
        y = int(self.arena_config.screen_height * 0.78)
        print(f"Clicking battle button at ({x}, {y})")
        return self.adb.click(x, y)
    
    def click_ok_button(self) -> bool:
        """
        Find and click the OK button (after battle ends)
        
        Returns:
            True if clicked successfully, False otherwise
        """
        return self.click_button('ok_button.png', threshold=0.6)
    
    def visualize_button(self, button_name: str, output_path: str = "button_debug.png") -> bool:
        """
        Visualize button detection (for debugging/calibration)
        
        Args:
            button_name: Button template filename
            output_path: Where to save visualization
            
        Returns:
            True if visualization saved, False otherwise
        """
        screenshot = self.adb.screenshot()
        if screenshot is None:
            return False
        
        return self.image_matcher.visualize_match(screenshot, button_name, 0.8, output_path)
