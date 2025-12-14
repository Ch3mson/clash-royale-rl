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


class GameController:
    def __init__(self, instance_id: int):
        """
        Initialize game controller for a specific BlueStacks instance
        
        Args:
            instance_id: Instance ID (matches config file)
        """
        self.instance_id = instance_id
        
        # Load configuration
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
        
        # Initialize controllers
        self.adb = ADBController(self.adb_port)
        self.grid = GridSystem(self.arena_config)
        
        # Card positions in hand (pixels) - bottom of screen
        # These are approximate and may need calibration for your screen
        # Format: (x, y) for each card slot (0-3, left to right)
        screen_width = self.arena_config.screen_width
        screen_height = self.arena_config.screen_height
        
        # Cards are typically at ~92% down the screen
        card_y = int(screen_height * 0.92)
        
        # Evenly spaced across width, with margins
        margin = int(screen_width * 0.15)
        card_width = (screen_width - 2 * margin) / 4
        
        self.card_positions = [
            (int(margin + card_width * 0.5 + i * card_width), card_y)
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
        # Validate inputs
        if card_slot not in range(4):
            print(f"❌ Invalid card slot: {card_slot} (must be 0-3)")
            return False
        
        # Get starting position (card in hand)
        start_x, start_y = self.card_positions[card_slot]
        
        # Convert grid position to pixel coordinates
        target_x, target_y = self.grid.grid_to_pixel(row, col)
        
        # Add small random offset for humanization (anti-detection)
        target_x += random.randint(-3, 3)
        target_y += random.randint(-3, 3)
        
        print(f"🎴 Playing card {card_slot}: ({start_x},{start_y}) → ({target_x},{target_y}) [Grid: {row},{col}]")
        
        # Execute drag (swipe from card to target)
        # Duration ~200ms feels natural
        success = self.adb.swipe(start_x, start_y, target_x, target_y, duration=200)
        
        if not success:
            print(f"❌ Swipe failed")
            return False
        
        # Small delay to simulate human timing
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


# Example usage / testing
if __name__ == "__main__":
    # Test the game controller
    print("Testing GameController...")
    
    try:
        gc = GameController(instance_id=0)
        
        # Test connection
        if gc.test_connection():
            print("✓ ADB connection working")
        else:
            print("✗ ADB connection failed")
            exit(1)
        
        # Show grid info
        info = gc.get_grid_info()
        print(f"\nGrid: {info['rows']}x{info['cols']}")
        print(f"Cell size: {info['cell_size']['width']:.1f}x{info['cell_size']['height']:.1f} px")
        
        # Uncomment to test card playing (make sure you're in training mode!)
        print("\nTesting card play in 3 seconds...")
        print("Switch to Clash Royale and start training mode!")
        time.sleep(3)
        gc.play_card(card_slot=0, row=24, col=9)  # Play card 0 at center-bottom
        
    except FileNotFoundError as e:
        print(f"✗ {e}")
    except Exception as e:
        print(f"✗ Error: {e}")