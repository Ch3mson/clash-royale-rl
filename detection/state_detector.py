import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from enum import Enum
import numpy as np
from detection.image_matcher import ImageMatcher


class GameState(Enum):
    MAIN_MENU = 0
    IN_BATTLE = 1
    BATTLE_END = 2
    LOADING = 3
    UNKNOWN = 4

class StateDetector:
    def __init__(self, image_matcher: ImageMatcher):
        self.image_matcher = image_matcher
        self.elixir_regions = (202, 1220, 231, 1257)

    def detect_state(self, screenshot: np.ndarray) -> GameState:
        if self._check_elixir_bar_visible(screenshot):
            return GameState.IN_BATTLE
        
        battle_button = self.image_matcher.find_template(screenshot, 'battle_button', threshold=0.6)
        if battle_button:
            return GameState.MAIN_MENU
        
        ok_button = self.image_matcher.find_template(screenshot, 'ok_button', threshold=0.8)
        if ok_button:
            return GameState.BATTLE_END
        
        return GameState.UNKNOWN
    
    def _check_elixir_bar_visible(self, screenshot: np.ndarray) -> bool:
        """Check if elixir bar is visible (indicates in battle)"""
        try:
            test_positions = [
                (67, 1255),
                (67, 1260),
                (200, 1255),
                (200, 1260),
                (350, 1255),
            ]
            
            for x, y in test_positions:
                if y >= screenshot.shape[0] or x >= screenshot.shape[1]:
                    continue
                
                pixel = screenshot[y, x]
                is_pink = (pixel[2] > 150 and
                          pixel[1] < 150 and
                          pixel[0] > 150)
                
                if is_pink:
                    return True
            return False
        
        except Exception as e:
            return False
        
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import cv2
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from controllers.adb_controller import ADBController
    from detection.image_matcher import ImageMatcher
    
    print("=" * 60)
    print("State Detector Test")
    print("=" * 60)
    print("Make sure BlueStacks is running and Clash Royale is open!\n")
    
    # Initialize components
    adb = ADBController(adb_port=5555)
    matcher = ImageMatcher()
    detector = StateDetector(matcher)
    
    # Test connection
    if not adb.test_connection():
        print("❌ ADB connection failed. Is BlueStacks running?")
        exit(1)
    
    print("✓ Connected to BlueStacks\n")
    
    while True:
        print("\n" + "=" * 60)
        input("Press ENTER to detect current state (or Ctrl+C to exit)...")
        
        screenshot = adb.screenshot()
        if screenshot is None:
            print("❌ Failed to capture screenshot")
            continue
        
        # Detect state
        state = detector.detect_state(screenshot)
        
        # Display result with color coding
        state_emoji = {
            GameState.MAIN_MENU: "🏠",
            GameState.IN_BATTLE: "⚔️",
            GameState.BATTLE_END: "🏆",
            GameState.LOADING: "⏳",
            GameState.UNKNOWN: "❓"
        }
        
        print(f"\n{state_emoji.get(state, '?')} Current State: {state.name}")
        
        # Show what was detected
        print("\nDetection Details:")
        
        # Check elixir bar
        elixir_visible = detector._check_elixir_bar_visible(screenshot)
        if elixir_visible:
            print(f"  ✓ Elixir bar detected (in battle)")
        else:
            print(f"  ✗ Elixir bar not detected")
        
        # Check battle button
        battle_btn = matcher.find_template(screenshot, 'battle_button', threshold=0.6)
        if battle_btn:
            x, y, conf = battle_btn
            print(f"  ✓ Battle button found at ({x}, {y}) - confidence: {conf:.2%}")
        else:
            print(f"  ✗ Battle button not found")
        
        # Check OK button
        ok_btn = matcher.find_template(screenshot, 'ok_button', threshold=0.8)
        if ok_btn:
            x, y, conf = ok_btn
            print(f"  ✓ OK button found at ({x}, {y}) - confidence: {conf:.2%}")
        else:
            print(f"  ✗ OK button not found")
        
        # Optionally save screenshot for debugging
        save = input("\nSave screenshot for debugging? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"debug_state_{state.name.lower()}.png"
            cv2.imwrite(filename, screenshot)
            print(f"✓ Saved to {filename}")