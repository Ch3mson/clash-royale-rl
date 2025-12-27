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
    QUEUEING = 3
    LOADING = 4
    UNKNOWN = 5

class StateDetector:
    def __init__(self, image_matcher: ImageMatcher):
        self.image_matcher = image_matcher

    def detect_state(self, screenshot: np.ndarray, verbose: bool = False) -> GameState:
        """
        Detect game state with improved reliability using region-based checks

        Detection order:
        1. IN_BATTLE (most reliable - elixir bar color check)
        2. MAIN_MENU (bottom navigation bar check - always visible)
        3. BATTLE_END (OK button after battle - check before QUEUEING)
        4. QUEUEING (loading screen / matchmaking)
        5. UNKNOWN (fallback)

        Args:
            screenshot: Game screenshot
            verbose: If True, print detection details
        """
        # 1. Check if in battle (most reliable - pink elixir bar)
        if self._check_elixir_bar_visible(screenshot):
            if verbose:
                print("[STATE] Detected: IN_BATTLE (elixir bar visible)")
            return GameState.IN_BATTLE

        # 2. Check for main menu (bottom navigation bar - always visible regardless of page)
        main_menu_result = self._check_main_menu(screenshot, verbose=verbose)
        if main_menu_result:
            if verbose:
                print("[STATE] Detected: MAIN_MENU")
            return GameState.MAIN_MENU

        # 3. Check for battle end (OK button) - BEFORE queueing check
        # Battle end screen also has dark background, so check OK button first
        ok_button = self.image_matcher.find_template(screenshot, 'ok_button', threshold=0.85)
        if ok_button:
            if verbose:
                x, y, conf = ok_button
                print(f"[STATE] Detected: BATTLE_END (OK button at {x},{y} conf={conf:.2f})")
            return GameState.BATTLE_END

        # 4. Check for queueing/loading state (after OK button to avoid false positives)
        if self._check_queueing_state(screenshot):
            if verbose:
                print("[STATE] Detected: QUEUEING (dark overlay)")
            return GameState.QUEUEING

        if verbose:
            print("[STATE] Detected: UNKNOWN (no match)")
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

        except Exception:
            return False

    def _check_main_menu(self, screenshot: np.ndarray, verbose: bool = False) -> bool:
        """
        Check if in main menu by detecting the bottom UI elements

        The bottom UI is always present on the home screen regardless of which
        of the 5 pages you're on. This is more reliable than template matching buttons.

        Checks for:
        1. Battle button (large yellow/orange button in lower center)
        2. Bottom navigation icons (very bottom of screen)
        """
        try:
            h, w = screenshot.shape[:2]

            # Sample multiple y positions for Battle button (it's quite large)
            # Battle button is in lower portion but above the bottom nav
            battle_y_positions = [h - 250, h - 200, h - 150]

            yellow_count = 0
            for y_pos in battle_y_positions:
                # Sample across the width at this y position
                sample_positions = [
                    (w // 2, y_pos),      # Center
                    (w // 2 - 60, y_pos), # Left of center
                    (w // 2 + 60, y_pos), # Right of center
                ]

                for x, y in sample_positions:
                    if y >= h or x >= w or y < 0 or x < 0:
                        continue

                    pixel = screenshot[y, x]
                    b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])

                    # Check for yellow/gold/orange Battle button (BGR format)
                    # More lenient thresholds to catch variations
                    is_yellow = (
                        r > 100 and        # Red component
                        g > 80 and         # Green component
                        b < 150 and        # Low blue (not blue/cyan)
                        r > b and          # More red than blue
                        g > b - 30         # Green roughly equal to or higher than blue
                    )

                    if is_yellow:
                        yellow_count += 1
                        if yellow_count >= 3:  # Early exit if we found enough
                            return True

            # Check bottom navigation bar (very bottom - around y = h - 30 to h - 70)
            # This area has the navigation icons (chest, cards, battle, clan, etc.)
            nav_y_positions = [h - 30, h - 50, h - 70]
            nav_detected = 0

            for y_pos in nav_y_positions:
                nav_samples = [
                    (w // 5, y_pos),       # Far left
                    (w // 2, y_pos),       # Center
                    (4 * w // 5, y_pos),   # Far right
                ]

                for x, y in nav_samples:
                    if y >= h or x >= w or y < 0 or x < 0:
                        continue

                    pixel = screenshot[y, x]
                    brightness = int(pixel[0]) + int(pixel[1]) + int(pixel[2])

                    # Navigation bar elements are either very dark or bright
                    is_nav_element = brightness < 100 or brightness > 350

                    if is_nav_element:
                        nav_detected += 1
                        if nav_detected >= 4:  # Early exit
                            return True

            # Main menu if we detect yellow Battle button OR navigation elements
            if verbose:
                print(f"[MAIN_MENU] Yellow pixels: {yellow_count}/9, Nav elements: {nav_detected}/9")

            if yellow_count >= 3 or nav_detected >= 4:
                return True

            return False

        except Exception as e:
            if verbose:
                print(f"[MAIN_MENU] Exception: {e}")
            return False

    def _check_queueing_state(self, screenshot: np.ndarray) -> bool:
        """
        Check if in queueing/matchmaking state

        Queueing screen typically has:
        - Dark/blurred background
        - "BATTLE" text at top
        - Loading spinner or matchmaking animation
        - Cancel button at bottom

        We'll check for dark regions and lack of battle/menu UI elements
        """
        try:
            # Check center area for dark/blurred background (typical of queueing)
            # Sample several points in middle of screen
            h, w = screenshot.shape[:2]
            center_samples = [
                (w // 2, h // 2),
                (w // 2, h // 3),
                (w // 2, 2 * h // 3),
            ]

            dark_count = 0
            for x, y in center_samples:
                if y >= h or x >= w:
                    continue

                pixel = screenshot[y, x]
                # Check if pixel is relatively dark (typical of overlay/loading screens)
                brightness = int(pixel[0]) + int(pixel[1]) + int(pixel[2])
                if brightness < 200:  # Dark pixel
                    dark_count += 1

            # If most center pixels are dark, likely queueing
            # Also check that we don't have elixir bar (already checked above)
            # and no battle button (already checked above)
            if dark_count >= 2:
                return True

            return False

        except Exception:
            return False
        
if __name__ == "__main__":
    import cv2

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