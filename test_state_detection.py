#!/usr/bin/env python3
"""
Quick test script for state detection
Run this to verify state detection is working correctly
"""
import cv2
import time
from controllers.game_controller import GameController
from detection.state_detector import StateDetector, GameState

print("="*60)
print("State Detection Test")
print("="*60)
print("Make sure Clash Royale is open in BlueStacks!")
print("\nThis will check the current state every 2 seconds.")
print("Navigate through different screens to test detection.")
print("Press Ctrl+C to exit\n")

gc = GameController(instance_id=0)
detector = StateDetector(gc.image_matcher)

state_emoji = {
    GameState.MAIN_MENU: "🏠",
    GameState.IN_BATTLE: "⚔️",
    GameState.BATTLE_END: "🏆",
    GameState.QUEUEING: "⏳",
    GameState.LOADING: "🔄",
    GameState.UNKNOWN: "❓"
}

try:
    while True:
        screenshot = gc.take_screenshot()
        if screenshot is None:
            print("❌ Failed to capture screenshot")
            time.sleep(2)
            continue
        
        state = detector.detect_state(screenshot)
        emoji = state_emoji.get(state, "?")
        
        print(f"{emoji} Current State: {state.name}")
        
        time.sleep(2)
        
except KeyboardInterrupt:
    print("\n\nTest stopped by user")
