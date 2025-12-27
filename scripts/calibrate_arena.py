# $env:Path += ";C:\Program Files\BlueStacks_nxt"
# HD-Adb devices
# python scripts/calibrate_arena.py --instance-id 0 --port 5554

import cv2
import numpy as np
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from controllers.adb_controller import ADBController
from detection.grid_system import ArenaConfig, GridSystem

class ArenaCalibrator:
    def __init__(self, adb_port: int):
        self.adb = ADBController(adb_port)
        self.points = list()
        self.screenshot = None
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Added point: {x}, {y}")

            if self.screenshot is not None:
                cv2.circle(self.screenshot, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Calibration", self.screenshot)

    def calibrate(self):
        print("ADB Connection Test:")
        if not self.adb.test_connection():
            raise RuntimeError("Failed to connect to ADB. Please check your connection and try again.")
        
        print("Taking screenshot:")
        screenshot = self.adb.screenshot()
        if screenshot is None:
            raise RuntimeError("Failed to capture screenshot. Please check your ADB connection and try again.")
        
        self.screenshot = screenshot.copy()
        h, w = screenshot.shape[:2]
        print(f"Screen size: {w}x{h}")

        print("click 4 points in this order:")
        print("1. Top left corner")
        print("2. Top right corner")
        print("3. Bottom left corner")
        print("4. Bottom right corner")

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        cv2.imshow("Calibration", self.screenshot)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(self.points) == 4: # quit
                break
            elif key == ord('r'): # reset
                self.points = list()
                self.screenshot = screenshot.copy()
                cv2.imshow("Calibration", self.screenshot)
                print("Resetted points. click 4 points again")

        cv2.destroyAllWindows()

        top_left, top_right, bottom_left, bottom_right = self.points

        arena_left = min(top_left[0], bottom_left[0])
        arena_right = max(top_right[0], bottom_right[0])
        arena_top = min(top_left[1], top_right[1])
        arena_bottom = max(bottom_left[1], bottom_right[1])

        return ArenaConfig(
            screen_width=w,
            screen_height=h,
            arena_top=arena_top,
            arena_bottom=arena_bottom,
            arena_left=arena_left,
            arena_right=arena_right
        )
    
    def visualize_arena(self, config: ArenaConfig) -> None:
        grid = GridSystem(config)
        if self.screenshot is None:
            raise RuntimeError("Screenshot is not available. Please calibrate first.")
        overlay = self.screenshot.copy()
        
        # Draw grid lines
        for row in range(grid.grid_rows + 1):
            y = int(config.arena_top + row * grid.cell_height)
            cv2.line(overlay, (config.arena_left, y), 
                    (config.arena_right, y), (0, 255, 0), 1)
        
        for col in range(grid.grid_cols + 1):
            x = int(config.arena_left + col * grid.cell_width)
            cv2.line(overlay, (x, config.arena_top), 
                    (x, config.arena_bottom), (0, 255, 0), 1)
        
        # Highlight valid positions (player's half)
        valid_positions = grid.get_valid_positions()
        for row, col in valid_positions:
            x, y = grid.grid_to_pixel(row, col)
            cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
        
        # Draw arena boundary
        cv2.rectangle(overlay, 
                     (config.arena_left, config.arena_top),
                     (config.arena_right, config.arena_bottom),
                     (255, 0, 0), 2)
        
        cv2.imshow("Grid Overlay", overlay)
        print("\nGrid visualization:")
        print(f"- Green lines: Grid cells ({grid.grid_rows}x{grid.grid_cols})")
        print(f"- Red dots: Valid placement positions ({len(valid_positions)} total)")
        print(f"- Blue rectangle: Arena boundary")
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate arena boundaries")
    parser.add_argument("--port", type=int, default=5555, help="ADB port")
    parser.add_argument("--instance-id", type=int, default=0, help="Instance ID")
    args = parser.parse_args()
    
    print(f"Calibrating instance {args.instance_id} on port {args.port}")
    
    calibrator = ArenaCalibrator(args.port)
    
    # Calibrate
    config = calibrator.calibrate()
    
    print("\n=== Calibration Results ===")
    print(f"Screen: {config.screen_width}x{config.screen_height}")
    print(f"Arena: left={config.arena_left}, right={config.arena_right}")
    print(f"       top={config.arena_top}, bottom={config.arena_bottom}")
    
    # Save config
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / f"instance_{args.instance_id}.json"
    
    config_data = {
        "instance_id": args.instance_id,
        "adb_port": args.port,
        "arena_config": {
            "screen_width": config.screen_width,
            "screen_height": config.screen_height,
            "arena_top": config.arena_top,
            "arena_bottom": config.arena_bottom,
            "arena_left": config.arena_left,
            "arena_right": config.arena_right
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n✓ Config saved to {config_path}")
    
    # Visualize grid
    calibrator.visualize_arena(config)

if __name__ == "__main__":
    main()