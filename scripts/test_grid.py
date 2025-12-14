# scripts/test_grid.py
import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from controllers.adb_controller import ADBController
from detection.grid_system import ArenaConfig, GridSystem
import cv2

def load_config(instance_id: int):
    """Load instance configuration"""
    config_path = Path(__file__).parent.parent / "config" / f"instance_{instance_id}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}\nRun calibrate_arena.py first!")
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    arena_config = ArenaConfig(**data["arena_config"])
    return data["adb_port"], arena_config

def test_grid_clicking(instance_id: int):
    """Test clicking all valid grid positions"""
    # Load config
    adb_port, arena_config = load_config(instance_id)
    
    # Initialize controllers
    adb = ADBController(adb_port)
    grid = GridSystem(arena_config)
    
    # Test connection
    print(f"Testing connection to port {adb_port}...")
    if not adb.test_connection():
        print("❌ ADB connection failed!")
        return
    print("✓ ADB connection successful!")
    
    # Get valid positions
    valid_positions = grid.get_valid_positions()
    print(f"\n✓ Grid system initialized: {grid.grid_rows}x{grid.grid_cols}")
    print(f"✓ Valid positions: {len(valid_positions)}")
    
    # Test clicking each position
    print("\n🎯 Testing grid positions...")
    print("This will click each valid position with a visual marker")
    input("Press Enter to start (make sure BlueStacks is visible)...")
    
    for i, (row, col) in enumerate(valid_positions[:10]):  # Test first 10 positions
        x, y = grid.grid_to_pixel(row, col)
        print(f"  Position {i+1}/10: Grid({row},{col}) -> Pixel({x},{y})")
        
        # Click position
        success = adb.click(x, y)
        if not success:
            print(f"    ❌ Click failed!")
        else:
            print(f"    ✓ Clicked")
        
        time.sleep(0.5)  # Delay to see the click
    
    print("\n✓ Test complete!")

def visualize_current_screen(instance_id: int):
    """Take a screenshot and show grid overlay"""
    adb_port, arena_config = load_config(instance_id)
    
    adb = ADBController(adb_port)
    grid = GridSystem(arena_config)
    
    print("Taking screenshot...")
    screenshot = adb.screenshot()
    if screenshot is None:
        print("❌ Failed to capture screenshot!")
        return
    
    # Draw grid overlay
    overlay = screenshot.copy()
    
    # Draw grid lines
    for row in range(grid.grid_rows + 1):
        y = int(arena_config.arena_top + row * grid.cell_height)
        cv2.line(overlay, (arena_config.arena_left, y), 
                (arena_config.arena_right, y), (0, 255, 0), 1)
    
    for col in range(grid.grid_cols + 1):
        x = int(arena_config.arena_left + col * grid.cell_width)
        cv2.line(overlay, (x, arena_config.arena_top), 
                (x, arena_config.arena_bottom), (0, 255, 0), 1)
    
    # Mark valid positions
    valid_positions = grid.get_valid_positions()
    for row, col in valid_positions:
        x, y = grid.grid_to_pixel(row, col)
        cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)
        # Add grid coordinates as text
        cv2.putText(overlay, f"{row},{col}", (x-15, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    cv2.imshow("Grid Overlay", overlay)
    print("\n✓ Screenshot captured with grid overlay")
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test grid system")
    parser.add_argument("--instance-id", type=int, default=0, help="Instance ID")
    parser.add_argument("--mode", choices=["click", "visualize"], default="visualize",
                       help="Test mode: 'click' to test clicking, 'visualize' to show grid")
    args = parser.parse_args()
    
    try:
        if args.mode == "click":
            test_grid_clicking(args.instance_id)
        else:
            visualize_current_screen(args.instance_id)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nRun this first:")
        print(f"  python scripts/calibrate_arena.py --instance-id {args.instance_id}")

if __name__ == "__main__":
    main()