# python scripts/find_coordinates.py --instance-id 0 --port 5554
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from controllers.adb_controller import ADBController

def mouse_callback(event, x, y, flags, param):
    """Callback to show coordinates when clicking"""
    if event == cv2.EVENT_LBUTTONDOWN:
        screenshot, original = param
        print(f"\n📍 Clicked at: ({x}, {y})")
        
        # Get pixel color at that position
        pixel = original[y, x]
        print(f"   Color (BGR): {pixel}")
        print(f"   Color (RGB): [{pixel[2]}, {pixel[1]}, {pixel[0]}]")
        
        # Draw a marker
        cv2.circle(screenshot, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(screenshot, f"({x},{y})", (x+10, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow("Find Coordinates", screenshot)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find pixel coordinates interactively")
    parser.add_argument("--port", type=int, default=5554, help="ADB port")
    parser.add_argument("--instance-id", type=int, default=0, help="Instance ID")
    args = parser.parse_args()
    
    print(f"Connecting to instance {args.instance_id} on port {args.port}")
    
    # Use ADBController directly like calibrate_arena does
    adb = ADBController(adb_port=args.port)
    
    # Test connection
    if not adb.test_connection():
        print("❌ ADB connection failed!")
        return
    
    print("✓ ADB connection successful")
    print("Taking screenshot...")
    
    original = adb.screenshot()
    if original is None:
        print("❌ Failed to capture screenshot!")
        return
    
    screenshot = original.copy()
    h, w = screenshot.shape[:2]
    
    print(f"\n✓ Screenshot captured: {w}x{h}")
    print("\n🖱️  Click anywhere to see coordinates")
    print("Press 'q' to quit, 'r' to reset markers\n")
    
    cv2.namedWindow("Find Coordinates")
    cv2.setMouseCallback("Find Coordinates", mouse_callback, (screenshot, original))
    cv2.imshow("Find Coordinates", screenshot)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            screenshot = original.copy()
            cv2.imshow("Find Coordinates", screenshot)
            print("\n✓ Markers reset")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()