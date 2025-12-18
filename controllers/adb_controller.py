import subprocess
import cv2
import numpy as np
from typing import Optional, Tuple


class ADBController:
    def __init__(self, adb_port: int):
        self.adb_port = adb_port
        self.device_serial = f"127.0.0.1:{self.adb_port}"
    
    def _run_adb_command(self, command: str, binary_output: bool = False) -> subprocess.CompletedProcess:
        full_command = f'adb -s {self.device_serial} {command}'
        return subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            check=False
        )

    def screenshot(self) -> Optional[np.ndarray]:
        try:
            result = self._run_adb_command('exec-out screencap -p', binary_output=True)
            if result.returncode != 0 or not result.stdout:
                raise RuntimeError(f"ADB screencap failed: {result.stderr.decode('utf-8', 'ignore')}")
            img_array = np.frombuffer(result.stdout, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Failed to decode screenshot. Image data may be corrupt or empty.")
                return None
            
            return img
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
        
    def click(self, x: int, y: int) -> bool:
        try:
            result = self._run_adb_command(f'shell input tap {x} {y}')
            if result.returncode != 0:
                raise RuntimeError(f"ADB click failed: {result.stderr.decode('utf-8', 'ignore')}")
            return True
        except Exception as e:
            print(f"Error clicking: {e}")
            return False
        
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
        try:
            result = self._run_adb_command(
                f"shell input swipe {x1} {y1} {x2} {y2} {duration}"
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Exception swiping: {e}")
            return False
        
    def test_connection(self) -> bool:
        """
        Test if ADB connection works
        
        Returns:
            True if connection works, False otherwise
        """
        try:
            result = self._run_adb_command("shell echo test")
            return result.returncode == 0
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def get_screen_size(self) -> Optional[Tuple[int, int]]:
        """
        Get screen dimensions
        
        Returns:
            (width, height) tuple, or None if failed
        """
        try:
            result = self._run_adb_command("shell wm size")
            if result.returncode == 0:
                output = result.stdout.decode('utf-8', errors='ignore')
                # Parse output like "Physical size: 1280x720"
                if 'x' in output:
                    parts = output.split('x')
                    if len(parts) == 2:
                        width = int(parts[0].strip().split()[-1])
                        height = int(parts[1].strip().split()[0])
                        return (width, height)
            return None
        except Exception as e:
            print(f"Failed to get screen size: {e}")
            return None