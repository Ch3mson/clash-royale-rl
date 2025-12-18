# clash-royale-rl

Reinforcement learning agent for Clash Royale using computer vision and ADB.

## Requirements

- Python 3.10+
- Android emulator with Clash Royale installed
- ADB access to the emulator

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Platform Setup

### Windows (BlueStacks)

1. Install [BlueStacks](https://www.bluestacks.com/)
2. Enable ADB in BlueStacks settings
3. Add BlueStacks to PATH:
   ```powershell
   $env:Path += ";C:\Program Files\BlueStacks_nxt"
   ```
4. Verify connection:
   ```bash
   HD-Adb devices
   ```
5. Update `adb_controller.py` to use BlueStacks ADB:
   ```python
   self.device_serial = f"emulator-{self.adb_port}"
   full_command = f'HD-Adb -s {self.device_serial} {command}'
   ```

### Mac (MuMu Player)

1. Install [MuMu Player Pro](https://www.mumuplayer.com/mac/) (native Apple Silicon support)
2. Enable ADB in MuMu settings
3. Find your ADB port:
   ```bash
   cd /tmp/com.netease.mumu.nemux-global && ./mumutool info all
   ```
4. Connect via ADB:
   ```bash
   adb connect 127.0.0.1:<port>
   ```
5. Update `adb_controller.py` to use standard ADB:
   ```python
   self.device_serial = f"127.0.0.1:{self.adb_port}"
   full_command = f'adb -s {self.device_serial} {command}'
   ```

#### Alternative Mac Emulators

- **Android Studio Emulator** - Free, but Clash Royale may crash due to emulator detection
- **Genymotion** - Free for personal use, good multi-instance support

## Calibration

Before running, calibrate the arena for your screen resolution:

```bash
python scripts/calibrate_arena.py --port <adb_port> --instance-id 0
```

This creates `config/instance_0.json` with:
- ADB port
- Screen dimensions
- Arena boundaries

## Usage

```bash
python main.py --instance 0 --games 1
```

## Configuration

### Config File (`config/instance_0.json`)

```json
{
  "instance_id": 0,
  "adb_port": 26625,
  "arena_config": {
    "screen_width": 900,
    "screen_height": 1600,
    "arena_top": 116,
    "arena_bottom": 1245,
    "arena_left": 70,
    "arena_right": 832
  }
}
```

### Resolution Recommendations

| Resolution | Screenshot Speed | Notes |
|------------|------------------|-------|
| 720x1280   | ~300ms | Fastest, recommended for training |
| 900x1600   | ~400ms | Good balance |
| 1080x1920  | ~600ms | High quality, slower |

## Troubleshooting

### ADB Connection Failed
- Verify emulator is running
- Check ADB port with emulator tools
- Try `adb connect 127.0.0.1:<port>`

### Button Detection Not Working
- Template images may need updating for your resolution
- Use `scripts/find_coordinates.py` to find correct positions
- Consider using fixed-position clicks instead of template matching

### Slow Screenshot Speed
- Lower emulator resolution
- Check ADB connection type (TCP vs local)

## Project Structure

```
clashroyalerl/
├── main.py                 # Main agent entry point
├── config/                 # Instance configurations
├── controllers/
│   ├── adb_controller.py   # ADB communication
│   └── game_controller.py  # Game actions
├── detection/
│   ├── state_detector.py   # Game state detection
│   ├── image_matcher.py    # Template matching
│   ├── grid_system.py      # Arena grid system
│   └── images/             # Template images
└── scripts/
    ├── calibrate_arena.py  # Arena calibration
    └── find_coordinates.py # Coordinate finder
```
