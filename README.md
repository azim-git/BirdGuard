# BirdGuard

An edge-AI bird detection and deterrent system running on a Raspberry Pi Zero 2 W. BirdGuard uses a YOLO vision model and a dual-microphone HAT to detect and deter birds using a servo-controlled laser turret driven by a Raspberry Pi Pico W.

---

## Table of Contents

1. [Hardware Components](#1-hardware-components)
2. [Hardware Connections](#2-hardware-connections)
3. [Raspberry Pi Zero 2 W — Software Setup](#3-raspberry-pi-zero-2-w--software-setup)
4. [Raspberry Pi Pico W — Firmware Setup](#4-raspberry-pi-pico-w--firmware-setup)
5. [YOLO Model Setup](#5-yolo-model-setup)
6. [Running BirdGuard](#6-running-birdguard)
7. [MQTT Reference](#7-mqtt-reference)

---

## 1. Hardware Components

| Component | Purpose |
|---|---|
| Raspberry Pi Zero 2 W | Main compute — runs inference, audio processing, MQTT |
| Raspberry Pi Pico W (on RoboPico board) | Microcontroller — drives servos and laser via PWM/GPIO |
| ReSpeaker 2-Mic Pi HAT | Dual-channel audio for bird sound detection and bearing estimation |
| USB Camera | Visual input for YOLO inference |
| Pan servo | Horizontal turret rotation |
| Tilt servo | Vertical turret rotation |
| Laser module | Bird deterrent |

---

## 2. Hardware Connections

### Servo & Laser Wiring

Wire colours follow the standard servo convention:

| Wire Colour | Signal |
|---|---|
| Yellow | Signal (PWM / GPIO) |
| Red | Power (+) |
| Black / Brown | Ground (GND) |
| White | Unused |

### Pico W GPIO Assignments

| GPIO Pin | Component | Notes |
|---|---|---|
| GPIO 14 | Pan servo (PWM) | 50 Hz, 0–180° |
| GPIO 12 | Tilt servo (PWM) | 50 Hz, 0–180° |
| GPIO 28 | Laser module | Digital output, active high |

### Angle Reference

**Pan (horizontal, ↔)**

| Angle | Direction |
|---|---|
| 0° | Right |
| 75° | Centre (home) |
| 180° | Left |

**Tilt (vertical, ↕)**

| Angle | Direction |
|---|---|
| 0° | Straight up |
| 85° | Straight ahead (home) |
| 180° | Straight down |

> **Note:** Avoid driving tilt past ~150° — the camera cable can snag.

### Camera

| Property | Value |
|---|---|
| Interface | USB (udev symlink: `/dev/birdguard_cam`) |
| Horizontal FOV | 21.8° |
| Vertical FOV | 16.8° |
| Capture resolution | 1280 × 720 |

### ReSpeaker 2-Mic HAT

Stacks directly onto the Pi Zero 2 W 40-pin header. No additional wiring required.

### Pico W ↔ Pi Zero 2 W

Connect the Pico W (on the RoboPico board) to any USB port on the Pi Zero 2 W. The Pi communicates with the Pico over USB serial at 115200 baud. A udev symlink (`/dev/pico_turret`) is created during setup to give it a stable device path.

---

## 3. Raspberry Pi Zero 2 W — Software Setup

### 3.1 Flash the OS

Use **Raspberry Pi Imager** and select:

- OS: **Raspberry Pi OS (other) → Raspberry Pi OS Lite (64-bit)**

During flashing, pre-configure in the advanced settings:

| Setting | Value |
|---|---|
| Hostname | `edge` |
| Username | `azimpi` |
| Password | `pipi` |
| Wi-Fi SSID | `thethinker` |
| Wi-Fi Password | `ubyp1980` |

### 3.2 Initial SSH & System Update

```bash
ssh azimpi@<pi-ip-address>
# Password: pipi

sudo apt update && sudo apt upgrade -y
```

> This takes approximately 20 minutes.

### 3.3 Install System Packages

```bash
sudo apt install -y \
  python3-pip python3-venv python3-opencv python3-numpy \
  python3-pyaudio alsa-utils python3-dev libasound2-dev git \
  mosquitto mosquitto-clients time linux-perf sysstat
```

### 3.4 Install Python Packages

```bash
pip install flask --break-system-packages
pip install pyserial --break-system-packages
pip3 install pyalsaaudio --break-system-packages
pip install onnxruntime --break-system-packages
pip install paho-mqtt --break-system-packages
```

### 3.5 Stable Device Paths (udev Rules)

First, confirm the hardware attributes of your Pico and camera:

```bash
udevadm info -a -n /dev/ttyACM0 | grep -E 'idVendor|idProduct|serial' | head
udevadm info -a -n /dev/video0  | grep -E 'idVendor|idProduct|serial'
```

Create the udev rules files:

```bash
sudo nano /etc/udev/rules.d/99-pico-turret.rules
```

Add (replace values with those confirmed above):

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="0005", SYMLINK+="pico_turret"
```

```bash
sudo nano /etc/udev/rules.d/99-birdguard-cam.rules
```

Add (replace values with those confirmed above):

```
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="5678", SYMLINK+="birdguard_cam", TAG+="systemd"
```

Reload and verify:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger

ls -l /dev/pico_turret
ls -l /dev/birdguard_cam
```

### 3.6 ReSpeaker 2-Mic HAT Driver

**Stack the ReSpeaker HAT onto the Pi Zero 2 W before proceeding.**

```bash
git clone https://github.com/HinTak/seeed-voicecard
cd seeed-voicecard
sudo ./install.sh
sudo reboot
```

> This takes approximately 5 minutes and triggers a reboot.

After rebooting, SSH back in and verify the HAT is detected:

```bash
arecord -l
# Should list: seeed2micvoicec
```

### 3.7 Enable I2S in Firmware Config

```bash
sudo nano /boot/firmware/config.txt
```

At the end of the file, add:

```
dtoverlay=wm8960-soundcard
```

Also ensure this line is **uncommented**:

```
dtparam=i2s=on
```

```bash
sudo reboot
```

### 3.8 Configure Audio Mixer

After rebooting, SSH back in and run:

```bash
amixer -c 0 set 'Capture' 40
amixer -c 0 set 'ADC PCM' 200
amixer -c 0 set 'Left Input Boost Mixer LINPUT1' 1
amixer -c 0 set 'Right Input Boost Mixer RINPUT1' 1
amixer -c 0 set 'Left Input Mixer Boost' on
amixer -c 0 set 'Right Input Mixer Boost' on
amixer -c 0 set 'Left Boost Mixer LINPUT1' on
amixer -c 0 set 'Right Boost Mixer RINPUT1' on
```

Test the microphone:

```bash
arecord -D hw:0,0 -f S16_LE -r 44100 -c 2 -d 5 test.wav
aplay -D hw:0,0 test.wav
```

### 3.9 MQTT Broker Setup

Enable Mosquitto to start on boot:

```bash
sudo systemctl enable mosquitto
```

Allow external connections:

```bash
sudo nano /etc/mosquitto/conf.d/remote.conf
```

Add:

```
listener 1883
allow_anonymous true
```

Restart the broker:

```bash
sudo systemctl restart mosquitto
```

### 3.10 Copy BirdGuard to the Pi

From your laptop/desktop:

```bash
scp -r birdguard azimpi@<pi-ip>:~/
```

---

## 4. Raspberry Pi Pico W — Firmware Setup

### 4.1 Flash MicroPython

1. Hold the **BOOTSEL** button on the Pico W, connect it to your laptop via USB, then release BOOTSEL.
2. Open **Thonny**.
3. Go to **Tools → Install MicroPython** and follow the prompts.

### 4.2 Upload Firmware

Open `main.py` from the root of this repository in Thonny and save it to the Pico W (`File → Save as → Raspberry Pi Pico`).

The Pico will immediately begin listening for serial commands on USB at **115200 baud**.

### 4.3 Serial Command Protocol

The Pi sends newline-terminated commands in this format:

```
PAN<degrees>,TILT<degrees>,LASER<0|1>
```

Examples:

```
PAN90,TILT85,LASER1
PAN75,TILT85,LASER0
```

Fields are comma-separated, case-insensitive, and individually optional.

---

## 5. YOLO Model Setup

Run these steps on your **laptop / desktop** (not the Pi).

### 5.1 Install Dependencies

```bash
pip install ultralytics
pip install onnxruntime onnx onnxruntime-tools
```

### 5.2 Export & Quantise the Model

```bash
python3 models/create_yolo_model.py
```

This exports YOLOv8 to ONNX format at 416×416 input and quantises the weights to INT8 for faster inference on the Pi.

### 5.3 Transfer Model to Pi

```bash
scp models/yolov8s_int8.onnx azimpi@<pi-ip>:~/birdguard/
```

The default model path expected by BirdGuard is `yolov8s.onnx` inside the `birdguard/` directory. Update `model_path` in the config if using a different filename.

---

## 6. Running BirdGuard

All commands are run from inside the `birdguard/` directory on the Pi:

```bash
cd ~/birdguard
```

### Basic Run

```bash
python3 birdguard.py
```

Starts in **Smart Patrol** mode with no video stream overhead.

### With Live Stream

```bash
python3 birdguard.py --stream
```

Serves an MJPEG live feed at `http://<pi-ip>:5000`. The feed includes bounding box overlays, current pan/tilt angles, and the active state.

### Custom Stream Port

```bash
python3 birdguard.py --stream --port 8080
```

### Disable MQTT

```bash
python3 birdguard.py --no-mqtt
```

### Copy Files to Pi (from laptop)

```bash
scp -r birdguard azimpi@<pi-ip>:~/
```

### Test & Retrieve Audio Recording

```bash
# On the Pi:
arecord -D hw:0,0 -f S16_LE -r 44100 -c 2 -d 5 test.wav

# Copy back to your desktop to listen:
scp azimpi@<pi-ip>:~/test.wav .
```

---

## 7. MQTT Reference

The broker runs on the Pi at port **1883**. All messages are JSON unless noted.

### Topics Overview

| Topic | Direction | Purpose |
|---|---|---|
| `birdguard/config` | → Pi | Update tunable parameters |
| `birdguard/command` | → Pi | System commands and mode changes |
| `birdguard/mode` | → Pi | Mode-specific runtime commands |
| `birdguard/telemetry` | Pi → | Status, events, and acknowledgements |

---

### `birdguard/config` — Tune Parameters

Send a JSON object with any combination of the tunable fields below. Changes persist across restarts.

```bash
mosquitto_pub -h <pi-ip> -t birdguard/config -m '{"energy_threshold": 500}'
```

#### Smart Patrol Parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `energy_threshold` | float | `500.0` | RMS audio threshold. Lower = more sensitive. ~500 is a good midpoint; servo movement noise can self-trigger at lower values |
| `audio_cooldown` | float | `2.0` | Seconds between audio wake events |
| `confidence_threshold` | float | `0.10` | YOLO detection confidence. 0.10 is recommended (lenient is better) |
| `deter_cycles` | int | `5` | Number of deterrent sweep cycles per detection |
| `smart_patrol_tilt` | float | `85.0` | Tilt angle held during patrol. Takes effect on next angle change |
| `patrol_pan_steps` | int | `7` | Waypoints per 180° patrol sweep. Takes effect after the current sweep ends |
| `patrol_pan_min` | float | `10.0` | Minimum pan angle during patrol (degrees) |
| `patrol_pan_max` | float | `170.0` | Maximum pan angle during patrol (degrees) |
| `centre_crop` | bool | `false` | Crop 1280×720 → 720×720 before inference. Better matches the model's square input aspect ratio |
| `sahi_enabled` | bool | `false` | Split the frame into a 2×2 tile grid and run inference on each tile. Improves small-object detection at 4× the inference cost |
| `sahi_overlap_ratio` | float | `0.25` | Tile overlap ratio when SAHI is enabled |
| `scan_half_angle` | float | `20.0` | ±degrees scanned from an audio bearing |
| `deter_sweep_half` | float | `10.0` | ±pan degrees covered during a deterrent sweep |
| `deter_tilt_half` | float | `8.0` | ±tilt degrees covered during a deterrent sweep |
| `search_timeout` | float | `16.0` | Seconds to search for a bird before returning to patrol |
| `snapshot_settle` | float | `0.4` | Seconds to wait after a servo move before capturing a frame |

#### Regular Patrol Parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `regular_patrol_speed` | float | `15.0` | Pan speed in degrees/second |
| `regular_patrol_pan_min` | float | `10.0` | Sweep left bound (degrees) |
| `regular_patrol_pan_max` | float | `170.0` | Sweep right bound (degrees) |
| `regular_patrol_tilt` | float | `85.0` | Fixed tilt angle during patrol |
| `regular_patrol_laser` | bool | `true` | Laser on during regular patrol |

#### Safety

| Field | Type | Default | Description |
|---|---|---|---|
| `laser_enabled` | bool | `true` | Master laser enable/disable |

---

### `birdguard/command` — System Commands

#### Mode Changes (JSON)

```bash
mosquitto_pub -h <pi-ip> -t birdguard/command -m '{"change_mode": "smart_patrol"}'
mosquitto_pub -h <pi-ip> -t birdguard/command -m '{"change_mode": "patrol"}'
mosquitto_pub -h <pi-ip> -t birdguard/command -m '{"change_mode": "manual"}'
```

#### Plain-Text Commands

```bash
mosquitto_pub -h <pi-ip> -t birdguard/command -m 'get_status'
mosquitto_pub -h <pi-ip> -t birdguard/command -m 'get_config'
mosquitto_pub -h <pi-ip> -t birdguard/command -m 'get_mode'
mosquitto_pub -h <pi-ip> -t birdguard/command -m 'reset_config'
mosquitto_pub -h <pi-ip> -t birdguard/command -m 'restart'
mosquitto_pub -h <pi-ip> -t birdguard/command -m 'shutdown'
```

| Command | Effect |
|---|---|
| `get_status` | Publishes current state, pan/tilt, and mode to telemetry |
| `get_config` | Publishes full current config to telemetry |
| `get_mode` | Publishes active mode to telemetry |
| `reset_config` | Resets all tunable parameters to their defaults |
| `restart` | Gracefully restarts BirdGuard |
| `shutdown` | Stops BirdGuard cleanly |

#### Restart with Options (JSON)

```bash
mosquitto_pub -h <pi-ip> -t birdguard/command -m '{"restart": true}'
mosquitto_pub -h <pi-ip> -t birdguard/command -m '{"restart": true, "stream": true, "port": 8080}'
mosquitto_pub -h <pi-ip> -t birdguard/command -m '{"restart": true, "no_mqtt": true}'
```

---

### `birdguard/mode` — Mode-Specific Commands

All mode commands are JSON with a `"command"` key.

#### Smart Patrol Mode

Smart Patrol has no `birdguard/mode` commands. Use `birdguard/config` to tune its behaviour at runtime.

#### Regular Patrol Mode

```bash
mosquitto_pub -h <pi-ip> -t birdguard/mode -m '{"command": "patrol_speed",   "value": 20.0}'
mosquitto_pub -h <pi-ip> -t birdguard/mode -m '{"command": "patrol_laser",   "value": 1}'
mosquitto_pub -h <pi-ip> -t birdguard/mode -m '{"command": "patrol_tilt",    "value": 90.0}'
mosquitto_pub -h <pi-ip> -t birdguard/mode -m '{"command": "patrol_pan_min", "value": 20.0}'
mosquitto_pub -h <pi-ip> -t birdguard/mode -m '{"command": "patrol_pan_max", "value": 160.0}'
```

| Command | Value type | Description |
|---|---|---|
| `patrol_speed` | float (deg/s) | Pan sweep speed |
| `patrol_laser` | `0` or `1` | Toggle laser during sweep |
| `patrol_tilt` | float (°) | Fixed tilt angle. Takes effect on next move |
| `patrol_pan_min` | float (°) | Left sweep bound. Setting min > max causes spasms |
| `patrol_pan_max` | float (°) | Right sweep bound. Setting max < min causes spasms |

#### Manual Mode

```bash
# Move pan only
mosquitto_pub -h <pi-ip> -t birdguard/mode -m '{"command": "manual_pan",   "value": 90.0}'

# Move tilt only
mosquitto_pub -h <pi-ip> -t birdguard/mode -m '{"command": "manual_tilt",  "value": 85.0}'

# Toggle laser
mosquitto_pub -h <pi-ip> -t birdguard/mode -m '{"command": "manual_laser", "value": 1}'

# Move pan, tilt, and laser in one command
mosquitto_pub -h <pi-ip> -t birdguard/mode -m '{"command": "manual_move", "pan": 90, "tilt": 85, "laser": 1}'

# Trigger a deterrent sweep at the current position
mosquitto_pub -h <pi-ip> -t birdguard/mode -m '{"command": "manual_deter"}'
```

| Command | Description |
|---|---|
| `manual_pan` | Set pan to `value` degrees (0–180) |
| `manual_tilt` | Set tilt to `value` degrees (0–180) |
| `manual_laser` | `1` = laser on, `0` = laser off |
| `manual_move` | Set pan, tilt, and laser simultaneously |
| `manual_deter` | Run a deterrent sweep at the current pan/tilt |

---

### `birdguard/telemetry` — Outbound Events

Subscribe to receive events from BirdGuard:

```bash
mosquitto_sub -h <pi-ip> -t birdguard/telemetry
```

All messages are JSON and include `timestamp` (Unix epoch float) and `event` fields.

| Event | Triggered by | Key fields |
|---|---|---|
| `mode_changed` | Successful mode switch | `from_mode`, `to_mode` |
| `mode_change_rejected` | Mode change blocked (e.g., camera unavailable) | `reason` |
| `state_change` | Smart Patrol state machine transition | `from_state`, `to_state` |
| `detection` | Bird detected by YOLO | `confidence`, `pan`, `tilt` |
| `config_ack` | Config update applied | Updated field names and values |
| `config_state` | Response to `get_config` | Full config snapshot |
| `mode_status` | Response to `get_mode` | `active_mode`, `available_modes` |
| `status` | Response to `get_status` | `mode`, `state`, `pan`, `tilt`, `laser` |
| `health_alert` | Component failure or recovery | `component`, `status`, `detail` |
| `restarting` | Restart command received | — |
| `shutting_down` | Shutdown command received | — |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `arecord -l` shows no seeed card | Check `dtoverlay=wm8960-soundcard` is in `/boot/firmware/config.txt` and reboot |
| `/dev/pico_turret` missing | Verify the udev rule's `idVendor`/`idProduct` match the output of `udevadm info` for your Pico |
| `/dev/birdguard_cam` missing | Same as above for the camera |
| Turret spasms during patrol | `patrol_pan_min` is higher than `patrol_pan_max` — correct via MQTT `birdguard/mode` |
| Audio triggers too easily | Raise `energy_threshold` via MQTT (try 600–1000); servo movement noise can self-trigger at low values |
| Inference is too slow | Reduce `input_size` to 320/256 in config & use the appropriate yolo model with the same input dimensions|