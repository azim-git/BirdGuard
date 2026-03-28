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
8. [Design Justifications](#8-design-justifications)
9. [Task Distribution](#9-task-distribution)

---

## 1. Hardware Components

| Component | Purpose |
|---|---|
| Raspberry Pi Zero 2 W | Main compute — runs inference, audio processing, MQTT |
| Raspberry Pi Pico W (on RoboPico board) | Microcontroller — drives servos and laser via PWM/GPIO |
| ReSpeaker 2-Mic Pi HAT | Dual-channel audio for bird sound detection and bearing estimation |
| USB Hub | Hub for all USB connections - Pi Zero 2 W, Pico W, USB Camera |
| USB Camera | Visual input for YOLO inference |
| Micro USB to USB 2.0 OTG Adapter | Adapter for Pi Zero 2's USB In to USB Hub |
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

### Pi Zero 2 W ↔ USB Hub

Connect the USB Hub to the USB In of the Pico W. This will be the main interface for all other USB components.

### USB Hub ← Pico W

Connect the Pico W (on the RoboPico board) to any USB port on the hub. The Pi communicates with the Pico over USB serial at 115200 baud. A udev symlink (`/dev/pico_turret`) is created during setup to give it a stable device path.

### USB Hub ← USB Camera

Connect the USB Camera to any USB port on the hub. A udev symlink (`/dev/birdguard_cam`) is created during setup to give it a stable device path.

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
  python3-pip python3-venv python3-numpy \
  python3-pyaudio alsa-utils python3-dev libasound2-dev git \
  mosquitto mosquitto-clients time linux-perf sysstat
```

### 3.4 Install Python 3.11 via pyenv (required for TFLite runtime)

The TFLite runtime requires Python 3.11, which is not available in the default Raspberry Pi OS repositories. Use **pyenv** to install and manage it.

#### 3.4.1 Install pyenv

```bash
curl https://pyenv.run | bash
```

#### 3.4.2 Configure shell environment

Add the following lines to the bottom of both `~/.bashrc` and `~/.profile`:

```bash
nano ~/.bashrc
```

```bash
nano ~/.profile
```

Add these lines to both files:

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"
```

Reload the shell:

```bash
source ~/.bashrc
```

#### 3.4.3 Install Python build dependencies

```bash
sudo apt install -y \
  libssl-dev libbz2-dev libreadline-dev \
  libffi-dev libncurses-dev
```

#### 3.4.4 Install Python 3.11.9

```bash
pyenv install 3.11.9
```

> This compiles Python from source and takes approximately 20 minutes on the Pi Zero 2 W.

Set it as the default:

```bash
pyenv global 3.11.9
```

Verify:

```bash
python --version
# Should show: Python 3.11.9
```

### 3.5 Install Python Packages

```bash
pip install tflite-runtime
pip install flask
pip install pyserial
pip install pyalsaaudio
pip install onnxruntime
pip install paho-mqtt
pip install "numpy<2"
pip install opencv-python-headless
pip install psutil
```

### 3.6 Stable Device Paths (udev Rules)

First, confirm the hardware attributes of your Pico and camera:

```bash
udevadm info -a -n /dev/ttyACM0 | grep -E 'idVendor|idProduct|serial' | head
udevadm info --query=all /dev/video0 | grep -E 'ID_VENDOR_ID|ID_MODEL_ID'
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

Add (replace vendor/product IDs with those confirmed above):

```
SUBSYSTEM=="video4linux", ATTR{index}=="0", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="5678", SYMLINK+="birdguard_cam", TAG+="systemd"
```

> **Important:** The `ATTR{index}=="0"` filter ensures the symlink points to the video capture node, not the metadata node. USB cameras register two `/dev/video` devices — only index 0 can capture frames.

Reload and verify:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger

ls -l /dev/pico_turret
ls -l /dev/birdguard_cam
```

### 3.7 ReSpeaker 2-Mic HAT Driver

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

### 3.8 Enable I2S in Firmware Config

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

### 3.9 Configure Audio Mixer

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

### 3.10 MQTT Broker Setup

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

### 3.11 Copy BirdGuard to the Pi

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

Run these steps on your **laptop / desktop** (not the Pi). You can perform these in a Colab Notebook.

### 5.1 Install Dependencies

```bash
!pip install ultralytics
```

### 5.2 Export the Model

```bash
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads automatically

model.export(
    format="tflite",
    imgsz=416,
    half=True,        # FP16
    int8=False,       # not INT8 — FP16 is better on Cortex-A53
    nms=False,        # keep postprocessing on the Pi side, same as current
)
```

This exports YOLOv8n to TFLite format at 416×416 input and quantises the weights to FP16 for faster inference on the Pi.

### 5.3 Transfer Model to Pi

```bash
scp yolov8n_float16.tflite azimpi@<pi-ip>:~/birdguard/
```

The default model path expected by BirdGuard is set in `model_path` in the config. Change to the correct filename if using a different model.

---

## 6. Running BirdGuard

All commands are run from inside the `birdguard/` directory on the Pi:

```bash
cd ~/birdguard
```

### Basic Run

```bash
python birdguard.py
```

Starts in **Smart Patrol** mode with no video stream overhead.

### With Live Stream

```bash
python birdguard.py --stream
```

Serves an MJPEG live feed at `http://<pi-ip>:5000`. The feed includes bounding box overlays, current pan/tilt angles, and the active state.

### Custom Stream Port

```bash
python birdguard.py --stream --port 8080
```

### Disable MQTT

```bash
python birdguard.py --no-mqtt
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

## 8. Design Justifications

This section documents the rationale behind every major hardware, software, model, and architectural decision in BirdGuard, and why each was chosen over the alternatives considered.

### 8.1 Hardware Justifications

#### 8.1.1 Raspberry Pi Zero 2 W — Main Compute Board

The Pi Zero 2 W was selected as the perception and decision node. Its quad-core Cortex-A53 at 1 GHz with 512 MB RAM provides sufficient compute for INT8/FP16 model inference while drawing only ~1.5 W. This low power envelope is critical for the target deployment scenarios (solar-powered field installations, rooftop arrays, airfield perimeters) where sustained power budgets of under 5 W are required.

The primary alternative considered was the Raspberry Pi 5 (quad Cortex-A76 @ 2.4 GHz, 4–8 GB RAM). While the Pi 5 would complete inference faster, the additional speed is unnecessary — the Pi Zero 2 already fits within the latency budget. The Pi 5 draws 5–10 W (3–7× more), has a larger 85×56 mm footprint that complicates turret integration, and costs 4–5× more. In edge system design, the correct platform is the one that meets requirements with the smallest resource footprint, not the most powerful available.

The 40-pin GPIO header enables direct stacking of the ReSpeaker HAT via I2S, eliminating the need for a USB audio device and keeping the audio path low-latency.

#### 8.1.2 Raspberry Pi Pico W on RoboPico Board — Actuator Controller

Servo PWM requires microsecond-level timing precision. Linux on the Pi Zero 2 is not a real-time operating system and cannot guarantee jitter-free GPIO — the kernel scheduler introduces unpredictable delays that cause servo jitter under CPU load. The Pico W running MicroPython on bare metal (RP2040, dual Cortex-M0+ @ 133 MHz) provides deterministic PWM output with command-to-movement latency typically under 10 ms.

The RoboPico carrier board was specifically chosen because it provides built-in servo headers and power regulation, eliminating external wiring, breadboarding, and a separate servo driver board. This reduces assembly complexity and potential failure points in an outdoor deployment.

The alternative considered was the Arduino Nano 33 IoT (SAMD21, single Cortex-M0+ @ 48 MHz). It is a capable MCU but offers a slower single core, no carrier board for easy servo integration, and costs ~$20 vs ~$6 for the Pico W. The RP2040's dual cores and higher clock also provide headroom for future firmware expansion.

Communication between the Pi Zero 2 and Pico W uses USB serial (CDC ACM class) at 115200 baud with a simple text protocol (`PAN<deg>,TILT<deg>,LASER<0|1>\n`). This is a local USB connection — not a network socket — so it is unaffected by Wi-Fi or internet connectivity. The Pico W has no autonomous behaviour; it is a deterministic actuator that executes exactly what it is told. If the serial link fails, the turret holds its last position and the Pi Zero 2 detects the failure.

#### 8.1.3 ReSpeaker 2-Mic Pi HAT v1.0 — Audio Input

The ReSpeaker 2-Mic HAT is the only option among those considered that enables direction-of-arrival estimation. Its two digital MEMS microphones in a fixed stereo array allow TDOA (Time Difference of Arrival) cross-correlation to produce a coarse left-right bearing estimate. While the angular resolution is limited by the small 58 mm inter-microphone spacing, a coarse estimate is sufficient — the camera refines the target position in the subsequent visual pipeline stage.

The alternative was a single USB condenser microphone. A mono microphone can detect the presence of sound but cannot estimate direction, leaving the camera to scan blindly. The HAT's I2S interface is also advantageous because it communicates over the 40-pin GPIO header and does not consume a USB port on the already-constrained single-port Pi Zero 2 W. The stacking form factor keeps the assembly compact and mechanically robust.

Audio processing is computationally trivial (short-time energy + GCC-PHAT cross-correlation), consuming negligible CPU, and acts as the system's wake trigger — the rest of the pipeline remains idle during quiet periods, saving power.

#### 8.1.4 USB Camera (HBVCAM W2312 V11)

A USB camera was chosen over the Pi Camera Module (CSI ribbon) for mechanical reasons: the camera is mounted on the moving turret assembly and must rotate with the pan/tilt mechanism. A USB cable can flex freely during turret movement, whereas a CSI ribbon cable is rigid and would be stressed by repeated rotation, eventually failing. The USB interface also allows the camera to be connected through the USB hub alongside the Pico W.

The camera captures at 1280×720 with a narrow 21.8° horizontal / 16.8° vertical field of view. The narrow FOV acts as a natural "zoom" that increases the pixel density on distant targets (2–10 m engagement range), improving detection of small birds without requiring higher resolution or larger models.

#### 8.1.5 Class 1 Laser Module

The laser provides a moving, unpredictable visual stimulus that birds perceive as a predator's eye or a threatening light pattern, without causing physical harm. Class 1 rating ensures eye safety for bystanders. The laser is digitally controlled via a single GPIO pin on the Pico W (active high) and draws current only when activated.

#### 8.1.6 USB Hub

The Pi Zero 2 W has only a single micro-USB OTG port. A USB hub is required to connect both the camera and the Pico W simultaneously. A micro-USB to USB 2.0 OTG adapter bridges the Pi's port to the hub. The hub also powers downstream devices from the shared 5V rail.

### 8.2 Software Justifications

#### 8.2.1 Python 3.11.9 via pyenv

The TFLite runtime requires Python 3.11.x. Raspberry Pi OS ships with Python 3.13, which is incompatible with the `tflite-runtime` package. Rather than downgrading the system Python (which could break OS packages), pyenv was used to install Python 3.11.9 alongside the system Python. This required compiling from source on the Pi Zero 2 W (~20 min), resolving tmpfs space constraints, and pinning NumPy to <2.0 to avoid ABI incompatibilities with the TFLite build.

#### 8.2.2 Threading Architecture (Pi Zero 2 W)

The pipeline on the Pi Zero 2 is divided across dedicated threads pinned to specific responsibilities:

- **AudioMonitor thread** — Continuously captures audio from the ReSpeaker via ALSA, computes short-time energy on each buffer, and performs TDOA cross-correlation when a wake event is detected. Always active during Smart Patrol, lightweight, acts as the system's "ears."
- **FrameGrabber thread** — Continuously reads frames from the USB camera in the background, keeping a single fresh frame available at all times. This eliminates the ~680 ms cost of draining 5 stale USB-buffered frames on every snapshot.
- **SnapshotWorker thread** — Waits for snapshot requests, retrieves the latest frame from FrameGrabber (near-instant), runs preprocessing + model inference + postprocessing, and posts results to the visual queue. This is the most CPU-intensive thread.
- **Decision/State Machine** — The main mode thread reads from shared queues, fuses audio bearings with visual detections, computes servo angles, and sends serial commands. Lightweight arithmetic + serial write.

Thread-safe communication uses `queue.Queue` (bounded, non-blocking) and `threading.Event` flags. A `turret_moving` event suppresses audio processing during servo movements to avoid self-triggering from servo noise.

#### 8.2.3 Multi-Mode Architecture

BirdGuard supports three operating modes, switchable at runtime via MQTT:

- **Smart Patrol** — Full AI pipeline: audio wake → visual inference → deterrent sweep. The original and primary mode.
- **Regular Patrol** — Continuous left-right sweep with laser on, no inference. Useful as a fallback when camera fails or as a simpler deterrent.
- **Manual** — User-controlled pan/tilt/laser via MQTT. Used for testing, calibration, and remote operation.

The ModeManager handles lifecycle (start/stop), health monitoring, and graceful degradation. If the camera fails during Smart Patrol, the system automatically falls back to Regular Patrol and emits a telemetry alert. Mode changes to Smart Patrol are blocked while the camera is unhealthy.

#### 8.2.4 MQTT for Configuration and Telemetry

MQTT (Mosquitto broker running locally on the Pi) was chosen for runtime configuration, mode switching, and telemetry because it provides a lightweight pub/sub interface that decouples the control plane from the detection pipeline. All tunable parameters (energy threshold, confidence threshold, deterrent cycles, patrol bounds, etc.) can be updated live without restarting the process, via JSON messages on `birdguard/config`. Config changes are persisted to disk (`~/.birdguard_config.json`) and survive restarts.

MQTT operates over the local Wi-Fi network only. The core detection-to-actuation loop has zero network dependency — MQTT is used only for optional telemetry and user control. Non-blocking publishes with timeouts ensure that network delays or disconnections never stall the real-time pipeline.

#### 8.2.5 MJPEG Web Stream (Flask)

An optional MJPEG stream (served by Flask on port 5000) provides a live camera feed with bounding box overlays, pan/tilt angles, and state information. This is useful for debugging and monitoring but is disabled by default (`--stream` flag) because encoding and serving frames adds CPU overhead. The stream runs in a separate daemon thread and does not affect the inference pipeline's latency.

#### 8.2.6 udev Symlinks for Stable Device Paths

USB devices can re-enumerate to different `/dev/videoN` or `/dev/ttyACMN` nodes across reboots or reconnects. Custom udev rules create stable symlinks (`/dev/birdguard_cam`, `/dev/pico_turret`) based on vendor/product IDs, ensuring the software always finds the correct device. The camera rule includes `ATTR{index}=="0"` to target the video capture node rather than the metadata node — USB cameras register two V4L2 devices, and only index 0 can capture frames.

#### 8.2.7 Health Monitoring and Auto-Recovery

The ModeManager runs periodic health checks (every 500 ms in the main loop) covering:

- **Camera health** — The FrameGrabber tracks consecutive read failures. After 30 failures (~3 seconds), the camera is marked unhealthy. In non-Smart-Patrol modes, an active grab probe detects disconnections. On failure, the system attempts to reopen the camera every 3 seconds, scanning all `/dev/videoN` nodes (since USB re-enumeration may assign a different device number).
- **Turret health** — The PicoTurret class tracks serial write failures. After two consecutive failures, the turret is marked unhealthy and telemetry alerts are emitted.

All health events are published to `birdguard/telemetry` so external monitoring systems can react.

#### 8.2.8 Graceful Degradation

The system never becomes entirely non-functional in any single-component failure scenario:

| Failure | Behaviour |
|---|---|
| Camera disconnected | Smart Patrol falls back to Regular Patrol automatically. User can switch back when camera recovers. |
| Audio subsystem fails | System continues with camera-only periodic scanning at reduced duty cycle. |
| Turret serial link lost | Turret holds last position. Pi detects failure and logs it. Actuation resumes on reconnect. |
| CPU thermal throttling | Visual inference is skipped; system degrades to audio-only aiming with sweep pattern. |

### 8.3 Model Justifications

#### 8.3.1 Model Selection: YOLOv8n (TFLite FP16) as Primary

Two model families were evaluated for the bird detection task, each tested in multiple formats and resolutions on the actual Pi Zero 2 W hardware:

| Model | Format | Input Size | Inference Time (Pi Zero 2) | File Size |
|---|---|---|---|---|
| YOLOv8n | TFLite FP16 | 416×416 | ~373 ms | ~6 MB |
| YOLOv8n | TFLite INT8 | 416×416 | ~550 ms | ~3 MB |
| YOLOv8s | ONNX FP32 | 320×320 | ~680 ms | ~22 MB |
| YOLOv8s | TFLite FP16 | 416×416 | ~1120 ms | ~21 MB |
| YOLOv8s | ONNX FP32 | 416×416 | ~1500 ms | ~43 MB |
| YOLOv8s | ONNX INT8 | 416×416 | ~1672 ms | ~11 MB |

**YOLOv8n TFLite FP16** was selected as the primary model. At ~373 ms per inference pass, it leaves sufficient headroom within the overall latency budget for audio processing, frame grab, postprocessing, and serial command transmission. YOLOv8n at ~3–12 MB fits comfortably in the Pi Zero 2's 512 MB RAM alongside the OS, audio threads, and camera buffers.

**YOLOv8s** (the "small" variant) was initially tested as it offers better detection accuracy (higher mAP on COCO), but its ~1500 ms inference time on the Pi Zero 2 via ONNX Runtime exceeds the latency budget on its own, before accounting for any other pipeline stage. Even at a reduced 320×320 input resolution (~680 ms), it consumes too much of the timing budget. YOLOv8s was retained as a tested alternative and its ONNX Runtime path remains in the codebase, but it is not used in the production configuration.

#### 8.3.2 Runtime Selection: TFLite over ONNX Runtime

Both TFLite Runtime and ONNX Runtime are supported in the codebase and were benchmarked:

- **TFLite Runtime** — Optimised for ARM devices, supports FP16 and INT8 quantisation natively, and produces consistently lower inference latency on the Pi Zero 2's Cortex-A53. Uses NHWC tensor layout (no transpose needed). Requires Python 3.11 (resolved via pyenv).
- **ONNX Runtime** — More portable and supports a wider range of models, but produces higher latency on the Pi Zero 2 for the same model architecture. Uses NCHW tensor layout (requires HWC→CHW transpose in preprocessing).

TFLite was chosen as the default runtime because it delivers ~3× faster inference for YOLOv8n on the target hardware. ONNX Runtime remains available as a fallback (controlled by `use_tflite` in config) for models that are not available in TFLite format.

#### 8.3.3 Input Resolution: 416×416

The model input resolution of 416×416 was chosen as a balance between detection accuracy and inference speed. Smaller inputs (256×256, 320×320) speed up inference but reduce the effective resolution for detecting small birds at distance. Larger inputs (640×640) improve accuracy but push inference time well beyond the latency budget on the Pi Zero 2.

At 416×416, the 1280×720 camera frame is letterboxed (resized with aspect ratio preservation and padded with value 114) to fit the square input. The letterbox parameters (scale factor, padding offsets) are tracked through postprocessing to correctly map bounding box coordinates back to the original frame.

#### 8.3.4 SAHI (Sliced Aided Hyper Inference)

For scenarios where birds appear very small in the frame (far distance or wide scene), BirdGuard supports SAHI-style tiled inference: the frame is split into a 2×2 grid of overlapping tiles, and inference is run on each tile independently. This effectively increases the resolution seen by the model at the cost of 4× the inference time.

SAHI is disabled by default (`sahi_enabled: false`) because the 4× latency cost (~1.5 s total) exceeds the target budget for typical use. It is configurable at runtime via MQTT and includes an early-exit optimisation — if a bird is detected on any tile, remaining tiles are skipped. Overlapping detections across tile boundaries are merged via NMS.

#### 8.3.5 Bird Class ID and Confidence Threshold

The model uses COCO class ID 14 ("bird"). The confidence threshold is set deliberately low at 0.10 to favour recall over precision — in a deterrent system, a false positive (startling an empty area) is a low-cost event, while a false negative (missing a real bird) defeats the system's purpose. NMS with an IoU threshold of 0.50 suppresses duplicate detections.

#### 8.3.6 Preprocessing Pipeline

The image preprocessing pipeline follows the standard YOLOv8 input contract:

1. **Letterbox resize** — Resize to 416×416 preserving aspect ratio, pad with value 114 (neutral grey).
2. **Float32 normalisation** — Scale pixel values from [0, 255] to [0.0, 1.0].
3. **Layout transpose** — For ONNX: HWC→CHW transpose + batch dimension (NCHW). For TFLite: batch dimension only (NHWC).
4. **Postprocessing coordinate unscaling** — Bounding box coordinates are converted from model input space back to original frame pixel space by reversing the letterbox padding and scale, then normalised to [0, 1] relative to the original frame dimensions.

An optional centre-crop mode (`centre_crop: true`) crops the 1280×720 frame to 720×720 before letterboxing. This removes the wide horizontal margins and better matches the model's square input aspect ratio, concentrating resolution on the centre of the scene.

### 8.4 Architectural Design Choices

#### 8.4.1 Two-Tier Edge Topology

BirdGuard follows a deliberate two-tier architecture: the Pi Zero 2 W serves as the perception and decision node (all sensing, processing, and decision-making), while the Pico W serves as a deterministic actuator node (PWM and GPIO only). This separation exists because:

1. **Real-time PWM** — Linux cannot guarantee microsecond-level PWM timing; the Pico W on bare metal can.
2. **Failure isolation** — If the Pico W hangs, the Pi can detect it via serial timeout. If the Pi crashes, the turret holds its last position safely.
3. **Simplicity** — The Pico firmware is a 50-line command parser. All intelligence lives in one place (the Pi).

#### 8.4.2 Audio-Visual Fusion Pipeline

The system uses a two-stage detection pipeline: audio first, then visual. Audio monitoring is computationally trivial and runs continuously, while visual inference is expensive and runs on-demand. The audio wake trigger acts as a supplementary to the visual pipeline, directing scans towards the direction of a potential acoustic event when detected, saving power and CPU cycles during quiet periods.

When both audio bearing and visual detection are available, the visual bounding box takes priority because it is more precise. If visual detection fails (bird outside FOV, inference miss), the system falls back to the audio bearing and commands a sweep of the estimated zone. This fusion approach ensures the system responds to events it can hear but not yet see.

#### 8.4.3 State Machine (Smart Patrol)

The Smart Patrol mode operates as a four-state machine:

- **IDLE** — Patrol sweep left-right-left, take snapshots at each waypoint, listen for audio wake events.
- **SCANNING** — Audio wake received; slew to estimated bearing, take snapshots across ± scan angle to visually confirm bird presence.
- **TRACKING** — Bird confirmed visually; run deterrent sweep, re-check with snapshots. If bird persists, repeat. If bird disappears for 2 consecutive misses, transition to SEARCHING.
- **SEARCHING** — Bird lost; scan the area around the last known position. If found, return to TRACKING. If timeout (16 s), return to IDLE.

This state machine ensures persistent deterrence (re-checking after sweeps) while avoiding infinite loops (timeout-based fallback to IDLE).

#### 8.4.4 Privacy-by-Design

All sensor data (camera frames, audio buffers) is processed entirely on-device. The only data that crosses a device boundary is the serial command to the Pico W (three numeric values: pan, tilt, laser state). If MQTT telemetry is enabled, only structured event metadata is transmitted (timestamp, event type, confidence score, pan/tilt angles) — never raw frames or audio. No raw imagery is stored to disk. This architecture ensures that bystanders captured by the camera are never exposed, meeting the privacy requirements for deployment in semi-public spaces.

#### 8.4.5 End-to-End Latency

The original design target was sub-500 ms end-to-end. Measured performance on the Pi Zero 2 W with YOLOv8n TFLite FP16 at 416×416 yields an end-to-end latency of approximately ~470 ms, which includes audio capture, TDOA computation, frame grab, inference (~373 ms), settle time, postprocessing, and serial command. The settle time is necessary to avoid motion blur from servo vibration.

---

## 9. Task Distribution

The table below shows each team member's contributions across all BirdGuard deliverables. All members contributed to integration testing, debugging, and the final demo.

### Abdul Azim Bin Mohd Said (2401980) — Team Lead / System Architect

| Area | Contributions |
|---|---|
| System Architecture | Designed the two-tier edge topology (Pi Zero 2 W + Pico W), defined the threading model, and established the audio-visual fusion pipeline. Architected the multi-mode framework (Smart Patrol, Regular Patrol, Manual) with the ModeManager lifecycle and health monitoring system. |
| Core Pipeline | Implemented the Smart Patrol state machine (IDLE → SCANNING → TRACKING → SEARCHING), the FrameGrabber background thread for eliminating stale-frame latency, decision fusion logic, and the deterrent sweep patterns (circular, Lissajous, random jitter). |
| Inference & Model Integration | Integrated both TFLite and ONNX Runtime backends, implemented the letterbox preprocessing pipeline, coordinate unscaling in postprocessing, and SAHI tiled inference with early-exit and NMS merging. Resolved the double-normalisation bug in `_postprocess` and the TFLite Python 3.11 compatibility issue via pyenv. |
| MQTT & Configuration | Built the full MQTT subsystem: config topic with runtime-tunable parameters, command topic with mode switching and restart logic, mode-specific command forwarding, telemetry publishing, and config persistence to disk. |
| Health & Reliability | Implemented camera health monitoring with auto-fallback, turret health tracking, USB camera reconnection with multi-strategy device scanning, and graceful degradation across all failure scenarios. |
| Profiling & Analysis | Wrote the `profile_smart_patrol.py` profiling suite with per-stage latency instrumentation, memory tracking, CSV export, matplotlib visualisations, and companion shell scripts for `cProfile`, `perf stat`, `perf record`, `taskset`, and `chrt`. |
| Audio Diagnostics | Developed `diagnose_audio.py` — a six-test diagnostic tool covering ALSA enumeration, mixer settings, raw capture verification, BirdGuard pipeline simulation, turret-moving gate checks, and continuous RMS monitoring. |
| Documentation | Authored the Design & Justification Report, the comprehensive README (setup guide, MQTT reference, design justifications), and the video presentation script. |

### Amiirulhasan Bin Jumali (2400886)

| Area | Contributions |
|---|---|
| Audio Pipeline | Implemented the AudioMonitor thread: ALSA capture via pyalsaaudio, RMS energy threshold detection, GCC-PHAT TDOA cross-correlation for bearing estimation, audio cooldown logic, and ALSA mixer initialisation at startup. Fixed the audio wake system bug where queue drain was mispositioned after settle sleep, and the `turret_moving` flag blocking the AudioMonitor thread during move windows. |
| Pico W Firmware | Wrote the MicroPython firmware (`main.py`) for the Pico W: serial command parser, PWM servo control with angle-to-duty conversion, and laser GPIO control. Defined the serial protocol format. |
| Hardware Assembly | Led physical assembly of the turret mechanism, wiring of servos and laser to the RoboPico board, and mounting of the laser on the turret. |
| Testing | Conducted audio pipeline testing with simulated bird sounds at known positions, measured TDOA accuracy, and tuned energy threshold and mixer gain values for the deployment environment. |

### Qusyairie Dani Bin Qamarul Huda (2400922)

| Area | Contributions |
|---|---|
| YOLO Model Pipeline | Handled model training, export, and quantisation: exported YOLOv8n and YOLOv8s from Ultralytics to ONNX format, performed INT8 quantisation, converted YOLOv8n to TFLite FP16, and benchmarked all model variants on the Pi Zero 2 W to establish the latency comparison data. |
| Camera Integration | Implemented the shared camera subsystem: `_find_working_camera()` with multi-strategy device scanning (symlink, resolved path, `/dev/video*` scan, index fallback), `open_shared_camera()` with retry logic, and `grab_camera_frame()` with configurable stale-frame draining. Set up udev rules for stable device paths. |
| Web Stream | Built the MJPEG streaming server using Flask, including the frame generator with placeholder rendering, the HTML viewer page, and integration with the `draw_overlay` function for live bounding box / state display. |
| Regular Patrol Mode | Implemented `mode_patrol.py`: continuous sweep loop with configurable speed, pan bounds, tilt angle, and laser state; parallel streaming thread; and MQTT mode command handling for live parameter updates. |

### Sim Yue Chong Samuel (2400695)

| Area | Contributions |
|---|---|
| Manual Mode | Implemented `mode_manual.py`: user-controlled pan/tilt/laser via MQTT commands, immediate (non-interpolated) movement for responsive control, continuous camera streaming, and the background deterrent sweep trigger (`manual_deter`). |
| Shared Infrastructure | Co-authored `shared.py`: the Config dataclass with all default values, `TUNABLE_FIELDS` set, config persistence loading, `ModeBase` abstract class with `start()`/`stop()`/`cmd()`/`smooth_move()`/`deterrent_sweep()`, and all shared state variables (queues, events, locks, position tracking). |
| ReSpeaker HAT Setup | Handled the ReSpeaker 2-Mic HAT driver installation (seeed-voicecard), I2S configuration in `/boot/firmware/config.txt`, and audio mixer calibration. Documented the full audio setup procedure. |
| Poster Design | Designed and produced the project poster for the final presentation, including system architecture diagrams, performance data visualisations, and the deployment scenario overview. |

### Ng Jing Xiang Edson (2400677)

| Area | Contributions |
|---|---|
| Pi Zero 2 W Setup | Led the initial Raspberry Pi Zero 2 W setup: OS flashing, SSH configuration, system package installation, Python 3.11 pyenv build (resolving tmpfs space and dependency issues), and pip package installation with NumPy <2.0 pinning. |
| Turret Communication | Implemented `pico_turret.py`: serial connection management with auto-reconnect, health tracking via `is_healthy` flag, command formatting and encoding, and two-attempt write-with-retry logic. |
| Network & MQTT Setup | Configured the Mosquitto MQTT broker on the Pi (systemd service, remote listener, anonymous access), verified local network connectivity, and tested MQTT round-trip with `mosquitto_pub`/`mosquitto_sub`. |
| Video Presentation | Scripted and coordinated the ~11-minute five-speaker video presentation, including section allocation, timing, and narrative flow across all team members. |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `arecord -l` shows no seeed card | Check `dtoverlay=wm8960-soundcard` is in `/boot/firmware/config.txt` and reboot |
| `/dev/pico_turret` missing | Verify the udev rule's `idVendor`/`idProduct` match the output of `udevadm info` for your Pico |
| `/dev/birdguard_cam` missing | Same as above for the camera. Ensure `ATTR{index}=="0"` is in the rule |
| `/dev/birdguard_cam` exists but camera fails | The symlink may point to the metadata node (video1) instead of the capture node (video0). Re-check the udev rule includes `ATTR{index}=="0"` |
| Turret spasms during patrol | `patrol_pan_min` is higher than `patrol_pan_max` — correct via MQTT `birdguard/mode` |
| Audio triggers too easily | Raise `energy_threshold` via MQTT (try 600–1000); servo movement noise can self-trigger at low values |
| `python --version` shows wrong version | Run `source ~/.bashrc` then `pyenv global 3.11.9` |
