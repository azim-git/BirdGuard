#!/usr/bin/env python3
"""
BirdGuard - Shared state, configuration, helpers, and mode base class.
=====================================================================
All modules import from here to avoid circular dependencies.
"""

import enum
import json
import logging
import math
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

log = logging.getLogger("birdguard")

# ========================== OPERATING MODES =================================

class Mode(enum.Enum):
    SMART_PATROL = "smart_patrol"
    PATROL       = "patrol"
    MANUAL       = "manual"

# ========================== STATE MACHINE (Smart Patrol only) ===============

class State(enum.Enum):
    IDLE      = "IDLE"
    SCANNING  = "SCANNING"
    TRACKING  = "TRACKING"
    SEARCHING = "SEARCHING"

# ========================== CONFIGURATION ===================================

@dataclass
class Config:
    # ---- Audio ----------------------------------------------------------
    audio_device: str = "plughw:0,0"
    sample_rate: int = 16000
    audio_channels: int = 2
    audio_chunk: int = 1024
    audio_cooldown: float = 1.0
    energy_threshold: float = 500.0
    mic_spacing_m: float = 0.058
    speed_of_sound: float = 343.0

    # ---- Camera ---------------------------------------------------------
    camera_index: int = 0
    camera_device: str = "/dev/birdguard_cam"
    cam_capture_w: int = 1280
    cam_capture_h: int = 720

    # ---- Model ----------------------------------------------------------
    model_path: str = "yolov8n_float16.tflite"
    use_tflite: bool = True
    confidence_threshold: float = 0.1
    nms_iou_threshold: float = 0.50
    bird_class_id: int = 14
    input_size: int = 416

    # ---- Centre-crop (Option 1) -----------------------------------------
    centre_crop: bool = False

    # ---- Sliced inference / SAHI (Option 2) -----------------------------
    sahi_enabled: bool = False
    sahi_overlap_ratio: float = 0.25
    sahi_slices_x: int = 2
    sahi_slices_y: int = 2
    sahi_merge_iou: float = 0.50

    # ---- Turret ---------------------------------------------------------
    pan_min: float = 0.0
    pan_max: float = 180.0
    tilt_min: float = 0.0
    tilt_max: float = 180.0
    pan_centre: float = 75.0
    tilt_centre: float = 85.0

    # ---- Camera FOV (measured HBVCAM W2312 V11) -------------------------
    cam_hfov_deg: float = 21.8
    cam_vfov_deg: float = 16.8

    # ---- Smart Patrol: idle patrol (sweeping left-right-left) -----------
    patrol_pan_min: float = 10.0
    patrol_pan_max: float = 170.0
    patrol_pan_steps: int = 7
    smart_patrol_tilt: float = 85.0       # default tilt for idle sweeps & scanning

    # ---- Smart Patrol: scanning -----------------------------------------
    scan_half_angle: float = 20.0
    scan_pan_steps: int = 2

    # ---- Smart Patrol: deterrent sweep ----------------------------------
    deter_sweep_half: float = 10.0
    deter_tilt_half: float = 8.0
    deter_sweep_steps: int = 8
    deter_sweep_dwell: float = 0.08
    deter_cycles: int = 5

    # ---- Smart Patrol: tracking -----------------------------------------
    track_recheck_max: int = 5

    # ---- Smart Patrol: searching ----------------------------------------
    search_half_angle: float = 25.0
    search_pan_steps: int = 2
    search_timeout: float = 16

    # ---- Snapshot settle (wait for servo vibration to stop) -------------
    snapshot_settle: float = 0.4

    # ---- Smooth servo movement ------------------------------------------
    smooth_move_steps: int = 10
    smooth_move_interval: float = 0.02

    # ---- Regular Patrol mode --------------------------------------------
    regular_patrol_speed: float = 15.0     # degrees per second
    regular_patrol_pan_min: float = 10.0
    regular_patrol_pan_max: float = 170.0
    regular_patrol_tilt: float = 85.0      # fixed tilt during patrol
    regular_patrol_laser: bool = True       # laser on during patrol

    # ---- Stream ---------------------------------------------------------
    stream_port: int = 5000

    # ---- MQTT -----------------------------------------------------------
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_config: str = "birdguard/config"
    mqtt_topic_telemetry: str = "birdguard/telemetry"
    mqtt_topic_command: str = "birdguard/command"
    mqtt_topic_mode_command: str = "birdguard/mode"
    mqtt_enabled: bool = True

    # ---- Safety ---------------------------------------------------------
    laser_enabled: bool = True


# Runtime-tunable fields (safe to change via MQTT without restart)
TUNABLE_FIELDS = {
    "energy_threshold", "audio_cooldown",
    "confidence_threshold",
    "centre_crop", "sahi_enabled", "sahi_overlap_ratio",
    "sahi_slices_x", "sahi_slices_y", "sahi_merge_iou",
    "patrol_pan_min", "patrol_pan_max", "patrol_pan_steps",
    "smart_patrol_tilt",
    "scan_half_angle", "scan_pan_steps",
    "deter_sweep_half", "deter_tilt_half", "deter_sweep_steps", "deter_sweep_dwell", "deter_cycles",
    "track_recheck_max",
    "search_half_angle", "search_pan_steps",
    "search_timeout",
    "snapshot_settle",
    "smooth_move_steps", "smooth_move_interval",
    "laser_enabled",
    "regular_patrol_speed", "regular_patrol_pan_min", "regular_patrol_pan_max",
    "regular_patrol_tilt", "regular_patrol_laser",
}

CONFIG_PERSIST_PATH = os.path.expanduser("~/.birdguard_config.json")

CFG = Config()

# Load persisted config overrides
if os.path.exists(CONFIG_PERSIST_PATH):
    try:
        with open(CONFIG_PERSIST_PATH, "r") as f:
            saved = json.load(f)
        for key, val in saved.items():
            if key in TUNABLE_FIELDS and hasattr(CFG, key):
                setattr(CFG, key, type(getattr(CFG, key))(val))
    except Exception:
        pass

# ========================== SHARED DATA =====================================

@dataclass
class AudioEvent:
    timestamp: float
    bearing_deg: float
    rms_energy: float

@dataclass
class VisualResult:
    timestamp: float
    detected: bool
    bbox: Optional[tuple] = None
    confidence: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    capture_pan: float = 0.0
    capture_tilt: float = 0.0
    inf_ms: float = 0.0
    dets: list = field(default_factory=list)
    frame: Optional[np.ndarray] = None

# Audio -> Decision (Smart Patrol only)
audio_queue: queue.Queue = queue.Queue(maxsize=4)

# Visual (background) -> Decision (Smart Patrol only)
visual_queue: queue.Queue = queue.Queue(maxsize=2)

# Decision -> Visual: request a snapshot (Smart Patrol only)
snapshot_request = threading.Event()

shutdown_flag = threading.Event()

# Stream globals
stream_enabled = False
latest_frame = None
frame_lock = threading.Lock()

# Current state (Smart Patrol states)
current_state = State.IDLE
state_lock = threading.Lock()

# Current operating mode
current_mode = Mode.SMART_PATROL
mode_lock = threading.Lock()

# Mode change request — set by MQTT, consumed by ModeManager
mode_change_request: Optional[Mode] = None
mode_change_lock = threading.Lock()

# Shared turret position
turret_pos_lock = threading.Lock()
turret_pos = {"pan": CFG.pan_centre, "tilt": CFG.tilt_centre}

# Servo noise suppression
turret_moving = threading.Event()

# Measured single-pass inference time
measured_pass_s = 4.0

# Shared camera instance (managed by ModeManager, used by modes)
shared_camera = None
shared_camera_lock = threading.Lock()

# MQTT manager reference (set by birdguard.py main, used by modes for telemetry)
mqtt_mgr = None

# ========================== HEALTH STATUS ===================================

# Camera health — set False by FrameGrabber when camera fails,
# checked by ModeManager to trigger fallback to Regular Patrol.
camera_healthy = True
camera_healthy_lock = threading.Lock()

# Turret health — set False by PicoTurret when serial fails persistently,
# checked by ModeManager to emit telemetry alerts.
turret_healthy = True
turret_healthy_lock = threading.Lock()

# ========================== HELPERS =========================================

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def letterbox(frame, size):
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def centre_crop_frame(frame):
    h, w = frame.shape[:2]
    if w <= h:
        return frame, 0
    crop_size = h
    x_offset = (w - crop_size) // 2
    cropped = frame[:, x_offset:x_offset + crop_size]
    return cropped, x_offset


def generate_sahi_slices(frame_h, frame_w, slices_x, slices_y, overlap_ratio):
    slices = []
    tile_w = frame_w / slices_x
    tile_h = frame_h / slices_y
    actual_tile_w = int(round(tile_w * (1.0 + overlap_ratio)))
    actual_tile_h = int(round(tile_h * (1.0 + overlap_ratio)))

    if slices_x == 1:
        xs = [0]
    else:
        xs = [int(round(i * (frame_w - actual_tile_w) / (slices_x - 1)))
              for i in range(slices_x)]
    if slices_y == 1:
        ys = [0]
    else:
        ys = [int(round(i * (frame_h - actual_tile_h) / (slices_y - 1)))
              for i in range(slices_y)]

    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + actual_tile_w, frame_w)
            y2 = min(y1 + actual_tile_h, frame_h)
            slices.append((x1, y1, x2, y2))
    return slices


def draw_overlay(frame, dets, best_idx, cur_pan, cur_tilt, inf_ms, state,
                 mode_label=None):
    """Draw detection overlay on a frame. Accepts optional mode_label string."""
    h, w = frame.shape[:2]
    display = frame.copy()

    cx, cy = w // 2, h // 2
    cv2.line(display, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 1)
    cv2.line(display, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 1)

    for i, det in enumerate(dets):
        bx = int(det["cx"] * w)
        by = int(det["cy"] * h)
        bw = int(det["w"] * w)
        bh = int(det["h"] * h)
        x1, y1 = bx - bw // 2, by - bh // 2
        x2, y2 = bx + bw // 2, by + bh // 2

        if i == best_idx:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"BIRD {det['conf']:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(display, (bx, by), 4, (0, 255, 0), -1)
        else:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 1)

    colors = {
        State.IDLE: (128, 128, 128),
        State.SCANNING: (0, 165, 255),
        State.TRACKING: (0, 255, 0),
        State.SEARCHING: (0, 0, 255),
    }
    color = colors.get(state, (255, 255, 255))

    # Top-left: state or mode label
    label = mode_label if mode_label else state.value
    cv2.putText(display, label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Bottom-left: turret info
    info_str = f"Pan:{int(cur_pan)} Tilt:{int(cur_tilt)}"
    if inf_ms > 0:
        info_str += f" Inf:{int(inf_ms)}ms"
    cv2.putText(display, info_str, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return display


def update_stream_frame(frame):
    """Thread-safe update of the latest frame for MJPEG streaming."""
    global latest_frame
    with frame_lock:
        latest_frame = frame


def grab_camera_frame(drain=True):
    """Grab a frame from the shared camera. Returns (success, frame).
    
    Args:
        drain: If True, discard stale buffered frames first (use for
               infrequent snapshots). If False, just read the next
               available frame (use for continuous streaming).
    """
    with shared_camera_lock:
        if shared_camera is None or not shared_camera.isOpened():
            return False, None
        if drain:
            # USB cameras on the Pi buffer 3-5 frames internally.
            # After a period of inactivity (e.g. mode switch), the
            # buffer may be full of stale frames with wrong exposure.
            # Drain aggressively to get a fresh frame.
            for _ in range(5):
                shared_camera.grab()
        ret, frame = shared_camera.read()
        return ret, frame


# ========================== MODE BASE CLASS =================================

class ModeBase(ABC):
    """Base class for all operating modes.

    Each mode runs its logic in a loop inside `run()`. The ModeManager
    calls `start()` and `stop()` to manage lifecycle.

    Subclasses must implement:
      - run(): main loop (runs in a dedicated thread)
      - on_mode_command(cmd, payload): handle mode-specific MQTT commands

    The `stop()` method sets `self._stop_event` — the run() loop should
    check `self._stop_event.is_set()` frequently to exit promptly.
    """

    def __init__(self, turret):
        self.turret = turret
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def start(self):
        """Start the mode in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_wrapper, name=self.name, daemon=True
        )
        self._thread.start()
        log.info("Mode started: %s", self.name)

    def stop(self):
        """Signal the mode to stop and wait for it to finish."""
        log.info("Stopping mode: %s", self.name)
        self._stop_event.set()
        # Also set snapshot_request so SnapshotWorker unblocks if waiting
        snapshot_request.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("Mode stopped: %s", self.name)

    def _run_wrapper(self):
        """Wrapper that catches exceptions so the thread doesn't die silently."""
        try:
            self.run()
        except Exception as exc:
            log.error("Mode %s crashed: %s", self.name, exc, exc_info=True)

    @abstractmethod
    def run(self):
        """Main mode loop. Check self._stop_event.is_set() frequently."""
        ...

    @abstractmethod
    def on_mode_command(self, cmd: str, payload: dict):
        """Handle mode-specific MQTT command. cmd is the command name,
        payload is the parsed JSON dict (may be empty)."""
        ...

    def should_stop(self) -> bool:
        """Convenience: check if mode or system shutdown requested."""
        return self._stop_event.is_set() or shutdown_flag.is_set()

    def cmd(self, pan, tilt, laser):
        """Send a command to the turret with safety checks."""
        if laser and not CFG.laser_enabled:
            laser = 0
        with turret_pos_lock:
            turret_pos["pan"] = pan
            turret_pos["tilt"] = tilt
        if self.turret:
            self.turret.send_command(pan=pan, tilt=tilt, laser=laser)

    def smooth_move(self, target_pan, target_tilt, laser=0):
        """Interpolate from current position to target in small steps."""
        with turret_pos_lock:
            start_pan = turret_pos["pan"]
            start_tilt = turret_pos["tilt"]

        steps = CFG.smooth_move_steps
        interval = CFG.smooth_move_interval

        turret_moving.set()
        for i in range(1, steps + 1):
            if self.should_stop():
                return
            t = i / steps
            pan = start_pan + (target_pan - start_pan) * t
            tilt = start_tilt + (target_tilt - start_tilt) * t
            self.cmd(int(round(pan)), int(round(tilt)), laser)
            time.sleep(interval)
        turret_moving.clear()

    def deterrent_sweep(self, centre_pan, centre_tilt):
        """Erratic circular laser sweep around a target position.
        Combines pan and tilt for unpredictable spiral motion.
        Uses CFG deter_* settings. Available to all modes."""
        import random

        pan_r = CFG.deter_sweep_half
        tilt_r = CFG.deter_tilt_half
        steps = CFG.deter_sweep_steps

        for cycle in range(CFG.deter_cycles):
            if self.should_stop():
                return
            pattern = cycle % 3

            if pattern == 0:
                # Circular sweep with random phase
                phase = random.uniform(0, math.pi)
                for i in range(steps):
                    if self.should_stop():
                        return
                    angle = phase + (2 * math.pi * i / steps)
                    pan = clamp(centre_pan + pan_r * math.cos(angle),
                                CFG.pan_min, CFG.pan_max)
                    tilt = clamp(centre_tilt + tilt_r * math.sin(angle),
                                 CFG.tilt_min, CFG.tilt_max)
                    self.cmd(int(round(pan)), int(round(tilt)), 1)
                    time.sleep(CFG.deter_sweep_dwell)
                    turret_moving.clear()

            elif pattern == 1:
                # Figure-8 / lissajous
                for i in range(steps):
                    if self.should_stop():
                        return
                    t = 2 * math.pi * i / steps
                    pan = clamp(centre_pan + pan_r * math.sin(t),
                                CFG.pan_min, CFG.pan_max)
                    tilt = clamp(centre_tilt + tilt_r * math.sin(2 * t),
                                 CFG.tilt_min, CFG.tilt_max)
                    self.cmd(int(round(pan)), int(round(tilt)), 1)
                    time.sleep(CFG.deter_sweep_dwell)
                    turret_moving.clear()

            else:
                # Random jitter
                for i in range(steps):
                    if self.should_stop():
                        return
                    pan = clamp(centre_pan + random.uniform(-pan_r, pan_r),
                                CFG.pan_min, CFG.pan_max)
                    tilt = clamp(centre_tilt + random.uniform(-tilt_r, tilt_r),
                                 CFG.tilt_min, CFG.tilt_max)
                    self.cmd(int(round(pan)), int(round(tilt)), 1)
                    time.sleep(CFG.deter_sweep_dwell)
                    turret_moving.clear()

        # Return to centre, laser off
        self.cmd(int(round(centre_pan)), int(round(centre_tilt)), 0)
        time.sleep(0.1)
        turret_moving.clear()