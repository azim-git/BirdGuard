#!/usr/bin/env python3
"""
BirdGuard - Real-Time Audio-Visual Bird Deterrent Pipeline (v6 - Multi-Mode)
=============================================================================
Runs on Raspberry Pi Zero 2 W.

v6 changes from v5:
  - Multi-mode architecture: Smart Patrol, Regular Patrol, Manual
  - Mode switching via MQTT change commands on birdguard/command
  - Mode-specific MQTT commands on birdguard/mode
  - Shared camera managed by ModeManager
  - Real-time MJPEG streaming in Patrol and Manual modes
  - Audio monitoring disabled in non-smart modes

Modes:
  smart_patrol : AI detection + deterrence (original behaviour)
  patrol       : Continuous sweep with laser, real-time stream
  manual       : User-controlled pan/tilt/laser, real-time stream

MQTT Topics:
  birdguard/command    — Change commands: {"change_mode": "patrol"}
                         Also: reset_config, get_config, status
  birdguard/mode       — Mode commands: {"command": "patrol_speed", "value": 20}
  birdguard/config     — Tunable config: {"energy_threshold": 600}
  birdguard/telemetry  — Outbound events (detection, state changes, mode changes)

Usage:
    python3 birdguard.py                  # boots into smart_patrol
    python3 birdguard.py --stream         # with MJPEG stream on port 5000
    python3 birdguard.py --stream --port 8080
    python3 birdguard.py --no-mqtt

Dependencies:
    numpy, opencv-python-headless, onnxruntime, pyserial, pyalsaaudio
    flask (only if --stream), paho-mqtt
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

from pico_turret import PicoTurret

from shared import (
    Mode, State, Config, CFG,
    TUNABLE_FIELDS, CONFIG_PERSIST_PATH,
    shutdown_flag, frame_lock, latest_frame, stream_enabled,
    state_lock, current_state, mode_lock, current_mode,
    mode_change_request, mode_change_lock,
    turret_pos_lock, turret_pos,
    shared_camera, shared_camera_lock,
    log,
)

from mode_smart_patrol import SmartPatrolMode
from mode_patrol import RegularPatrolMode
from mode_manual import ManualMode

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

# ========================== LOGGING =========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-14s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ========================== MODE MANAGER ====================================

class ModeManager:
    """Manages mode lifecycle: starts/stops modes, handles transitions,
    owns the shared camera and turret connection.

    Health monitoring:
      - Checks camera_healthy flag periodically
      - If camera fails during Smart Patrol, falls back to Regular Patrol
      - Blocks mode changes to Smart Patrol while camera is down
      - Checks turret health and emits telemetry alerts
    """

    def __init__(self, turret: Optional[PicoTurret]):
        self.turret = turret
        self._active_mode: Optional[object] = None
        self._active_mode_enum: Mode = Mode.SMART_PATROL
        self._lock = threading.Lock()
        self._camera_was_healthy = True
        self._turret_was_healthy = True
        self._last_cam_reopen = 0.0

    @property
    def active_mode(self) -> Mode:
        return self._active_mode_enum

    def start_mode(self, mode: Mode):
        """Start a new mode (stops the current one first if running)."""
        with self._lock:
            # Stop current mode
            if self._active_mode is not None:
                log.info("Stopping current mode: %s", self._active_mode_enum.value)
                self._active_mode.stop()
                self._active_mode = None

            # Update global mode
            self._active_mode_enum = mode
            with mode_lock:
                import shared
                shared.current_mode = mode

            # Create and start the new mode
            if mode == Mode.SMART_PATROL:
                self._active_mode = SmartPatrolMode(self.turret)
            elif mode == Mode.PATROL:
                self._active_mode = RegularPatrolMode(self.turret)
            elif mode == Mode.MANUAL:
                self._active_mode = ManualMode(self.turret)
            else:
                log.error("Unknown mode: %s", mode)
                return

            log.info("Starting mode: %s", mode.value)
            self._active_mode.start()

    def stop(self):
        """Stop the current mode (for shutdown)."""
        with self._lock:
            if self._active_mode is not None:
                self._active_mode.stop()
                self._active_mode = None

    def change_mode(self, new_mode: Mode):
        """Request a mode change. Blocks Smart Patrol if camera is down."""
        import shared

        if new_mode == self._active_mode_enum:
            log.info("Already in mode %s, ignoring change request", new_mode.value)
            return

        # Gate: reject Smart Patrol if camera is unhealthy
        if new_mode == Mode.SMART_PATROL:
            with shared.camera_healthy_lock:
                cam_ok = shared.camera_healthy
            if not cam_ok:
                log.warning("Mode change to smart_patrol REJECTED — camera is "
                            "unavailable. Staying in %s.",
                            self._active_mode_enum.value)
                if shared.mqtt_mgr:
                    shared.mqtt_mgr.publish_telemetry(
                        "mode_change_rejected",
                        requested="smart_patrol",
                        reason="camera_unavailable",
                        current_mode=self._active_mode_enum.value,
                    )
                return

        log.info("Mode change requested: %s -> %s",
                 self._active_mode_enum.value, new_mode.value)
        self.start_mode(new_mode)

    def check_health(self):
        """Called periodically from the main loop. Handles degradation.

        Camera failure:
          Smart Patrol → falls back to Regular Patrol
          Emits telemetry alert

        Camera recovery:
          Emits telemetry (does NOT auto-switch back — user decides)

        Turret failure:
          Emits telemetry alert (no mode change — nothing useful to switch to)

        Turret recovery:
          Emits telemetry
        """
        import shared

        # ---- Camera health ----
        with shared.camera_healthy_lock:
            cam_ok = shared.camera_healthy

        # Active probe: in Patrol/Manual modes FrameGrabber doesn't run,
        # so camera_healthy is never updated.  Probe the VideoCapture
        # directly to detect disconnects.
        if cam_ok and self._active_mode_enum != Mode.SMART_PATROL:
            with shared_camera_lock:
                if shared.shared_camera is None or not shared.shared_camera.isOpened():
                    cam_ok = False
                    with shared.camera_healthy_lock:
                        shared.camera_healthy = False
                    log.error("HEALTH: Camera lost (device closed/missing)")
                else:
                    # Quick non-blocking grab test — if the device handle
                    # is open but USB is gone, grab() returns False
                    ok = shared.shared_camera.grab()
                    if not ok:
                        self._cam_probe_fails = getattr(
                            self, '_cam_probe_fails', 0) + 1
                        if self._cam_probe_fails >= 3:
                            cam_ok = False
                            with shared.camera_healthy_lock:
                                shared.camera_healthy = False
                            log.error("HEALTH: Camera lost (grab failed "
                                      "%d times)", self._cam_probe_fails)
                    else:
                        self._cam_probe_fails = 0

        if not cam_ok and self._camera_was_healthy:
            # Camera just went down
            log.error("HEALTH: Camera lost")
            if shared.mqtt_mgr:
                shared.mqtt_mgr.publish_telemetry(
                    "health_alert",
                    component="camera",
                    status="disconnected",
                    action="fallback_to_patrol"
                    if self._active_mode_enum == Mode.SMART_PATROL
                    else "no_action",
                )
            # Fall back from Smart Patrol to Regular Patrol
            if self._active_mode_enum == Mode.SMART_PATROL:
                log.warning("HEALTH: Falling back from Smart Patrol to "
                            "Regular Patrol (camera unavailable)")
                self.start_mode(Mode.PATROL)
            self._camera_was_healthy = False

        elif not cam_ok and not self._camera_was_healthy:
            # Camera still down — periodically try to reopen it.
            # Runs regardless of which mode is active.
            now = time.perf_counter()
            if now - self._last_cam_reopen >= 3.0:
                self._last_cam_reopen = now
                self._try_reopen_camera()

        elif cam_ok and not self._camera_was_healthy:
            # Camera recovered
            log.info("HEALTH: Camera recovered")
            if shared.mqtt_mgr:
                shared.mqtt_mgr.publish_telemetry(
                    "health_alert",
                    component="camera",
                    status="reconnected",
                    hint="smart_patrol now available — send change_mode "
                         "to switch back",
                )
            self._camera_was_healthy = True

        # ---- Turret health ----
        turret_ok = self.turret.is_healthy if self.turret else False

        if not turret_ok and self._turret_was_healthy:
            log.error("HEALTH: Turret (Pico W) connection lost — "
                      "actuation disabled")
            if shared.mqtt_mgr:
                shared.mqtt_mgr.publish_telemetry(
                    "health_alert",
                    component="turret",
                    status="disconnected",
                    impact="actuation_disabled",
                )
            self._turret_was_healthy = False

        elif turret_ok and not self._turret_was_healthy:
            log.info("HEALTH: Turret (Pico W) reconnected")
            if shared.mqtt_mgr:
                shared.mqtt_mgr.publish_telemetry(
                    "health_alert",
                    component="turret",
                    status="reconnected",
                )
            self._turret_was_healthy = True

    def _try_reopen_camera(self):
        """Attempt to reopen the camera after USB disconnect/reconnect.

        Tries multiple strategies since USB cameras can re-enumerate to
        different /dev/videoN nodes, and udev symlinks may point to the
        wrong node (metadata vs capture interface).
        """
        import shared
        log.info("HEALTH: Attempting camera reopen")
        with shared_camera_lock:
            if shared.shared_camera is not None:
                try:
                    shared.shared_camera.release()
                except Exception:
                    pass
                shared.shared_camera = None

            cap = _find_working_camera()
            if cap is not None:
                shared.shared_camera = cap
                with shared.camera_healthy_lock:
                    shared.camera_healthy = True
                log.info("HEALTH: Camera reopened successfully")
            else:
                log.warning("HEALTH: No working camera found")

    def forward_mode_command(self, cmd: str, payload: dict):
        """Forward a mode-specific command to the active mode."""
        with self._lock:
            if self._active_mode is not None:
                self._active_mode.on_mode_command(cmd, payload)
            else:
                log.warning("No active mode to forward command '%s'", cmd)


# Global mode manager instance
mode_mgr: Optional[ModeManager] = None

# ========================== MQTT MANAGER ====================================

class MQTTManager:
    """Handles MQTT communications: config updates, commands, mode changes,
    and telemetry publishing.

    Subscribes to:
      birdguard/config   — JSON with tunable fields to update
      birdguard/command  — Change commands + system commands
      birdguard/mode     — Mode-specific commands

    Publishes to:
      birdguard/telemetry — detection events, state changes, mode changes
    """

    def __init__(self):
        self.client = None

    def start(self):
        if mqtt is None:
            log.warning("paho-mqtt not installed - MQTT disabled")
            return
        if not CFG.mqtt_enabled:
            log.info("MQTT disabled in config")
            return

        try:
            self.client = mqtt.Client(client_id="birdguard", clean_session=True)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect_async(CFG.mqtt_broker, CFG.mqtt_port, keepalive=60)
            self.client.loop_start()
            log.info("MQTT connecting to %s:%d", CFG.mqtt_broker, CFG.mqtt_port)
        except Exception as exc:
            log.error("MQTT start failed: %s", exc)
            self.client = None

    def stop(self):
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass

    def publish_telemetry(self, event_type: str, **kwargs):
        if not self.client:
            return
        payload = {
            "timestamp": time.time(),
            "event": event_type,
            **kwargs,
        }
        try:
            self.client.publish(
                CFG.mqtt_topic_telemetry, json.dumps(payload), qos=0)
        except Exception:
            pass

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            log.info("MQTT connected")
            client.subscribe(CFG.mqtt_topic_config, qos=1)
            client.subscribe(CFG.mqtt_topic_command, qos=1)
            client.subscribe(CFG.mqtt_topic_mode_command, qos=1)
            self._publish_current_config()
            self._publish_mode_status()
        else:
            log.error("MQTT connect failed, rc=%d", rc)

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = msg.payload.decode("utf-8")
        except Exception:
            return

        if topic == CFG.mqtt_topic_config:
            self._handle_config(payload)
        elif topic == CFG.mqtt_topic_command:
            self._handle_command(payload)
        elif topic == CFG.mqtt_topic_mode_command:
            self._handle_mode_command(payload)

    # ---- Config topic ----

    def _handle_config(self, payload):
        try:
            updates = json.loads(payload)
        except json.JSONDecodeError as exc:
            log.error("MQTT config: bad JSON: %s", exc)
            return

        if not isinstance(updates, dict):
            log.error("MQTT config: expected JSON object")
            return

        applied = {}
        rejected = {}

        for key, val in updates.items():
            if key not in TUNABLE_FIELDS:
                rejected[key] = "not tunable at runtime"
                continue
            if not hasattr(CFG, key):
                rejected[key] = "unknown field"
                continue
            try:
                expected_type = type(getattr(CFG, key))
                setattr(CFG, key, expected_type(val))
                applied[key] = getattr(CFG, key)
            except (ValueError, TypeError) as exc:
                rejected[key] = str(exc)

        if applied:
            log.info("MQTT config applied: %s", applied)
            self._persist_config()
        if rejected:
            log.warning("MQTT config rejected: %s", rejected)

        self.publish_telemetry("config_ack", applied=applied, rejected=rejected)
        self._publish_current_config()

    # ---- Command topic (change commands + system commands) ----

    def _handle_command(self, payload):
        """Handle change commands and system commands.

        Change commands (JSON):
          {"change_mode": "patrol"}
          {"change_mode": "smart_patrol"}
          {"change_mode": "manual"}
          {"restart": true}
          {"restart": true, "stream": true}
          {"restart": true, "stream": false}

        System commands (plain text):
          reset_config
          get_config
          status
        """
        # Try JSON first (for change commands)
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                if "change_mode" in data:
                    self._handle_change_mode(data["change_mode"])
                    return
                if "restart" in data and data["restart"]:
                    self._handle_restart(data)
                    return
        except json.JSONDecodeError:
            pass

        # Plain text system commands
        cmd = payload.strip().lower()

        if cmd == "reset_config":
            defaults = Config()
            for key in TUNABLE_FIELDS:
                if hasattr(defaults, key):
                    setattr(CFG, key, getattr(defaults, key))
            self._persist_config()
            log.info("MQTT: config reset to defaults")
            self.publish_telemetry("config_reset")
            self._publish_current_config()

        elif cmd == "get_config":
            self._publish_current_config()

        elif cmd == "status":
            with state_lock:
                st = current_state.value
            with turret_pos_lock:
                pan = turret_pos["pan"]
                tilt = turret_pos["tilt"]
            active_mode = mode_mgr.active_mode.value if mode_mgr else "unknown"
            self.publish_telemetry(
                "status", state=st, pan=pan, tilt=tilt, mode=active_mode)

        elif cmd == "get_mode":
            self._publish_mode_status()

        elif cmd == "restart":
            self._handle_restart({})

        elif cmd == "shutdown":
            log.info("MQTT: shutdown requested")
            self.publish_telemetry("shutting_down")
            time.sleep(0.5)
            shutdown_flag.set()

        else:
            log.warning("MQTT: unknown command '%s'", cmd)

    def _handle_restart(self, data: dict):
        """Restart the BirdGuard process.
        
        Optional fields in data:
          stream: bool — enable/disable stream on restart (default: keep current)
          port: int — stream port (default: keep current)
          no_mqtt: bool — disable MQTT on restart (default: false)
        """
        import shared

        # Build only the CLI flags (not executable or script path)
        flags = []

        # Stream: use explicit value if provided, otherwise keep current
        if "stream" in data:
            if data["stream"]:
                flags.append("--stream")
        elif shared.stream_enabled:
            flags.append("--stream")

        # Port
        if "port" in data:
            flags.extend(["--port", str(int(data["port"]))])

        # MQTT
        if data.get("no_mqtt", False):
            flags.extend(["--no-mqtt"])

        log.info("MQTT: restart requested with flags: %s", flags)
        self.publish_telemetry("restarting", flags=flags)

        # Give MQTT time to publish the telemetry message
        time.sleep(0.5)

        # Write restart flags to a temp file
        restart_marker = os.path.expanduser("~/.birdguard_restart")
        try:
            with open(restart_marker, "w") as f:
                json.dump({"flags": flags, "timestamp": time.time()}, f)
        except Exception:
            pass

        # Trigger clean shutdown, then re-exec
        shutdown_flag.set()

    def _handle_change_mode(self, mode_str: str):
        """Process a mode change command."""
        mode_str = mode_str.strip().lower()
        mode_map = {
            "smart_patrol": Mode.SMART_PATROL,
            "patrol": Mode.PATROL,
            "manual": Mode.MANUAL,
        }

        if mode_str not in mode_map:
            log.warning("MQTT: unknown mode '%s' (valid: %s)",
                        mode_str, ", ".join(mode_map.keys()))
            self.publish_telemetry("mode_change_rejected",
                                   requested=mode_str,
                                   reason="unknown mode")
            return

        new_mode = mode_map[mode_str]
        old_mode = mode_mgr.active_mode.value if mode_mgr else "unknown"

        if mode_mgr:
            mode_mgr.change_mode(new_mode)
            self.publish_telemetry("mode_changed",
                                   from_mode=old_mode,
                                   to_mode=new_mode.value)
            self._publish_mode_status()
        else:
            log.error("ModeManager not initialised")

    # ---- Mode command topic ----

    def _handle_mode_command(self, payload):
        """Forward mode-specific commands to the active mode.

        Expected JSON: {"command": "patrol_speed", "value": 20.0}
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            log.error("MQTT mode command: bad JSON: %s", exc)
            return

        if not isinstance(data, dict) or "command" not in data:
            log.error("MQTT mode command: expected {\"command\": \"...\", ...}")
            return

        cmd = data["command"]
        if mode_mgr:
            mode_mgr.forward_mode_command(cmd, data)
        else:
            log.error("ModeManager not initialised")

    # ---- Persistence ----

    def _persist_config(self):
        data = {}
        for key in TUNABLE_FIELDS:
            if hasattr(CFG, key):
                data[key] = getattr(CFG, key)
        try:
            with open(CONFIG_PERSIST_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            log.error("Config persist failed: %s", exc)

    def _publish_current_config(self):
        data = {}
        for key in TUNABLE_FIELDS:
            if hasattr(CFG, key):
                data[key] = getattr(CFG, key)
        self.publish_telemetry("config_state", config=data)

    def _publish_mode_status(self):
        active = mode_mgr.active_mode.value if mode_mgr else "unknown"
        self.publish_telemetry("mode_status", active_mode=active,
                               available_modes=["smart_patrol", "patrol", "manual"])


# Global MQTT manager instance
mqtt_mgr: Optional[MQTTManager] = None

# ========================== SHARED CAMERA ===================================

def _find_working_camera():
    """Find and open the first working USB camera.

    USB cameras register two V4L2 nodes: one for video capture (the one
    we need) and one for metadata (which can't capture frames). The udev
    symlink may point to either one. Additionally, after USB reconnect
    the device may re-enumerate to a different /dev/videoN.

    Strategy: try the configured device first, then fall back to scanning
    all /dev/videoN nodes for a USB camera that can actually produce frames.
    Skip bcm2835 (Pi GPU) devices.

    Returns an opened, verified cv2.VideoCapture or None.
    """
    candidates = []

    # Priority 1: configured device (symlink or path)
    if os.path.exists(CFG.camera_device):
        real = os.path.realpath(CFG.camera_device)
        candidates.append(("config", real))

    # Priority 2: resolved symlink target (in case OpenCV can't follow symlinks)
    if os.path.islink(CFG.camera_device):
        real = os.path.realpath(CFG.camera_device)
        if ("config", real) not in candidates:
            candidates.append(("symlink-resolved", real))

    # Priority 3: scan /dev/video* for USB cameras (skip Pi GPU devices)
    try:
        video_devs = sorted(
            [f"/dev/{f}" for f in os.listdir("/dev") if f.startswith("video")],
            key=lambda p: int(p.replace("/dev/video", "")) if p.replace("/dev/video", "").isdigit() else 999
        )
    except OSError:
        video_devs = []

    for dev_path in video_devs:
        # Quick filter: check if this is a USB camera via v4l2 sysfs
        # bcm2835 devices have index numbers >= 10 on Pi
        dev_name = os.path.basename(dev_path)
        dev_num = dev_name.replace("video", "")
        if not dev_num.isdigit():
            continue
        num = int(dev_num)
        # bcm2835 devices are typically video10+; USB cameras are video0-video9
        if num >= 10:
            continue
        label = f"scan-{dev_path}"
        if not any(c[1] == dev_path for c in candidates):
            candidates.append((label, dev_path))

    # Priority 4: integer index 0 (OpenCV's own device enumeration)
    candidates.append(("index-0", 0))

    for label, device in candidates:
        try:
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.cam_capture_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_capture_h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Drain stale buffer
            for _ in range(5):
                cap.grab()

            # Verify actual frame capture
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                log.info("Camera found via %s (%s) — %dx%d",
                         label, device, test_frame.shape[1], test_frame.shape[0])
                return cap
            else:
                cap.release()
        except Exception:
            try:
                cap.release()
            except Exception:
                pass

    return None


def open_shared_camera():
    """Open the shared camera instance used by all modes.

    Retries at startup because after reboot the USB camera may appear
    in /dev before its firmware is ready to stream.
    """
    import shared
    max_attempts = 5

    with shared_camera_lock:
        if shared.shared_camera is not None and shared.shared_camera.isOpened():
            return True

    for attempt in range(1, max_attempts + 1):
        with shared_camera_lock:
            cap = _find_working_camera()
            if cap is not None:
                shared.shared_camera = cap
                return True

        log.warning("Camera open attempt %d/%d failed", attempt, max_attempts)
        if attempt < max_attempts:
            time.sleep(2.0)

    log.error("Failed to open camera after %d attempts", max_attempts)
    with shared.camera_healthy_lock:
        shared.camera_healthy = False
    return False


def close_shared_camera():
    import shared
    with shared_camera_lock:
        if shared.shared_camera is not None:
            shared.shared_camera.release()
            shared.shared_camera = None
            log.info("Shared camera closed")

# ========================== WEB STREAM ======================================

def start_stream(port):
    try:
        from flask import Flask, Response
    except ImportError:
        log.error("Flask not installed - stream disabled")
        return

    app = Flask(__name__)

    def generate():
        while True:
            with frame_lock:
                import shared as _sh
                if _sh.latest_frame is None:
                    frame = None
                else:
                    frame = _sh.latest_frame.copy()

            if frame is None:
                # No frame yet — generate a placeholder so the browser
                # doesn't hang waiting for the first MJPEG boundary
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for camera...",
                            (120, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder,
                                       [cv2.IMWRITE_JPEG_QUALITY, 50])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)

    @app.route('/')
    def index():
        return '''<html><head><title>BirdGuard</title>
        <style>
            body { background: #1a1a1a; color: #fff; font-family: monospace;
                   display: flex; flex-direction: column; align-items: center; padding: 20px; }
            h2 { color: #4CAF50; }
            img { border: 2px solid #333; max-width: 100%; }
        </style>
        </head><body>
        <h2>BirdGuard v6 - Multi-Mode</h2>
        <img src="/stream">
        </body></html>'''

    @app.route('/stream')
    def stream_route():
        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    flask_log = logging.getLogger('werkzeug')
    flask_log.setLevel(logging.WARNING)

    # Allow port reuse so restarts don't fail with "Address already in use"
    from werkzeug.serving import make_server
    server = make_server('0.0.0.0', port, app, threaded=True)
    server.socket.setsockopt(
        __import__('socket').SOL_SOCKET,
        __import__('socket').SO_REUSEADDR, 1
    )
    server.serve_forever()

# ========================== MAIN ============================================

def main():
    global mqtt_mgr, mode_mgr

    import shared
    parser = argparse.ArgumentParser(
        description="BirdGuard v6 - Multi-Mode Bird Deterrent")
    parser.add_argument('--stream', action='store_true',
                        help='Enable live MJPEG camera stream')
    parser.add_argument('--port', type=int, default=CFG.stream_port,
                        help=f'Stream port (default: {CFG.stream_port})')
    parser.add_argument('--no-mqtt', action='store_true',
                        help='Disable MQTT even if paho-mqtt is installed')
    args = parser.parse_args()
    shared.stream_enabled = args.stream

    if args.no_mqtt:
        CFG.mqtt_enabled = False

    log.info("=" * 60)
    log.info("  BirdGuard Pipeline v6 - Multi-Mode")
    log.info("  Stream: %s", f"http://0.0.0.0:{args.port}" if shared.stream_enabled else "OFF")
    log.info("  MQTT: %s (%s:%d)", "ON" if CFG.mqtt_enabled and mqtt else "OFF",
             CFG.mqtt_broker, CFG.mqtt_port)
    log.info("  Modes: smart_patrol | patrol | manual")
    log.info("  Boot mode: smart_patrol")
    log.info("  Model: %s @ %dx%d", CFG.model_path, CFG.input_size, CFG.input_size)
    log.info("  Capture: %dx%d", CFG.cam_capture_w, CFG.cam_capture_h)
    log.info("  Centre-crop: %s", "ON" if CFG.centre_crop else "OFF")
    log.info("  SAHI: %s (%dx%d tiles, %.0f%% overlap)",
             "ON" if CFG.sahi_enabled else "OFF",
             CFG.sahi_slices_x, CFG.sahi_slices_y,
             CFG.sahi_overlap_ratio * 100)
    if os.path.exists(CONFIG_PERSIST_PATH):
        log.info("  Loaded saved config from %s", CONFIG_PERSIST_PATH)
    log.info("=" * 60)

    def _shutdown(sig, frame):
        log.info("Shutdown signal received")
        shutdown_flag.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Open shared camera
    open_shared_camera()

    # Connect turret
    turret = None
    try:
        turret = PicoTurret()
        log.info("Turret connected")
    except Exception as exc:
        log.error("Turret init failed: %s", exc)

    # Start MQTT
    mqtt_mgr = MQTTManager()
    mqtt_mgr.start()

    # Make mqtt_mgr accessible to modes via shared module
    # (modes can import and use it for telemetry)
    shared.mqtt_mgr = mqtt_mgr

    # Start web stream if requested
    if shared.stream_enabled:
        threading.Thread(target=start_stream, args=(args.port,),
                         daemon=True).start()

    # Create mode manager and boot into Smart Patrol
    mode_mgr = ModeManager(turret)
    mode_mgr.start_mode(Mode.SMART_PATROL)

    log.info("BirdGuard running. Ctrl+C to stop.")

    try:
        while not shutdown_flag.is_set():
            # Periodic health check — monitors camera and turret status,
            # triggers fallback from Smart Patrol to Regular Patrol if
            # camera is lost, emits telemetry alerts for turret failures.
            mode_mgr.check_health()
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_flag.set()

    # Shutdown
    mode_mgr.stop()
    if mqtt_mgr:
        mqtt_mgr.stop()
    close_shared_camera()
    if turret:
        turret.close()
    log.info("BirdGuard shut down.")

    # Check if this was a restart request — if so, re-exec the process
    restart_marker = os.path.expanduser("~/.birdguard_restart")
    if os.path.exists(restart_marker):
        try:
            with open(restart_marker, "r") as f:
                restart_data = json.load(f)
            os.remove(restart_marker)
            flags = restart_data.get("flags", [])
            exec_args = [sys.executable, os.path.abspath(__file__)] + flags
            log.info("Re-executing: %s", " ".join(exec_args))
            os.execv(sys.executable, exec_args)
        except Exception as exc:
            log.error("Restart failed: %s", exc)
            if os.path.exists(restart_marker):
                os.remove(restart_marker)


if __name__ == "__main__":
    main()