#!/usr/bin/env python3
"""
BirdGuard - Regular Patrol Mode
================================
The turret sweeps left-to-right-to-left continuously with the laser on.
A real-time camera stream (~10-15 fps) runs for the webapp.

Configurable via MQTT mode commands:
  - patrol_speed   : degrees per second (float)
  - patrol_laser   : laser on/off (bool, 0 or 1)
  - patrol_tilt    : fixed tilt angle (float) — takes effect immediately
  - patrol_pan_min : sweep left bound (float)
  - patrol_pan_max : sweep right bound (float)

No audio monitoring. No inference. Camera used only for streaming.
"""

import time
import threading

import cv2

from shared import (
    ModeBase, CFG, log,
    shutdown_flag, turret_pos_lock, turret_pos,
    stream_enabled, clamp,
    draw_overlay, update_stream_frame, grab_camera_frame,
    State,
)


class RegularPatrolMode(ModeBase):
    """Continuous sweep patrol with laser and real-time camera stream."""

    def __init__(self, turret):
        super().__init__(turret)
        # Local copies of configurable params (can be updated via MQTT)
        self._speed = CFG.regular_patrol_speed       # deg/sec
        self._pan_min = CFG.regular_patrol_pan_min
        self._pan_max = CFG.regular_patrol_pan_max
        self._tilt = CFG.regular_patrol_tilt
        self._laser = CFG.regular_patrol_laser
        self._stream_thread: threading.Thread = None

    def run(self):
        # Start continuous streaming thread
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="PatrolStream", daemon=True
        )
        self._stream_thread.start()

        # Start sweeping from current position or pan_min
        cur_pan = self._pan_min
        direction = 1  # 1 = increasing pan, -1 = decreasing

        # Move to start position
        self.cmd(int(round(cur_pan)), int(round(self._tilt)),
                 1 if self._laser else 0)
        time.sleep(0.3)  # brief settle

        # Sweep loop: move in small increments at the configured speed
        step_interval = 0.05  # 50ms per step = 20 updates/sec
        last_time = time.perf_counter()

        while not self.should_stop():
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # Calculate degrees to move this tick
            delta = self._speed * dt * direction

            cur_pan += delta

            # Check bounds and reverse
            if cur_pan >= self._pan_max:
                cur_pan = self._pan_max
                direction = -1
            elif cur_pan <= self._pan_min:
                cur_pan = self._pan_min
                direction = 1

            # Send command
            self.cmd(int(round(cur_pan)), int(round(self._tilt)),
                     1 if self._laser else 0)

            time.sleep(step_interval)

        # Cleanup: laser off
        with turret_pos_lock:
            pan = turret_pos["pan"]
            tilt = turret_pos["tilt"]
        self.cmd(int(round(pan)), int(round(tilt)), 0)

        log.info("RegularPatrolMode exited")

    def _stream_loop(self):
        """Grab frames continuously and push to the MJPEG stream."""
        target_fps = 12
        interval = 1.0 / target_fps

        while not self.should_stop():
            t0 = time.perf_counter()

            ret, frame = grab_camera_frame(drain=False)
            if ret and frame is not None:
                with turret_pos_lock:
                    pan = turret_pos["pan"]
                    tilt = turret_pos["tilt"]

                # Draw a simple overlay (no detections, just mode + position)
                display = draw_overlay(
                    frame, [], -1, pan, tilt, 0, State.IDLE,
                    mode_label="PATROL"
                )
                update_stream_frame(display)

            elapsed = time.perf_counter() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.info("Patrol stream loop stopped")

    def on_mode_command(self, cmd: str, payload: dict):
        """Handle patrol-specific configuration commands.

        Expected MQTT payload on birdguard/mode:
          {"command": "patrol_speed", "value": 20.0}
          {"command": "patrol_laser", "value": 1}
          {"command": "patrol_tilt", "value": 90.0}
          {"command": "patrol_pan_min", "value": 20.0}
          {"command": "patrol_pan_max", "value": 160.0}
        """
        value = payload.get("value")
        if value is None:
            log.warning("Patrol mode command '%s' missing 'value'", cmd)
            return

        try:
            if cmd == "patrol_speed":
                self._speed = max(1.0, min(90.0, float(value)))
                CFG.regular_patrol_speed = self._speed
                log.info("Patrol speed set to %.1f deg/s", self._speed)

            elif cmd == "patrol_laser":
                self._laser = bool(int(value))
                CFG.regular_patrol_laser = self._laser
                log.info("Patrol laser set to %s", "ON" if self._laser else "OFF")

            elif cmd == "patrol_tilt":
                self._tilt = clamp(float(value), CFG.tilt_min, CFG.tilt_max)
                CFG.regular_patrol_tilt = self._tilt
                log.info("Patrol tilt set to %.1f", self._tilt)

            elif cmd == "patrol_pan_min":
                self._pan_min = clamp(float(value), CFG.pan_min, CFG.pan_max)
                CFG.regular_patrol_pan_min = self._pan_min
                log.info("Patrol pan_min set to %.1f", self._pan_min)

            elif cmd == "patrol_pan_max":
                self._pan_max = clamp(float(value), CFG.pan_min, CFG.pan_max)
                CFG.regular_patrol_pan_max = self._pan_max
                log.info("Patrol pan_max set to %.1f", self._pan_max)

            else:
                log.warning("Patrol: unknown mode command '%s'", cmd)
        except (ValueError, TypeError) as exc:
            log.error("Patrol mode command '%s' bad value: %s", cmd, exc)
