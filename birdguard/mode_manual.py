#!/usr/bin/env python3
"""
BirdGuard - Manual Mode
========================
The turret stays in place. The user controls pan, tilt, and laser
entirely through MQTT commands. Real-time camera stream for the webapp.

MQTT mode commands (on birdguard/mode topic):
  - manual_pan    : set pan angle (float)
  - manual_tilt   : set tilt angle (float)
  - manual_laser  : set laser state (0 or 1)
  - manual_move   : set pan + tilt + laser at once
                     {"command": "manual_move", "pan": 90, "tilt": 85, "laser": 1}

Immediate movement (no smooth interpolation) for responsive control.
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


class ManualMode(ModeBase):
    """Manual turret control via MQTT with real-time camera stream."""

    def __init__(self, turret):
        super().__init__(turret)
        self._cur_pan = CFG.pan_centre
        self._cur_tilt = CFG.tilt_centre
        self._cur_laser = 0

    def run(self):
        # Move to centre, laser off
        self._cur_pan = CFG.pan_centre
        self._cur_tilt = CFG.tilt_centre
        self._cur_laser = 0
        self.cmd(int(round(self._cur_pan)), int(round(self._cur_tilt)),
                 self._cur_laser)

        # Run continuous streaming
        target_fps = 12
        interval = 1.0 / target_fps

        while not self.should_stop():
            t0 = time.perf_counter()

            ret, frame = grab_camera_frame(drain=False)
            if ret and frame is not None:
                # Draw overlay with current position
                laser_str = "LASER ON" if self._cur_laser else "LASER OFF"
                display = draw_overlay(
                    frame, [], -1,
                    self._cur_pan, self._cur_tilt, 0, State.IDLE,
                    mode_label=f"MANUAL ({laser_str})"
                )
                update_stream_frame(display)

            elapsed = time.perf_counter() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup: laser off
        self.cmd(int(round(self._cur_pan)), int(round(self._cur_tilt)), 0)
        log.info("ManualMode exited")

    def on_mode_command(self, cmd: str, payload: dict):
        """Handle manual control commands.

        Expected MQTT payload on birdguard/mode:
          {"command": "manual_pan", "value": 90.0}
          {"command": "manual_tilt", "value": 85.0}
          {"command": "manual_laser", "value": 1}
          {"command": "manual_move", "pan": 90, "tilt": 85, "laser": 1}
          {"command": "manual_deter"}
        """
        try:
            if cmd == "manual_pan":
                value = float(payload.get("value", self._cur_pan))
                self._cur_pan = clamp(value, CFG.pan_min, CFG.pan_max)
                self.cmd(int(round(self._cur_pan)), int(round(self._cur_tilt)),
                         self._cur_laser)
                log.info("Manual pan set to %.1f", self._cur_pan)

            elif cmd == "manual_tilt":
                value = float(payload.get("value", self._cur_tilt))
                self._cur_tilt = clamp(value, CFG.tilt_min, CFG.tilt_max)
                self.cmd(int(round(self._cur_pan)), int(round(self._cur_tilt)),
                         self._cur_laser)
                log.info("Manual tilt set to %.1f", self._cur_tilt)

            elif cmd == "manual_laser":
                value = int(payload.get("value", 0))
                self._cur_laser = 1 if value else 0
                self.cmd(int(round(self._cur_pan)), int(round(self._cur_tilt)),
                         self._cur_laser)
                log.info("Manual laser set to %s",
                         "ON" if self._cur_laser else "OFF")

            elif cmd == "manual_move":
                pan = float(payload.get("pan", self._cur_pan))
                tilt = float(payload.get("tilt", self._cur_tilt))
                laser = int(payload.get("laser", self._cur_laser))
                self._cur_pan = clamp(pan, CFG.pan_min, CFG.pan_max)
                self._cur_tilt = clamp(tilt, CFG.tilt_min, CFG.tilt_max)
                self._cur_laser = 1 if laser else 0
                self.cmd(int(round(self._cur_pan)), int(round(self._cur_tilt)),
                         self._cur_laser)
                log.info("Manual move: pan=%.1f tilt=%.1f laser=%d",
                         self._cur_pan, self._cur_tilt, self._cur_laser)

            elif cmd == "manual_deter":
                log.info("Manual deter sweep at pan=%.1f tilt=%.1f",
                         self._cur_pan, self._cur_tilt)
                # Run deterrent sweep in a background thread so the
                # main stream loop and MQTT commands aren't blocked
                threading.Thread(
                    target=self._run_deter_sweep,
                    name="ManualDeter", daemon=True
                ).start()

            else:
                log.warning("Manual: unknown mode command '%s'", cmd)

        except (ValueError, TypeError) as exc:
            log.error("Manual mode command '%s' bad value: %s", cmd, exc)

    def _run_deter_sweep(self):
        """Execute a deterrent sweep at the current position, then
        return to that position with the previous laser state."""
        sweep_pan = self._cur_pan
        sweep_tilt = self._cur_tilt
        prev_laser = self._cur_laser

        self.deterrent_sweep(sweep_pan, sweep_tilt)

        # Restore position and previous laser state after sweep
        self.cmd(int(round(sweep_pan)), int(round(sweep_tilt)), prev_laser)
        log.info("Manual deter sweep complete, restored to pan=%.1f tilt=%.1f",
                 sweep_pan, sweep_tilt)