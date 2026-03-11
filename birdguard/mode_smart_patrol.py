#!/usr/bin/env python3
"""
BirdGuard - Smart Patrol Mode
==============================
AI-powered bird detection and deterrence. This is the original BirdGuard
behaviour extracted into a mode class.

Uses: AudioMonitor (wake trigger) + SnapshotWorker (inference) + DecisionEngine
(state machine: IDLE -> SCANNING -> TRACKING -> SEARCHING).

Audio monitoring runs ONLY in this mode.
Camera is used for inference snapshots (not continuous streaming).
"""

import math
import queue
import random
import threading
import time
from typing import Optional

import numpy as np
import cv2

from shared import (
    ModeBase, State, Mode, CFG, log,
    AudioEvent, VisualResult,
    audio_queue, visual_queue, snapshot_request,
    shutdown_flag, frame_lock, state_lock, turret_pos_lock,
    turret_moving, turret_pos, current_state, measured_pass_s,
    stream_enabled, mode_lock, current_mode,
    shared_camera, shared_camera_lock,
    clamp, letterbox, centre_crop_frame, generate_sahi_slices,
    draw_overlay, update_stream_frame,
)
import shared  # for mqtt_mgr access

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import alsaaudio
except ImportError:
    alsaaudio = None


class AudioMonitor(threading.Thread):
    """Continuously monitors audio for wake events.
    Only runs during Smart Patrol mode."""

    def __init__(self, stop_event: threading.Event):
        super().__init__(name="AudioMonitor", daemon=True)
        self._stop_event = stop_event
        self.last_wake_time = 0.0

    def run(self):
        if alsaaudio is None:
            log.warning("alsaaudio not available - audio idle")
            while not self._stop_event.is_set() and not shutdown_flag.is_set():
                time.sleep(1.0)
            return

        # Configure ReSpeaker 2-Mic HAT mixer settings explicitly.
        # alsactl restore is unreliable across reboots because the state
        # file may not contain the correct gain/boost settings.
        try:
            import subprocess
            mixer_cmds = [
                ["amixer", "-c", "0", "set", "Capture", "40"],
                ["amixer", "-c", "0", "set", "ADC PCM", "200"],
                ["amixer", "-c", "0", "set", "Left Input Boost Mixer LINPUT1", "1"],
                ["amixer", "-c", "0", "set", "Right Input Boost Mixer RINPUT1", "1"],
                ["amixer", "-c", "0", "set", "Left Input Mixer Boost", "on"],
                ["amixer", "-c", "0", "set", "Right Input Mixer Boost", "on"],
                ["amixer", "-c", "0", "set", "Left Boost Mixer LINPUT1", "on"],
                ["amixer", "-c", "0", "set", "Right Boost Mixer RINPUT1", "on"],
            ]
            for cmd in mixer_cmds:
                subprocess.run(cmd, capture_output=True, timeout=5)
            log.info("ALSA mixer settings applied (ReSpeaker 2-Mic)")
        except Exception as exc:
            log.warning("ALSA mixer setup failed: %s", exc)

        try:
            stream = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL,
                device=CFG.audio_device, channels=CFG.audio_channels,
                rate=CFG.sample_rate, format=alsaaudio.PCM_FORMAT_S16_LE,
                periodsize=CFG.audio_chunk,
            )
        except alsaaudio.ALSAAudioError as exc:
            log.error("ALSA open failed: %s", exc)
            while not self._stop_event.is_set() and not shutdown_flag.is_set():
                time.sleep(1.0)
            return

        log.info("Audio started (%s, %d Hz)", CFG.audio_device, CFG.sample_rate)

        while not self._stop_event.is_set() and not shutdown_flag.is_set():
            try:
                length, data = stream.read()
                if length > 0:
                    self._process(data)
            except Exception as exc:
                log.error("Audio error: %s", exc)
                time.sleep(0.1)

        log.info("AudioMonitor stopped")

    def _process(self, raw):
        if time.perf_counter() - self.last_wake_time < CFG.audio_cooldown:
            return
        if turret_moving.is_set():
            return

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if len(samples) < 4:
            return
        ch0 = samples[0::2]
        ch1 = samples[1::2]

        rms = float(np.sqrt(np.mean(ch0 ** 2)))
        if rms < CFG.energy_threshold:
            return

        bearing = self._tdoa(ch0, ch1)
        try:
            audio_queue.put_nowait(AudioEvent(time.perf_counter(), bearing, rms))
        except queue.Full:
            pass
        self.last_wake_time = time.perf_counter()

    def _tdoa(self, ch0, ch1):
        n = len(ch0) + len(ch1) - 1
        fft_size = 1
        while fft_size < n:
            fft_size <<= 1

        X0 = np.fft.rfft(ch0, n=fft_size)
        X1 = np.fft.rfft(ch1, n=fft_size)
        cross = X0 * np.conj(X1)
        gcc = np.fft.irfft(cross / (np.abs(cross) + 1e-10))

        max_lag = int(CFG.mic_spacing_m / CFG.speed_of_sound * CFG.sample_rate) + 1
        search = np.concatenate([gcc[:max_lag + 1], gcc[-max_lag:]])
        lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
        lag = lags[int(np.argmax(np.abs(search)))]

        tdoa = lag / CFG.sample_rate
        sin_t = clamp(tdoa * CFG.speed_of_sound / CFG.mic_spacing_m, -1, 1)
        bearing = CFG.pan_centre - math.degrees(math.asin(sin_t))
        return clamp(bearing, CFG.pan_min, CFG.pan_max)


class FrameGrabber(threading.Thread):
    """Continuously reads frames from the shared camera in the background.

    Instead of draining 5 stale frames on every snapshot (costing ~680 ms),
    this thread keeps a single fresh frame available at all times. The
    SnapshotWorker can grab it instantly via get_frame().

    Reports camera loss via shared.camera_healthy after consecutive failures.
    Camera reconnection is handled by ModeManager.check_health() in the
    main loop, which runs regardless of which mode is active.
    """

    FAIL_THRESHOLD = 30  # ~3 seconds of failures before marking unhealthy

    def __init__(self, stop_event: threading.Event):
        super().__init__(name="FrameGrabber", daemon=True)
        self._stop_event = stop_event
        self._frame = None
        self._lock = threading.Lock()
        self._frame_time = 0.0
        self._consecutive_fails = 0

    def run(self):
        log.info("FrameGrabber started")

        # Initial warmup: drain old frames from the USB camera buffer
        with shared_camera_lock:
            if shared.shared_camera is not None and shared.shared_camera.isOpened():
                for _ in range(10):
                    shared.shared_camera.grab()
        log.info("FrameGrabber: warmup drain complete")

        while not self._stop_event.is_set() and not shutdown_flag.is_set():
            with shared_camera_lock:
                if shared.shared_camera is None or not shared.shared_camera.isOpened():
                    self._consecutive_fails += 1
                    self._update_health()
                    time.sleep(0.1)
                    continue
                ret, frame = shared.shared_camera.read()

            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.perf_counter()
                if self._consecutive_fails > 0:
                    self._consecutive_fails = 0
                    with shared.camera_healthy_lock:
                        if not shared.camera_healthy:
                            shared.camera_healthy = True
                            log.info("FrameGrabber: camera recovered")
            else:
                self._consecutive_fails += 1
                self._update_health()
                time.sleep(0.01)

        log.info("FrameGrabber stopped")

    def _update_health(self):
        """Mark camera unhealthy after sustained failures."""
        if self._consecutive_fails >= self.FAIL_THRESHOLD:
            with shared.camera_healthy_lock:
                if shared.camera_healthy:
                    shared.camera_healthy = False
                    log.error("FrameGrabber: camera lost after %d consecutive "
                              "failures", self._consecutive_fails)

    def get_frame(self):
        """Get the latest frame. Returns (frame, age_ms) or (None, -1)."""
        with self._lock:
            if self._frame is None:
                return None, -1
            age_ms = (time.perf_counter() - self._frame_time) * 1000.0
            return self._frame.copy(), age_ms


class SnapshotWorker(threading.Thread):
    """Runs inference in background. Waits for snapshot_request, grabs a
    frame from the shared camera, runs the model, and posts to visual_queue.
    Only active during Smart Patrol mode."""

    def __init__(self, stop_event: threading.Event, frame_grabber: FrameGrabber):
        super().__init__(name="SnapshotWorker", daemon=True)
        self._stop_event = stop_event
        self._frame_grabber = frame_grabber
        self.session = None
        self.inp_name = None
        self._last_pass_ms = 4000.0

    def run(self):
        self._load_model()
        log.info("SnapshotWorker ready  model=%s",
                 "OK" if self.session else "FAIL")

        while not self._stop_event.is_set() and not shutdown_flag.is_set():
            snapshot_request.wait(timeout=1.0)
            if self._stop_event.is_set() or shutdown_flag.is_set():
                break
            if not snapshot_request.is_set():
                continue
            snapshot_request.clear()

            result = self._do_snapshot()

            # Update stream overlay (snapshot-driven in smart patrol)
            if shared.stream_enabled and result.frame is not None:
                with state_lock:
                    st = current_state
                best_idx = -1
                if result.dets:
                    best_idx = max(range(len(result.dets)),
                                   key=lambda i: result.dets[i]["conf"])
                display = draw_overlay(
                    result.frame, result.dets, best_idx,
                    result.capture_pan, result.capture_tilt,
                    result.inf_ms, st, mode_label="SMART PATROL"
                )
                update_stream_frame(display)

            try:
                visual_queue.put_nowait(result)
            except queue.Full:
                try:
                    visual_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    visual_queue.put_nowait(result)
                except queue.Full:
                    pass

        log.info("SnapshotWorker stopped")

    def _do_snapshot(self) -> VisualResult:
        t0 = time.perf_counter()

        # Get the latest frame from the background FrameGrabber.
        # This is nearly instant (~0 ms) compared to the old approach
        # of draining 5 stale USB frames (~680 ms).
        frame, age_ms = self._frame_grabber.get_frame()
        with turret_pos_lock:
            cap_pan = turret_pos["pan"]
            cap_tilt = turret_pos["tilt"]

        if frame is None:
            log.warning("SnapshotWorker: no frame available from FrameGrabber")
            return VisualResult(t0, detected=False,
                                capture_pan=cap_pan, capture_tilt=cap_tilt)

        if age_ms > 500:
            log.warning("SnapshotWorker: frame is %.0f ms old, may be stale", age_ms)

        ret = True

        if self.session is None:
            return VisualResult(t0, detected=False,
                                capture_pan=cap_pan, capture_tilt=cap_tilt,
                                frame=frame)

        crop_x_offset = 0
        working_frame = frame
        if CFG.centre_crop:
            working_frame, crop_x_offset = centre_crop_frame(frame)

        wh, ww = working_frame.shape[:2]
        oh, ow = frame.shape[:2]

        t_inf_start = time.perf_counter()
        if CFG.sahi_enabled and self.session is not None:
            all_dets = self._run_sahi(working_frame, crop_x_offset, ow, oh)
        else:
            all_dets = self._run_inference_on_region(
                working_frame, crop_x_offset, 0, ow, oh)
        t_inf_end = time.perf_counter()
        inf_ms = (t_inf_end - t_inf_start) * 1000

        if all_dets:
            best = max(all_dets, key=lambda d: d["conf"])
            return VisualResult(
                timestamp=t0, detected=True,
                bbox=(best["cx"], best["cy"], best["w"], best["h"]),
                confidence=best["conf"],
                offset_x=best["cx"] - 0.5,
                offset_y=best["cy"] - 0.5,
                capture_pan=cap_pan, capture_tilt=cap_tilt,
                inf_ms=inf_ms, dets=all_dets, frame=frame,
            )
        else:
            return VisualResult(
                t0, detected=False,
                capture_pan=cap_pan, capture_tilt=cap_tilt,
                inf_ms=inf_ms, dets=all_dets, frame=frame,
            )

    def _run_sahi(self, working_frame, crop_x_offset, orig_w, orig_h):
        wh, ww = working_frame.shape[:2]
        slices = generate_sahi_slices(
            wh, ww, CFG.sahi_slices_x, CFG.sahi_slices_y, CFG.sahi_overlap_ratio)
        n_tiles = len(slices)
        log.info("SAHI: %d tiles from %dx%d", n_tiles, ww, wh)

        all_dets = []
        for idx, (sx1, sy1, sx2, sy2) in enumerate(slices):
            tile = working_frame[sy1:sy2, sx1:sx2]
            t_tile = time.perf_counter()
            tile_dets = self._run_inference_on_region(
                tile, crop_x_offset + sx1, sy1, orig_w, orig_h)
            log.info("  tile %d/%d (%dx%d): %.0f ms, %d dets",
                     idx + 1, n_tiles, tile.shape[1], tile.shape[0],
                     (time.perf_counter() - t_tile) * 1000, len(tile_dets))
            all_dets.extend(tile_dets)

            # Early exit: bird found, skip remaining tiles
            if tile_dets:
                log.info("SAHI early exit: bird found on tile %d/%d, "
                         "skipping %d remaining tiles",
                         idx + 1, n_tiles, n_tiles - idx - 1)
                break

        pre_merge = len(all_dets)
        if len(all_dets) > 1:
            all_dets = self._merge_sahi_dets(all_dets, orig_w, orig_h)

        log.info("SAHI total: %d raw -> %d merged", pre_merge, len(all_dets))
        return all_dets

    def _run_inference_on_region(self, region, x_offset_px, y_offset_px,
                                 orig_w, orig_h):
        global measured_pass_s
        img, scale, pad_x, pad_y = letterbox(region, CFG.input_size)
        blob = np.expand_dims(
            np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1)), 0)

        t_pass = time.perf_counter()
        outputs = self.session.run(None, {self.inp_name: blob})
        pass_ms = (time.perf_counter() - t_pass) * 1000
        self._last_pass_ms = pass_ms
        measured_pass_s = pass_ms / 1000.0

        rh, rw = region.shape[:2]
        dets = self._postprocess(outputs, scale, pad_x, pad_y, rw, rh)

        for d in dets:
            px = d["cx"] * rw + x_offset_px
            py = d["cy"] * rh + y_offset_px
            pw = d["w"] * rw
            ph = d["h"] * rh
            d["cx"] = px / orig_w
            d["cy"] = py / orig_h
            d["w"] = pw / orig_w
            d["h"] = ph / orig_h

        return dets

    def _merge_sahi_dets(self, dets, orig_w, orig_h):
        if not dets:
            return []
        boxes = []
        confs = []
        for d in dets:
            px = d["cx"] * orig_w
            py = d["cy"] * orig_h
            pw = d["w"] * orig_w
            ph = d["h"] * orig_h
            boxes.append([px - pw/2, py - ph/2, pw, ph])
            confs.append(d["conf"])

        indices = cv2.dnn.NMSBoxes(
            boxes, confs, CFG.confidence_threshold, CFG.sahi_merge_iou)
        merged = []
        if len(indices) > 0:
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                merged.append(dets[idx])
        return merged

    def _load_model(self):
        if ort is None:
            log.error("onnxruntime not available")
            return
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 3
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                CFG.model_path, sess_options=opts,
                providers=["CPUExecutionProvider"])
            self.inp_name = self.session.get_inputs()[0].name
            log.info("Model loaded: %s (input %dx%d)",
                     CFG.model_path, CFG.input_size, CFG.input_size)
        except Exception as exc:
            log.error("Model load failed: %s", exc)

    def _postprocess(self, outputs, scale, pad_x, pad_y, orig_w, orig_h):
        preds = outputs[0]
        if preds.shape[1] == 84:
            preds = preds[0]
        else:
            preds = preds[0].T

        boxes = preds[:4, :]
        scores = preds[4:, :]
        bird = scores[CFG.bird_class_id, :]

        top_score = float(np.max(bird)) if len(bird) > 0 else 0.0
        n_above = int(np.sum(bird > CFG.confidence_threshold))
        if top_score > 0.1:
            log.info("Postprocess: top bird=%.3f, %d above %.2f",
                     top_score, n_above, CFG.confidence_threshold)

        mask = bird > CFG.confidence_threshold
        if not np.any(mask):
            return []

        bird = bird[mask]
        bx = boxes[:, mask]
        cx = (bx[0] - pad_x) / scale
        cy = (bx[1] - pad_y) / scale
        w = bx[2] / scale
        h = bx[3] / scale

        nms_boxes = np.stack([cx - w/2, cy - h/2, w, h], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, bird.tolist(),
                                    CFG.confidence_threshold, CFG.nms_iou_threshold)
        dets = []
        if len(indices) > 0:
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                dets.append({
                    "cx": float(cx[idx]) / orig_w,
                    "cy": float(cy[idx]) / orig_h,
                    "w": float(w[idx]) / orig_w,
                    "h": float(h[idx]) / orig_h,
                    "conf": float(bird[idx]),
                })
        return dets


class SmartPatrolMode(ModeBase):
    """Smart Patrol: AI-powered audio + visual bird detection and deterrence.

    Manages its own AudioMonitor and SnapshotWorker threads.
    Runs the full state machine: IDLE -> SCANNING -> TRACKING -> SEARCHING.
    """

    def __init__(self, turret):
        super().__init__(turret)
        self._audio_thread: Optional[AudioMonitor] = None
        self._frame_grabber: Optional[FrameGrabber] = None
        self._snapshot_thread: Optional[SnapshotWorker] = None
        self.state = State.IDLE
        self.cur_pan = CFG.pan_centre
        self.cur_tilt = CFG.smart_patrol_tilt
        self.last_bird_pan = CFG.pan_centre
        self.last_bird_tilt = CFG.smart_patrol_tilt
        self._patrol_direction = 1
        self._patrol_has_run = False

    def run(self):
        # Start sub-threads — FrameGrabber first so frames are available
        # by the time SnapshotWorker needs them
        self._audio_thread = AudioMonitor(self._stop_event)
        self._frame_grabber = FrameGrabber(self._stop_event)
        self._snapshot_thread = SnapshotWorker(self._stop_event, self._frame_grabber)
        self._audio_thread.start()
        self._frame_grabber.start()
        self._snapshot_thread.start()

        self.cmd(int(CFG.pan_centre), int(CFG.smart_patrol_tilt), 0)
        self._set_state(State.IDLE)

        while not self.should_stop():
            if self.state == State.IDLE:
                self._do_idle()
            elif self.state == State.SCANNING:
                self._do_scanning()
            elif self.state == State.TRACKING:
                self._do_tracking()
            elif self.state == State.SEARCHING:
                self._do_searching()

        # Cleanup: laser off, centre turret
        self.cmd(int(CFG.pan_centre), int(CFG.tilt_centre), 0)

        # Wait for sub-threads
        if self._audio_thread:
            self._audio_thread.join(timeout=3)
        if self._frame_grabber:
            self._frame_grabber.join(timeout=3)
        if self._snapshot_thread:
            self._snapshot_thread.join(timeout=3)

        log.info("SmartPatrolMode exited")

    def on_mode_command(self, cmd: str, payload: dict):
        """Smart patrol has no mode-specific commands currently."""
        log.warning("SmartPatrol: unknown mode command '%s'", cmd)

    def _set_state(self, new_state):
        global current_state
        if self.state != new_state:
            log.info("STATE: %s -> %s", self.state.value, new_state.value)
            if shared.mqtt_mgr:
                shared.mqtt_mgr.publish_telemetry(
                    "state_change",
                    from_state=self.state.value,
                    to_state=new_state.value,
                    pan=self.cur_pan, tilt=self.cur_tilt,
                )
        self.state = new_state
        with state_lock:
            current_state = new_state

    def _drain_visual_queue(self):
        """Discard any stale visual results."""
        while not visual_queue.empty():
            try:
                stale = visual_queue.get_nowait()
                if stale.detected:
                    log.warning("Draining stale visual result with detection! "
                                "conf=%.2f pan=%.0f — this detection was lost",
                                stale.confidence, stale.capture_pan)
            except queue.Empty:
                break

    def _request_snapshot(self):
        self._drain_visual_queue()
        snapshot_request.set()

    def _inference_timeout(self):
        if CFG.sahi_enabled:
            n_passes = CFG.sahi_slices_x * CFG.sahi_slices_y
        else:
            n_passes = 1
        per_pass = max(measured_pass_s, 1.0)
        return n_passes * per_pass * 1.5 + 3.0

    def _wait_for_result_or_audio(self, timeout=None):
        """Poll both queues. Returns (visual_result, audio_event).
        
        If a visual result with a detection is already available when
        audio arrives, the visual result takes priority — a confirmed
        bird detection is more valuable than a speculative audio bearing.
        """
        if timeout is None:
            timeout = self._inference_timeout()
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.should_stop():
                return None, None

            # Check visual FIRST — a completed detection is high value
            try:
                result = visual_queue.get_nowait()
                log.info("Visual result: detected=%s conf=%.2f",
                         result.detected, result.confidence)
                return result, None
            except queue.Empty:
                pass

            # Then check audio
            try:
                event = audio_queue.get_nowait()
                # Before returning the audio event, do one final check
                # for a visual result that may have just landed
                try:
                    result = visual_queue.get_nowait()
                    if result.detected:
                        log.info("Visual result (bird detected, conf=%.2f) "
                                 "arrived alongside audio — prioritising visual",
                                 result.confidence)
                        return result, None
                except queue.Empty:
                    pass
                return None, event
            except queue.Empty:
                pass

            time.sleep(0.05)
        return None, None

    def _settle_snapshot_or_audio(self, settle_time: float):
        time.sleep(settle_time)
        turret_moving.clear()
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
        self._request_snapshot()
        return self._wait_for_result_or_audio()

    def _settle_and_snapshot_only(self, settle_time: float):
        time.sleep(settle_time)
        turret_moving.clear()
        self._drain_visual_queue()
        snapshot_request.set()
        try:
            return visual_queue.get(timeout=self._inference_timeout())
        except queue.Empty:
            return None

    # ---- IDLE (patrol sweep) ----

    def _do_idle(self):
        steps = CFG.patrol_pan_steps
        pan_lo = CFG.patrol_pan_min
        pan_hi = CFG.patrol_pan_max
        step_size = (pan_hi - pan_lo) / max(steps - 1, 1)

        if self._patrol_direction == 1:
            positions = [pan_lo + step_size * i for i in range(steps)]
        else:
            positions = [pan_hi - step_size * i for i in range(steps)]

        if self._patrol_has_run:
            positions = positions[1:]
        self._patrol_has_run = True

        for pan in positions:
            if self.should_stop():
                return

            self.smooth_move(pan, CFG.smart_patrol_tilt, laser=0)
            self.cur_pan = pan
            self.cur_tilt = CFG.smart_patrol_tilt

            # Settle: wait for servo vibration to stop before snapshot
            time.sleep(CFG.snapshot_settle)
            turret_moving.clear()
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break

            self._request_snapshot()
            result, audio_evt = self._wait_for_result_or_audio()

            if audio_evt is not None:
                log.info("Audio wake during patrol - RMS %.0f  bearing %.1f",
                         audio_evt.rms_energy, audio_evt.bearing_deg)
                self.cur_pan = audio_evt.bearing_deg
                self._patrol_direction *= -1
                self._set_state(State.SCANNING)
                return

            if result is not None and result.detected:
                log.info("Visual wake during patrol! conf=%.2f", result.confidence)
                target_pan = result.capture_pan + (-result.offset_x * CFG.cam_hfov_deg)
                target_tilt = result.capture_tilt + (result.offset_y * CFG.cam_vfov_deg)
                self.cur_pan = clamp(target_pan, CFG.pan_min, CFG.pan_max)
                self.cur_tilt = clamp(target_tilt, CFG.tilt_min, CFG.tilt_max)
                self.last_bird_pan = self.cur_pan
                self.last_bird_tilt = self.cur_tilt
                self._patrol_direction *= -1
                self._set_state(State.TRACKING)
                return

        self._patrol_direction *= -1

    # ---- SCANNING ----

    def _do_scanning(self):
        bearing = self.cur_pan
        log.info("Scanning: bearing=%.0f, +/-%.0f", bearing, CFG.scan_half_angle)

        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        self.cmd(int(round(bearing)), int(round(CFG.smart_patrol_tilt)), 0)
        result, audio_evt = self._settle_snapshot_or_audio(CFG.snapshot_settle)

        if audio_evt is not None and abs(audio_evt.bearing_deg - bearing) > 20:
            self.cur_pan = audio_evt.bearing_deg
            self._set_state(State.SCANNING)
            return

        if result is not None and result.detected:
            self._refine_and_track(result)
            return

        pan_lo = clamp(bearing - CFG.scan_half_angle, CFG.pan_min, CFG.pan_max)
        pan_hi = clamp(bearing + CFG.scan_half_angle, CFG.pan_min, CFG.pan_max)
        pan_step = (pan_hi - pan_lo) / max(CFG.scan_pan_steps - 1, 1)

        for dir_idx, direction in enumerate([1, -1]):
            if direction == 1:
                positions = [pan_lo + pan_step * i for i in range(CFG.scan_pan_steps)]
            else:
                positions = [pan_hi - pan_step * i for i in range(CFG.scan_pan_steps)]
                # Skip first position — same as last stop of forward sweep
                if len(positions) > 1:
                    positions = positions[1:]

            for pan in positions:
                if self.should_stop():
                    return
                self.cur_pan = pan
                self.cur_tilt = CFG.smart_patrol_tilt
                self.cmd(int(round(pan)), int(round(CFG.smart_patrol_tilt)), 0)
                result, audio_evt = self._settle_snapshot_or_audio(CFG.snapshot_settle)

                if audio_evt is not None:
                    if abs(audio_evt.bearing_deg - bearing) > 20:
                        self.cur_pan = audio_evt.bearing_deg
                        self._set_state(State.SCANNING)
                        return

                if result is not None and result.detected:
                    self._refine_and_track(result)
                    return

        log.info("Scan complete - no bird found")
        self._set_state(State.IDLE)

    def _refine_and_track(self, result):
        target_pan = result.capture_pan + (-result.offset_x * CFG.cam_hfov_deg)
        target_tilt = result.capture_tilt + (result.offset_y * CFG.cam_vfov_deg)
        self.cur_pan = clamp(target_pan, CFG.pan_min, CFG.pan_max)
        self.cur_tilt = clamp(target_tilt, CFG.tilt_min, CFG.tilt_max)
        self.last_bird_pan = self.cur_pan
        self.last_bird_tilt = self.cur_tilt
        self._set_state(State.TRACKING)

    # ---- TRACKING ----

    def _do_tracking(self):
        log.info("TRACKING at pan=%.0f tilt=%.0f", self.cur_pan, self.cur_tilt)
        self.cmd(int(round(self.cur_pan)), int(round(self.cur_tilt)), 0)
        self.deterrent_sweep(self.cur_pan, self.cur_tilt)

        consecutive_misses = 0
        for _ in range(CFG.track_recheck_max):
            if self.should_stop():
                return

            self.cmd(int(round(self.cur_pan)), int(round(self.cur_tilt)), 0)
            result, audio_evt = self._settle_snapshot_or_audio(CFG.snapshot_settle)

            if audio_evt is not None:
                if abs(audio_evt.bearing_deg - self.cur_pan) > 30:
                    self.cmd(int(round(self.cur_pan)), int(round(self.cur_tilt)), 0)
                    self.cur_pan = audio_evt.bearing_deg
                    self._set_state(State.SCANNING)
                    return

            if result is not None and result.detected:
                consecutive_misses = 0
                target_pan = result.capture_pan + (-result.offset_x * CFG.cam_hfov_deg)
                target_tilt = result.capture_tilt + (result.offset_y * CFG.cam_vfov_deg)
                self.cur_pan = clamp(target_pan, CFG.pan_min, CFG.pan_max)
                self.cur_tilt = clamp(target_tilt, CFG.tilt_min, CFG.tilt_max)
                self.last_bird_pan = self.cur_pan
                self.last_bird_tilt = self.cur_tilt
                self.cmd(int(round(self.cur_pan)), int(round(self.cur_tilt)), 0)
                self.deterrent_sweep(self.cur_pan, self.cur_tilt)

            elif result is not None and not result.detected:
                consecutive_misses += 1
                if consecutive_misses >= 2:
                    self.cmd(int(round(self.cur_pan)), int(round(self.cur_tilt)), 0)
                    self._set_state(State.SEARCHING)
                    return

        self.cmd(int(round(self.cur_pan)), int(round(self.cur_tilt)), 0)
        self._set_state(State.IDLE)

    # ---- SEARCHING ----

    def _do_searching(self):
        log.info("Searching: tilt=%.0f, pan +/-%.0f from %.0f",
                 self.last_bird_tilt, CFG.search_half_angle, self.last_bird_pan)

        t_start = time.perf_counter()
        pan_lo = clamp(self.last_bird_pan - CFG.search_half_angle,
                       CFG.pan_min, CFG.pan_max)
        pan_hi = clamp(self.last_bird_pan + CFG.search_half_angle,
                       CFG.pan_min, CFG.pan_max)
        tilt = self.last_bird_tilt
        pan_step = (pan_hi - pan_lo) / max(CFG.search_pan_steps - 1, 1)

        for dir_idx, direction in enumerate([1, -1]):
            if direction == 1:
                positions = [pan_lo + pan_step * i for i in range(CFG.search_pan_steps)]
            else:
                positions = [pan_hi - pan_step * i for i in range(CFG.search_pan_steps)]
                # Skip first position — it's the same as the last stop
                # of the forward sweep
                if len(positions) > 1:
                    positions = positions[1:]

            for pan in positions:
                if self.should_stop():
                    return
                if time.perf_counter() - t_start > CFG.search_timeout:
                    log.info("Search timeout - returning to IDLE")
                    self._set_state(State.IDLE)
                    return

                self.cur_pan = pan
                self.cur_tilt = tilt
                self.cmd(int(round(pan)), int(round(tilt)), 0)
                result, audio_evt = self._settle_snapshot_or_audio(CFG.snapshot_settle)

                if audio_evt is not None:
                    if abs(audio_evt.bearing_deg - self.last_bird_pan) > 20:
                        self.cur_pan = audio_evt.bearing_deg
                        self._set_state(State.SCANNING)
                        return

                if result is not None and result.detected:
                    target_pan = result.capture_pan + (-result.offset_x * CFG.cam_hfov_deg)
                    target_tilt = result.capture_tilt + (result.offset_y * CFG.cam_vfov_deg)
                    self.cur_pan = clamp(target_pan, CFG.pan_min, CFG.pan_max)
                    self.cur_tilt = clamp(target_tilt, CFG.tilt_min, CFG.tilt_max)
                    self.last_bird_pan = self.cur_pan
                    self.last_bird_tilt = self.cur_tilt
                    self._set_state(State.TRACKING)
                    return

        log.info("Search complete - bird gone, returning to IDLE")
        self._set_state(State.IDLE)