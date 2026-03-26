#!/usr/bin/env python3
"""
BirdGuard - Smart Patrol Mode: Comprehensive Profiling Script
==============================================================
INF2009 Lab - Profiling for Edge Computing

This script profiles the Smart Patrol pipeline on the Raspberry Pi Zero 2 W.
It instruments each stage of the pipeline to measure:
  - Wall-clock time vs CPU time
  - Latency distribution (mean, p50, p95, p99) per pipeline stage
  - End-to-end latency budget verification (target: <500ms)
  - Memory usage (RSS)
  - Per-function CPU breakdown

Usage:
    # Full profiling run (recommended):
    python3 profile_smart_patrol.py --iterations 50

    # Quick test run:
    python3 profile_smart_patrol.py --iterations 10

    # With cProfile function-level breakdown:
    python -m cProfile -s tottime profile_smart_patrol.py --iterations 50

    # With perf stat hardware counters:
    perf stat -e cycles,instructions,cache-misses,cs,migrations -- \
        python3 profile_smart_patrol.py --iterations 50

    # Single-core vs multi-core comparison:
    taskset -c 0 python3 profile_smart_patrol.py --iterations 50
    taskset -c 0-3 python3 profile_smart_patrol.py --iterations 50

    # With real-time priority:
    sudo chrt --rr 50 python3 profile_smart_patrol.py --iterations 50

Outputs:
    profiling_results/
        latency_summary.txt        - Full latency report
        latency_histogram.png      - Per-stage latency distributions
        latency_boxplot.png        - Box plots comparing stages
        memory_timeline.png        - RSS memory over time
        pipeline_timing.png        - End-to-end timing diagram
        raw_latencies.csv          - Raw timing data for external analysis

Dependencies:
    numpy, opencv-python, matplotlib, psutil
    + BirdGuard dependencies (tflite_runtime or onnxruntime, pyalsaaudio)
"""

import argparse
import csv
import math
import os
import sys
import time
import threading
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import cv2

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not installed — memory profiling disabled")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless Pi
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not installed — plot generation disabled")

# ============================================================================
# Import BirdGuard modules
# ============================================================================

# Add parent directory to path if running from profiling_results/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import (
    Config, CFG, State,
    clamp, letterbox, centre_crop_frame, generate_sahi_slices,
    turret_pos_lock, turret_pos, turret_moving,
    audio_queue, visual_queue, snapshot_request,
    shutdown_flag, shared_camera_lock,
)
import shared
from mode_smart_patrol import FrameGrabber

try:
    import tflite_runtime.interpreter as tflite
    HAS_TFLITE = True
except ImportError:
    HAS_TFLITE = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import alsaaudio
    HAS_ALSA = True
except ImportError:
    HAS_ALSA = False


# ============================================================================
# Data classes for profiling results
# ============================================================================

@dataclass
class StageLatency:
    """Latency measurement for a single pipeline stage."""
    wall_time_ms: float
    cpu_time_ms: float


@dataclass
class PipelineRun:
    """Complete timing for one end-to-end pipeline iteration."""
    iteration: int
    audio_capture_ms: float = 0.0
    energy_check_ms: float = 0.0
    tdoa_ms: float = 0.0
    frame_grab_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    decision_fusion_ms: float = 0.0
    serial_command_ms: float = 0.0
    total_e2e_ms: float = 0.0
    rss_mb: float = 0.0
    cpu_time_ms: float = 0.0
    wall_time_ms: float = 0.0


# ============================================================================
# Profiler class
# ============================================================================

class SmartPatrolProfiler:
    """Instruments and profiles each stage of the Smart Patrol pipeline."""

    def __init__(self, iterations=50):
        self.iterations = iterations
        self.results: List[PipelineRun] = []
        self.memory_samples: List[tuple] = []  # (timestamp, rss_mb)
        self.process = psutil.Process() if HAS_PSUTIL else None

        # Pipeline components
        self.model = None
        self.inp_details = None
        self.out_details = None
        self.camera = None
        self.frame_grabber: Optional[FrameGrabber] = None
        self.stop_event = threading.Event()

        # Output directory
        self.output_dir = "profiling_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def setup(self):
        """Initialise camera and model for profiling."""
        print("\n" + "=" * 70)
        print("  BirdGuard Smart Patrol — Pipeline Profiler")
        print("  INF2009 Edge Computing Lab")
        print("=" * 70)

        # System info
        print(f"\n[SYSTEM INFO]")
        if HAS_PSUTIL:
            print(f"  CPU count     : {psutil.cpu_count(logical=True)} logical cores")
            mem = psutil.virtual_memory()
            print(f"  Total RAM     : {mem.total / 1024**2:.0f} MB")
            print(f"  Available RAM : {mem.available / 1024**2:.0f} MB")
        print(f"  Python        : {sys.version.split()[0]}")
        print(f"  NumPy         : {np.__version__}")
        print(f"  OpenCV        : {cv2.__version__}")
        print(f"  TFLite        : {'available' if HAS_TFLITE else 'not available'}")
        print(f"  ONNX Runtime  : {'available' if HAS_ONNX else 'not available'}")
        print(f"  ALSA Audio    : {'available' if HAS_ALSA else 'not available'}")
        print(f"  Iterations    : {self.iterations}")

        # Load model
        print(f"\n[MODEL SETUP]")
        print(f"  Model path    : {CFG.model_path}")
        print(f"  Input size    : {CFG.input_size}x{CFG.input_size}")
        print(f"  Use TFLite    : {CFG.use_tflite}")
        self._load_model()

        # Open camera
        print(f"\n[CAMERA SETUP]")
        self._open_camera()

        # Initialize FrameGrabber exactly like Smart Patrol mode
        if self.camera is not None:
            shared.shared_camera = self.camera
            shared.camera_healthy = True
            self.frame_grabber = FrameGrabber(self.stop_event)
            self.frame_grabber.start()
            print("  FrameGrabber started")

        # Record baseline memory
        if HAS_PSUTIL:
            rss = self.process.memory_info().rss / 1024 ** 2
            print(f"\n[BASELINE MEMORY]")
            print(f"  RSS after init: {rss:.1f} MB")
            self.memory_samples.append((0, rss))

    def _load_model(self):
        """Load the inference model (TFLite or ONNX)."""
        if not os.path.exists(CFG.model_path):
            print(f"  [WARN] Model file not found: {CFG.model_path}")
            print(f"  [WARN] Inference profiling will use SYNTHETIC data")
            return

        t0 = time.perf_counter()

        if CFG.use_tflite and HAS_TFLITE:
            try:
                self.model = tflite.Interpreter(
                    model_path=CFG.model_path, num_threads=3)
                self.model.allocate_tensors()
                self.inp_details = self.model.get_input_details()
                self.out_details = self.model.get_output_details()
                load_ms = (time.perf_counter() - t0) * 1000
                print(f"  TFLite loaded in {load_ms:.0f} ms")
                print(f"  Input shape   : {self.inp_details[0]['shape']}")
                print(f"  Input dtype   : {self.inp_details[0]['dtype']}")
            except Exception as exc:
                print(f"  [ERROR] TFLite load failed: {exc}")
        elif not CFG.use_tflite and HAS_ONNX:
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 3
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.model = ort.InferenceSession(
                    CFG.model_path, sess_options=opts,
                    providers=["CPUExecutionProvider"])
                self.inp_name = self.model.get_inputs()[0].name
                load_ms = (time.perf_counter() - t0) * 1000
                print(f"  ONNX loaded in {load_ms:.0f} ms")
            except Exception as exc:
                print(f"  [ERROR] ONNX load failed: {exc}")
        else:
            print(f"  [WARN] No inference runtime available")

    def _open_camera(self):
        """Open USB camera for frame capture profiling."""
        # Try configured device first
        for device in [CFG.camera_device, 0]:
            try:
                cap = cv2.VideoCapture(device)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.cam_capture_w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_capture_h)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Warmup drain
                    for _ in range(10):
                        cap.grab()
                    ret, test = cap.read()
                    if ret and test is not None:
                        self.camera = cap
                        print(f"  Camera opened : {device}")
                        print(f"  Resolution    : {test.shape[1]}x{test.shape[0]}")
                        return
                    cap.release()
            except Exception:
                pass

        print(f"  [WARN] No camera found — frame grab will use SYNTHETIC frames")

    def run(self):
        """Execute the profiling run."""
        print(f"\n{'=' * 70}")
        print(f"  PROFILING START — {self.iterations} iterations")
        print(f"{'=' * 70}\n")

        # Warmup run (excluded from results)
        print("[WARMUP] Running 3 warmup iterations...")
        for i in range(3):
            self._profile_one_iteration(-1)
        print("[WARMUP] Complete\n")

        # Profiling runs
        for i in range(self.iterations):
            run = self._profile_one_iteration(i)
            self.results.append(run)

            # Memory sample
            if HAS_PSUTIL and i % 5 == 0:
                rss = self.process.memory_info().rss / 1024 ** 2
                self.memory_samples.append((time.perf_counter(), rss))

            # Progress
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1:3d}/{self.iterations}] "
                      f"E2E: {run.total_e2e_ms:7.1f} ms  "
                      f"Inf: {run.inference_ms:7.1f} ms  "
                      f"RSS: {run.rss_mb:5.1f} MB")

        # Final memory sample
        if HAS_PSUTIL:
            rss = self.process.memory_info().rss / 1024 ** 2
            self.memory_samples.append((time.perf_counter(), rss))

        print(f"\n[PROFILING COMPLETE]\n")

    def _profile_one_iteration(self, iteration: int) -> PipelineRun:
        """Profile one complete pipeline pass (audio → inference → command)."""
        run = PipelineRun(iteration=iteration)

        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        # ---- Stage 1: Audio Capture (simulated buffer fill) ----
        t0 = time.perf_counter()
        audio_data = self._simulate_audio_capture()
        run.audio_capture_ms = (time.perf_counter() - t0) * 1000

        # ---- Stage 2: Energy Threshold Check ----
        t0 = time.perf_counter()
        ch0, ch1, rms = self._compute_energy(audio_data)
        run.energy_check_ms = (time.perf_counter() - t0) * 1000

        # ---- Stage 3: TDOA Cross-Correlation ----
        t0 = time.perf_counter()
        bearing = self._compute_tdoa(ch0, ch1)
        run.tdoa_ms = (time.perf_counter() - t0) * 1000

        # ---- Stage 4: Camera Frame Grab ----
        t0 = time.perf_counter()
        frame = self._grab_frame()
        run.frame_grab_ms = (time.perf_counter() - t0) * 1000

        # ---- Stage 5: Preprocessing (letterbox + normalise) ----
        t0 = time.perf_counter()
        blob, scale, pad_x, pad_y = self._preprocess(frame)
        run.preprocess_ms = (time.perf_counter() - t0) * 1000

        # ---- Stage 6: Model Inference ----
        t0 = time.perf_counter()
        outputs = self._run_inference(blob)
        run.inference_ms = (time.perf_counter() - t0) * 1000

        # ---- Stage 7: Postprocessing (NMS + coordinate transform) ----
        t0 = time.perf_counter()
        dets = self._postprocess(outputs, scale, pad_x, pad_y,
                                 frame.shape[1], frame.shape[0])
        run.postprocess_ms = (time.perf_counter() - t0) * 1000

        # ---- Stage 8: Decision Fusion (bearing + bbox → servo angles) ----
        t0 = time.perf_counter()
        target_pan, target_tilt = self._decision_fusion(bearing, dets)
        run.decision_fusion_ms = (time.perf_counter() - t0) * 1000

        # ---- Stage 9: Serial Command Send (simulated) ----
        t0 = time.perf_counter()
        self._simulate_serial_send(target_pan, target_tilt)
        run.serial_command_ms = (time.perf_counter() - t0) * 1000

        # ---- End-to-End ----
        wall_end = time.perf_counter()
        cpu_end = time.process_time()

        run.total_e2e_ms = (wall_end - wall_start) * 1000
        run.wall_time_ms = run.total_e2e_ms
        run.cpu_time_ms = (cpu_end - cpu_start) * 1000

        if HAS_PSUTIL:
            run.rss_mb = self.process.memory_info().rss / 1024 ** 2

        return run

    # ---- Pipeline Stage Implementations ----

    def _simulate_audio_capture(self) -> np.ndarray:
        """Simulate audio buffer capture from ReSpeaker (2-ch, 16-bit)."""
        # Generate synthetic stereo audio with some noise
        # This simulates the ALSA read of a 1024-sample chunk
        n_samples = CFG.audio_chunk * CFG.audio_channels
        audio = np.random.randint(-2000, 2000, size=n_samples, dtype=np.int16)
        # Add a louder "bird chirp" burst to exceed energy threshold
        chirp_len = 200
        chirp = np.random.randint(-5000, 5000, size=chirp_len, dtype=np.int16)
        start = n_samples // 2 - chirp_len // 2
        audio[start:start + chirp_len] = chirp
        return audio

    def _compute_energy(self, raw: np.ndarray):
        """Compute RMS energy on channel 0 (as in AudioMonitor._process)."""
        samples = raw.astype(np.float32)
        ch0 = samples[0::2]
        ch1 = samples[1::2]
        rms = float(np.sqrt(np.mean(ch0 ** 2)))
        return ch0, ch1, rms

    def _compute_tdoa(self, ch0: np.ndarray, ch1: np.ndarray) -> float:
        """TDOA cross-correlation (exact copy of AudioMonitor._tdoa)."""
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

    def _grab_frame(self) -> np.ndarray:
        """Grab a frame from camera or generate synthetic frame."""
        if self.frame_grabber is not None:
            frame, age_ms = self.frame_grabber.get_frame()
            if frame is not None:
                return frame

        if self.camera is not None and self.camera.isOpened():
            # fallback: drain stale frames (like old behaviour)
            for _ in range(2):
                self.camera.grab()
            ret, frame = self.camera.read()
            if ret and frame is not None:
                return frame

        # Synthetic fallback: 1280x720 RGB frame with random noise
        return np.random.randint(0, 255,
                                 (CFG.cam_capture_h, CFG.cam_capture_w, 3),
                                 dtype=np.uint8)

    def _preprocess(self, frame: np.ndarray):
        """Letterbox resize + normalise to float32 [0,1]."""
        img, scale, pad_x, pad_y = letterbox(frame, CFG.input_size)
        blob = img.astype(np.float32) / 255.0
        return blob, scale, pad_x, pad_y

    def _run_inference(self, blob: np.ndarray):
        """Run model inference (real or synthetic)."""
        if self.model is not None and CFG.use_tflite and HAS_TFLITE:
            inp = np.expand_dims(blob, 0)
            self.model.set_tensor(self.inp_details[0]['index'], inp)
            self.model.invoke()
            return [self.model.get_tensor(self.out_details[0]['index'])]
        elif self.model is not None and not CFG.use_tflite and HAS_ONNX:
            inp = np.expand_dims(np.transpose(blob, (2, 0, 1)), 0)
            return self.model.run(None, {self.inp_name: inp})
        else:
            # Synthetic output: simulate YOLOv8n output shape [1, 84, 8400]
            time.sleep(0.25)  # Simulate ~250ms inference on Pi Zero 2
            return [np.random.randn(1, 84, 8400).astype(np.float32) * 0.01]

    def _postprocess(self, outputs, scale, pad_x, pad_y, orig_w, orig_h):
        """YOLOv8 postprocessing: NMS + coordinate transform."""
        preds = outputs[0]
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] == 84:
            # Shape [84, N] — standard YOLOv8 format
            boxes = preds[:4, :]
            scores = preds[4:, :]
        else:
            # Shape [N, 84] — transposed
            preds = preds.T
            boxes = preds[:4, :]
            scores = preds[4:, :]

        bird_class = min(CFG.bird_class_id, scores.shape[0] - 1)
        bird = scores[bird_class, :]

        mask = bird > CFG.confidence_threshold
        if not np.any(mask):
            return []

        bird_filtered = bird[mask]
        bx = boxes[:, mask]

        raw_cx, raw_cy, raw_w, raw_h = bx[0], bx[1], bx[2], bx[3]

        # Check coord format
        max_coord = float(max(np.max(np.abs(raw_cx)), np.max(np.abs(raw_cy)),
                              np.max(raw_w), np.max(raw_h)))
        if max_coord <= 1.0:
            raw_cx = raw_cx * CFG.input_size
            raw_cy = raw_cy * CFG.input_size
            raw_w = raw_w * CFG.input_size
            raw_h = raw_h * CFG.input_size

        cx = (raw_cx - pad_x) / scale
        cy = (raw_cy - pad_y) / scale
        w = raw_w / scale
        h = raw_h / scale

        nms_boxes = np.stack([cx - w/2, cy - h/2, w, h], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, bird_filtered.tolist(),
                                    CFG.confidence_threshold,
                                    CFG.nms_iou_threshold)
        dets = []
        if len(indices) > 0:
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                dets.append({
                    "cx": float(cx[idx]) / orig_w,
                    "cy": float(cy[idx]) / orig_h,
                    "w": float(w[idx]) / orig_w,
                    "h": float(h[idx]) / orig_h,
                    "conf": float(bird_filtered[idx]),
                })
        return dets

    def _decision_fusion(self, bearing, dets):
        """Fuse audio bearing with visual detection to compute servo angles."""
        if dets:
            best = max(dets, key=lambda d: d["conf"])
            target_pan = clamp(
                CFG.pan_centre + (-best["cx"] + 0.5) * CFG.cam_hfov_deg,
                CFG.pan_min, CFG.pan_max)
            target_tilt = clamp(
                CFG.smart_patrol_tilt + (best["cy"] - 0.5) * CFG.cam_vfov_deg,
                CFG.tilt_min, CFG.tilt_max)
        else:
            # Fallback to audio bearing
            target_pan = clamp(bearing, CFG.pan_min, CFG.pan_max)
            target_tilt = CFG.smart_patrol_tilt
        return target_pan, target_tilt

    def _simulate_serial_send(self, pan, tilt):
        """Simulate serial command send to Pico W."""
        cmd = f"PAN{int(pan)},TILT{int(tilt)},LASER1\n"
        _ = cmd.encode("ascii")
        # Actual serial write takes ~1-2ms over USB CDC

    # ---- Report Generation ----

    def generate_report(self):
        """Generate comprehensive profiling report and visualisations."""
        print("=" * 70)
        print("  PROFILING REPORT")
        print("=" * 70)

        self._print_latency_summary()
        self._print_timing_budget()
        self._print_cpu_vs_wall()
        self._print_memory_summary()
        self._save_raw_csv()

        if HAS_MATPLOTLIB:
            self._plot_latency_histogram()
            self._plot_latency_boxplot()
            self._plot_pipeline_timing()
            self._plot_memory_timeline()

        self._save_full_report()

        print(f"\n[OUTPUT] All results saved to: {self.output_dir}/")

    def _get_stage_arrays(self):
        """Extract per-stage latency arrays."""
        stages = {
            "Audio Capture":    [r.audio_capture_ms for r in self.results],
            "Energy Check":     [r.energy_check_ms for r in self.results],
            "TDOA (GCC-PHAT)":  [r.tdoa_ms for r in self.results],
            "Frame Grab":       [r.frame_grab_ms for r in self.results],
            "Preprocess":       [r.preprocess_ms for r in self.results],
            "Inference":        [r.inference_ms for r in self.results],
            "Postprocess":      [r.postprocess_ms for r in self.results],
            "Decision Fusion":  [r.decision_fusion_ms for r in self.results],
            "Serial Command":   [r.serial_command_ms for r in self.results],
            "End-to-End":       [r.total_e2e_ms for r in self.results],
        }
        return {k: np.array(v) for k, v in stages.items()}

    def _print_latency_summary(self):
        """Print latency statistics table."""
        stages = self._get_stage_arrays()

        print(f"\n--- Latency Summary (n={self.iterations}) ---\n")
        header = f"{'Stage':<22} {'Mean':>8} {'Median':>8} {'p95':>8} {'p99':>8} {'Min':>8} {'Max':>8} {'StdDev':>8}"
        print(header)
        print("-" * len(header))

        for name, data in stages.items():
            print(f"{name:<22} "
                  f"{np.mean(data):8.2f} "
                  f"{np.median(data):8.2f} "
                  f"{np.percentile(data, 95):8.2f} "
                  f"{np.percentile(data, 99):8.2f} "
                  f"{np.min(data):8.2f} "
                  f"{np.max(data):8.2f} "
                  f"{np.std(data):8.2f}")
        print()

    def _print_timing_budget(self):
        """Print end-to-end timing budget vs 500ms target."""
        e2e = np.array([r.total_e2e_ms for r in self.results])
        target = 500.0

        print(f"--- Timing Budget Verification (Target: <{target:.0f} ms) ---\n")

        stages_for_budget = {
            "Audio buffer fill":       [r.audio_capture_ms for r in self.results],
            "TDOA computation":        [r.tdoa_ms for r in self.results],
            "Camera frame grab":       [r.frame_grab_ms for r in self.results],
            "Bird detection inference": [r.inference_ms for r in self.results],
            "Decision + serial send":  [r.decision_fusion_ms + r.serial_command_ms
                                        for r in self.results],
        }

        cumulative = 0.0
        print(f"  {'Pipeline Stage':<30} {'Est. (ms)':>10} {'Measured (ms)':>14} {'Cumulative':>12}")
        print(f"  {'-' * 30} {'-' * 10} {'-' * 14} {'-' * 12}")

        # Estimated vs measured (from design report)
        estimates = {
            "Audio buffer fill": 50,
            "TDOA computation": 5,
            "Camera frame grab": 30,
            "Bird detection inference": "200-400",
            "Decision + serial send": 5,
        }

        for name, data in stages_for_budget.items():
            mean_ms = np.mean(data)
            cumulative += mean_ms
            est = estimates.get(name, "N/A")
            print(f"  {name:<30} {str(est):>10} {mean_ms:14.1f} {cumulative:12.1f}")

        print(f"\n  {'TOTAL (Pi side):':<30} {'290-490':>10} {cumulative:14.1f}")
        print(f"  {'+ Pico W actuation (est.):':<30} {'10-50':>10} {'10-50':>14}")
        print(f"  {'= Estimated E2E:':<30} {'300-540':>10} "
              f"{cumulative + 30:14.1f}")

        pct_under = np.mean(e2e < target) * 100
        print(f"\n  Iterations within {target:.0f} ms target: "
              f"{pct_under:.1f}% ({int(pct_under * self.iterations / 100)}/{self.iterations})")
        print(f"  Mean E2E: {np.mean(e2e):.1f} ms | "
              f"p95: {np.percentile(e2e, 95):.1f} ms | "
              f"p99: {np.percentile(e2e, 99):.1f} ms\n")

        if np.mean(e2e) < target:
            print(f"  ✓ PASS: Mean E2E ({np.mean(e2e):.1f} ms) < {target:.0f} ms target")
        else:
            print(f"  ✗ FAIL: Mean E2E ({np.mean(e2e):.1f} ms) >= {target:.0f} ms target")

        if np.percentile(e2e, 95) < target:
            print(f"  ✓ PASS: p95 ({np.percentile(e2e, 95):.1f} ms) < {target:.0f} ms target")
        else:
            print(f"  ✗ FAIL: p95 ({np.percentile(e2e, 95):.1f} ms) >= {target:.0f} ms target")
        print()

    def _print_cpu_vs_wall(self):
        """Print CPU time vs wall-clock time analysis."""
        wall = np.array([r.wall_time_ms for r in self.results])
        cpu = np.array([r.cpu_time_ms for r in self.results])

        print(f"--- CPU Time vs Wall-Clock Time ---\n")
        print(f"  Wall-clock (mean): {np.mean(wall):8.2f} ms")
        print(f"  CPU time (mean)  : {np.mean(cpu):8.2f} ms")
        ratio = np.mean(cpu) / np.mean(wall) if np.mean(wall) > 0 else 0
        print(f"  CPU/Wall ratio   : {ratio:.3f}")
        print(f"  Interpretation   : ", end="")
        if ratio < 0.5:
            print("I/O bound (significant time spent waiting for camera/serial)")
        elif ratio < 0.9:
            print("Mixed workload (some I/O wait, mostly compute)")
        elif ratio < 1.1:
            print("Single-core compute bound")
        else:
            print(f"Multi-core utilisation ({ratio:.1f}x cores used on average)")
        print()

    def _print_memory_summary(self):
        """Print memory usage statistics."""
        if not HAS_PSUTIL:
            return

        rss_values = [r.rss_mb for r in self.results]
        print(f"--- Memory Usage (RSS) ---\n")
        print(f"  Peak RSS     : {max(rss_values):.1f} MB")
        print(f"  Mean RSS     : {np.mean(rss_values):.1f} MB")
        print(f"  Min RSS      : {min(rss_values):.1f} MB")
        print(f"  RSS range    : {max(rss_values) - min(rss_values):.1f} MB")
        print(f"  RAM budget   : 512 MB (Pi Zero 2 W)")
        print(f"  Utilisation  : {max(rss_values) / 512 * 100:.1f}%\n")

    def _save_raw_csv(self):
        """Save raw latency data for external analysis."""
        path = os.path.join(self.output_dir, "raw_latencies.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "audio_capture_ms", "energy_check_ms",
                "tdoa_ms", "frame_grab_ms", "preprocess_ms",
                "inference_ms", "postprocess_ms", "decision_fusion_ms",
                "serial_command_ms", "total_e2e_ms", "rss_mb",
                "cpu_time_ms", "wall_time_ms"
            ])
            for r in self.results:
                writer.writerow([
                    r.iteration, f"{r.audio_capture_ms:.4f}",
                    f"{r.energy_check_ms:.4f}", f"{r.tdoa_ms:.4f}",
                    f"{r.frame_grab_ms:.4f}", f"{r.preprocess_ms:.4f}",
                    f"{r.inference_ms:.4f}", f"{r.postprocess_ms:.4f}",
                    f"{r.decision_fusion_ms:.4f}", f"{r.serial_command_ms:.4f}",
                    f"{r.total_e2e_ms:.4f}", f"{r.rss_mb:.2f}",
                    f"{r.cpu_time_ms:.4f}", f"{r.wall_time_ms:.4f}"
                ])
        print(f"  Raw data saved to: {path}")

    def _plot_latency_histogram(self):
        """Plot per-stage latency distributions."""
        stages = self._get_stage_arrays()

        # Skip trivial stages for the histogram
        plot_stages = {k: v for k, v in stages.items()
                       if k not in ["Energy Check", "Decision Fusion", "Serial Command"]}

        n = len(plot_stages)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
        if n == 1:
            axes = [axes]

        for ax, (name, data) in zip(axes, plot_stages.items()):
            ax.hist(data, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
            ax.axvline(np.mean(data), color='red', linestyle='--',
                       label=f'Mean: {np.mean(data):.1f} ms')
            ax.axvline(np.percentile(data, 95), color='orange', linestyle='--',
                       label=f'p95: {np.percentile(data, 95):.1f} ms')
            ax.set_title(f'{name} Latency Distribution')
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Count')
            ax.legend()

        plt.tight_layout()
        path = os.path.join(self.output_dir, "latency_histogram.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Histogram saved to: {path}")

    def _plot_latency_boxplot(self):
        """Plot box plots comparing all stages."""
        stages = self._get_stage_arrays()

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        names = list(stages.keys())
        data = [stages[n] for n in names]

        bp = ax.boxplot(data, labels=names, patch_artist=True, vert=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('Latency (ms)')
        ax.set_title('BirdGuard Smart Patrol — Per-Stage Latency Box Plots')
        plt.xticks(rotation=45, ha='right')

        # 500ms target line
        ax.axhline(500, color='red', linestyle='--', alpha=0.5,
                   label='500 ms target')
        ax.legend()

        plt.tight_layout()
        path = os.path.join(self.output_dir, "latency_boxplot.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Boxplot saved to: {path}")

    def _plot_pipeline_timing(self):
        """Plot stacked bar showing pipeline stage breakdown per iteration."""
        stage_names = [
            "Audio Capture", "Energy Check", "TDOA (GCC-PHAT)",
            "Frame Grab", "Preprocess", "Inference",
            "Postprocess", "Decision Fusion", "Serial Command"
        ]
        stages = self._get_stage_arrays()

        # Show a subset of iterations for readability
        n_show = min(30, self.iterations)
        x = np.arange(n_show)

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        bottom = np.zeros(n_show)
        colors = plt.cm.tab10(np.linspace(0, 1, len(stage_names)))

        for name, color in zip(stage_names, colors):
            data = stages[name][:n_show]
            ax.bar(x, data, bottom=bottom, label=name, color=color, width=0.8)
            bottom += data

        ax.axhline(500, color='red', linestyle='--', linewidth=2,
                   label='500 ms target')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('BirdGuard Smart Patrol — Pipeline Timing Breakdown')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()

        path = os.path.join(self.output_dir, "pipeline_timing.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Pipeline timing saved to: {path}")

    def _plot_memory_timeline(self):
        """Plot RSS memory over time."""
        if not self.memory_samples or len(self.memory_samples) < 2:
            return

        t0 = self.memory_samples[0][0]
        times = [(t - t0) for t, _ in self.memory_samples]
        rss = [r for _, r in self.memory_samples]

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(times, rss, 'b-o', markersize=4)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RSS (MB)')
        ax.set_title('BirdGuard Smart Patrol — Memory Usage Over Time')
        ax.axhline(512, color='red', linestyle='--', alpha=0.5,
                   label='Pi Zero 2 RAM (512 MB)')
        ax.legend()
        plt.tight_layout()

        path = os.path.join(self.output_dir, "memory_timeline.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Memory timeline saved to: {path}")

    def _save_full_report(self):
        """Save the complete text report."""
        import io
        buf = io.StringIO()

        # Redirect prints to buffer
        old_stdout = sys.stdout
        sys.stdout = buf

        print("=" * 70)
        print("BirdGuard Smart Patrol — Profiling Report")
        print(f"INF2009 Edge Computing Lab — Group 31")
        print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Iterations: {self.iterations}")
        print("=" * 70)

        self._print_latency_summary()
        self._print_timing_budget()
        self._print_cpu_vs_wall()
        self._print_memory_summary()

        print("\n--- External Profiling Commands ---\n")
        print("# /usr/bin/time (resource consumption):")
        print("/usr/bin/time -v python3 profile_smart_patrol.py --iterations 50\n")
        print("# cProfile (function-level CPU breakdown):")
        print("python -m cProfile -s tottime profile_smart_patrol.py --iterations 50\n")
        print("# perf stat (hardware counters: IPC, cache misses, context switches):")
        print("perf stat -e cycles,instructions,cache-misses,cs,migrations -- \\")
        print("    python3 profile_smart_patrol.py --iterations 50\n")
        print("# perf record (CPU hotspot sampling):")
        print("perf record -F 99 -g -- python3 profile_smart_patrol.py --iterations 50")
        print("perf report\n")
        print("# Single-core vs multi-core:")
        print("taskset -c 0 python3 profile_smart_patrol.py --iterations 50")
        print("taskset -c 0-3 python3 profile_smart_patrol.py --iterations 50\n")
        print("# Real-time priority:")
        print("sudo chrt --rr 50 python3 profile_smart_patrol.py --iterations 50\n")
        print("# pidstat (live monitoring for long runs):")
        print("pidstat -rudw -p $(pgrep -f profile_smart_patrol) 1\n")
        print("# MQTT End-to-End latency measurement:")
        print("mosquitto_sub -t 'birdguard/telemetry' -v | ts '%s%N' >> mqtt_log.txt\n")

        sys.stdout = old_stdout
        report_text = buf.getvalue()

        path = os.path.join(self.output_dir, "latency_summary.txt")
        with open(path, 'w') as f:
            f.write(report_text)
        print(f"  Full report saved to: {path}")

    def cleanup(self):
        """Release resources."""
        self.stop_event.set()
        if self.frame_grabber is not None:
            self.frame_grabber.join(timeout=3)

        if self.camera is not None:
            self.camera.release()

        # Reset shared state (particularly important if profile script is run multiple times)
        shared.shared_camera = None
        shared.camera_healthy = False


# ============================================================================
# CLI Script for External Profiling Tools
# ============================================================================

def create_helper_scripts(output_dir):
    """Create helper shell scripts for running external profiling tools."""

    # Script: run_all_profiles.sh
    script = os.path.join(output_dir, "run_all_profiles.sh")
    with open(script, 'w') as f:
        f.write("""#!/bin/bash
# BirdGuard Smart Patrol — Complete Profiling Suite
# INF2009 Edge Computing Lab
# Run this on the Raspberry Pi Zero 2 W

SCRIPT="profile_smart_patrol.py"
ITERS=50
OUTDIR="profiling_results"

echo "========================================="
echo " BirdGuard Profiling Suite"
echo "========================================="

# 1. Basic profiling run
echo ""
echo "[1/6] Running basic profiling ($ITERS iterations)..."
python3 $SCRIPT --iterations $ITERS
echo ""

# 2. /usr/bin/time (resource consumption)
echo "[2/6] /usr/bin/time profiling..."
/usr/bin/time -v python3 $SCRIPT --iterations $ITERS 2> $OUTDIR/time_verbose.txt
echo "  Saved to $OUTDIR/time_verbose.txt"

# 3. cProfile (function-level breakdown)
echo "[3/6] cProfile profiling..."
python3 -m cProfile -s tottime $SCRIPT --iterations $ITERS > $OUTDIR/cprofile_output.txt 2>&1
echo "  Saved to $OUTDIR/cprofile_output.txt"

# 4. perf stat (hardware counters)
echo "[4/6] perf stat (hardware counters)..."
perf stat -e cycles,instructions,cache-misses,cs,migrations -- \\
    python3 $SCRIPT --iterations $ITERS 2> $OUTDIR/perf_stat.txt
echo "  Saved to $OUTDIR/perf_stat.txt"

# 5. Single-core run (taskset)
echo "[5/6] Single-core profiling (Core 0 only)..."
taskset -c 0 python3 $SCRIPT --iterations $ITERS 2>&1 | tee $OUTDIR/singlecore_output.txt
echo ""

# 6. Multi-core run (all 4 cores)
echo "[6/6] Multi-core profiling (All cores)..."
taskset -c 0-3 python3 $SCRIPT --iterations $ITERS 2>&1 | tee $OUTDIR/multicore_output.txt
echo ""

echo "========================================="
echo " Profiling complete! Results in $OUTDIR/"
echo "========================================="
""")
    os.chmod(script, 0o755)

    # Script: run_mqtt_e2e.sh
    mqtt_script = os.path.join(output_dir, "run_mqtt_e2e.sh")
    with open(mqtt_script, 'w') as f:
        f.write("""#!/bin/bash
# BirdGuard — MQTT End-to-End Latency Measurement
# INF2009 Edge Computing Lab

echo "Starting MQTT E2E latency capture..."
echo "Press Ctrl+C to stop."
echo ""
echo "In another terminal, start birdguard.py to generate events."
echo ""

# Subscribe to telemetry and timestamp each message
mosquitto_sub -t "birdguard/telemetry" -v | ts '%s%N' >> mqtt_e2e_log.txt
""")
    os.chmod(mqtt_script, 0o755)

    print(f"  Helper scripts saved to: {output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BirdGuard Smart Patrol — Pipeline Profiler (INF2009)")
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of profiling iterations (default: 50)')
    args = parser.parse_args()

    profiler = SmartPatrolProfiler(iterations=args.iterations)

    try:
        profiler.setup()
        profiler.run()
        profiler.generate_report()
        create_helper_scripts(profiler.output_dir)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Generating partial report...")
        if profiler.results:
            profiler.generate_report()
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        traceback.print_exc()
    finally:
        profiler.cleanup()


if __name__ == "__main__":
    main()
