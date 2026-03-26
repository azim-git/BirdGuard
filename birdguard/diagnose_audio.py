#!/usr/bin/env python3
"""
BirdGuard — Audio Wake Diagnostic Tool
========================================
Run this on the Pi to diagnose why audio wake detection isn't firing.

Tests each stage of the audio pipeline independently:
  1. ALSA device enumeration
  2. Raw capture test (do bytes come in at all?)
  3. RMS energy levels (what values are you actually getting?)
  4. Energy threshold comparison
  5. Cooldown & turret_moving gate checks
  6. Full AudioMonitor simulation

Usage:
    python3 diagnose_audio.py              # run all tests
    python3 diagnose_audio.py --listen 10  # listen for 10 seconds and plot RMS
"""

import argparse
import struct
import subprocess
import sys
import time
import os
import numpy as np

# ============================================================
# Colour helpers for terminal output
# ============================================================
def green(s):  return f"\033[92m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"
def OK():      return green("OK")
def FAIL():    return red("FAIL")
def WARN():    return yellow("WARN")

# ============================================================
# Test 1: ALSA Device Enumeration
# ============================================================
def test_alsa_devices():
    print(bold("\n═══ TEST 1: ALSA Device Enumeration ═══"))
    
    # arecord -l
    print("\n[1a] arecord -l (list capture devices):")
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
        output = result.stdout + result.stderr
        print(output)
        if "no soundcards found" in output.lower() or not result.stdout.strip():
            print(f"  {FAIL()} — No capture devices found!")
            print(f"       Is the ReSpeaker HAT seated properly on the GPIO header?")
            print(f"       Check: sudo dmesg | grep -i 'wm8960\\|seeed\\|sound\\|audio'")
            return False
        if "seeed" in output.lower() or "wm8960" in output.lower():
            print(f"  {OK()} — ReSpeaker / WM8960 codec detected")
        else:
            print(f"  {WARN()} — Sound device found but doesn't look like ReSpeaker")
    except FileNotFoundError:
        print(f"  {FAIL()} — arecord not found. Install: sudo apt install alsa-utils")
        return False
    except subprocess.TimeoutExpired:
        print(f"  {FAIL()} — arecord -l timed out")
        return False

    # Check card number
    print("\n[1b] Checking card number assignment:")
    try:
        result = subprocess.run(["cat", "/proc/asound/cards"],
                                capture_output=True, text=True, timeout=5)
        print(result.stdout)
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if "wm8960" in line.lower() or "seeed" in line.lower():
                card_num = line.strip().split()[0]
                print(f"  {OK()} — ReSpeaker is card {card_num}")
                if card_num != "0":
                    print(f"  {WARN()} — Card is NOT card 0!")
                    print(f"       BirdGuard uses audio_device='plughw:0,0'")
                    print(f"       Fix: change audio_device to 'plughw:{card_num},0'")
                    print(f"       Or rearrange dtoverlay order in /boot/config.txt")
    except Exception as e:
        print(f"  {WARN()} — Could not read /proc/asound/cards: {e}")

    # Check dtoverlay
    print("\n[1c] Checking /boot/config.txt for seeed overlay:")
    boot_configs = ["/boot/config.txt", "/boot/firmware/config.txt"]
    found_overlay = False
    for cfg in boot_configs:
        if os.path.exists(cfg):
            try:
                with open(cfg, "r") as f:
                    content = f.read()
                if "seeed" in content.lower() or "wm8960" in content.lower():
                    for line in content.split("\n"):
                        if "seeed" in line.lower() or "wm8960" in line.lower():
                            if not line.strip().startswith("#"):
                                print(f"  {OK()} — Found: {line.strip()} (in {cfg})")
                                found_overlay = True
                else:
                    print(f"  {WARN()} — No seeed/wm8960 overlay in {cfg}")
            except PermissionError:
                print(f"  {WARN()} — Cannot read {cfg} (try sudo)")
    if not found_overlay:
        print(f"  {FAIL()} — dtoverlay for ReSpeaker not found!")
        print(f"       Add to config.txt: dtoverlay=seeed-2mic-voicecard")

    return True


# ============================================================
# Test 2: ALSA Mixer Settings
# ============================================================
def test_mixer_settings():
    print(bold("\n═══ TEST 2: ALSA Mixer Settings ═══"))
    
    print("\n[2a] Current mixer state (amixer -c 0):")
    try:
        result = subprocess.run(["amixer", "-c", "0"], capture_output=True, text=True, timeout=5)
        output = result.stdout
        
        # Parse and check critical controls
        critical_controls = {
            "Capture": {"expected_min": 20, "desc": "Master capture volume"},
            "ADC PCM": {"expected_min": 100, "desc": "ADC digital gain"},
        }
        
        lines = output.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            for ctrl_name, info in critical_controls.items():
                if f"Simple mixer control '{ctrl_name}'" in line:
                    # Look at next few lines for values
                    for j in range(i+1, min(i+5, len(lines))):
                        subline = lines[j]
                        if "Playback" in subline or "Capture" in subline or "Mono" in subline:
                            print(f"  {ctrl_name}: {subline.strip()}")
                            # Check if muted
                            if "[off]" in subline:
                                print(f"    {FAIL()} — {ctrl_name} is MUTED!")
                                print(f"    Fix: amixer -c 0 set '{ctrl_name}' cap")
                            # Check volume
                            if "[" in subline and "%" in subline:
                                try:
                                    pct = int(subline.split("[")[1].split("%")[0])
                                    if pct < info["expected_min"]:
                                        print(f"    {WARN()} — Volume is only {pct}%")
                                except (ValueError, IndexError):
                                    pass
            i += 1
        
        # Check boost mixers
        boost_keywords = ["Boost", "LINPUT", "RINPUT"]
        print(f"\n[2b] Boost mixer controls:")
        found_boost = False
        for line in lines:
            for kw in boost_keywords:
                if kw in line and "Simple mixer control" in line:
                    found_boost = True
                    # Get next line with value
                    idx = lines.index(line)
                    for j in range(idx+1, min(idx+4, len(lines))):
                        if "Mono" in lines[j] or "Playback" in lines[j]:
                            print(f"  {line.strip()}: {lines[j].strip()}")
                            if "[off]" in lines[j]:
                                print(f"    {WARN()} — Boost is OFF")
        if not found_boost:
            print(f"  {WARN()} — No boost controls found (may be normal)")
            
    except FileNotFoundError:
        print(f"  {FAIL()} — amixer not found")
        return False
    except subprocess.TimeoutExpired:
        print(f"  {FAIL()} — amixer timed out")
        return False

    # Apply recommended settings
    print(f"\n[2c] Applying recommended mixer settings:")
    mixer_cmds = [
        ["amixer", "-c", "0", "set", "Capture", "63"],
        ["amixer", "-c", "0", "set", "Capture", "cap"],  # unmute
        ["amixer", "-c", "0", "set", "ADC PCM", "235"],
        ["amixer", "-c", "0", "set", "Left Input Boost Mixer LINPUT1", "1"],
        ["amixer", "-c", "0", "set", "Right Input Boost Mixer RINPUT1", "1"],
        ["amixer", "-c", "0", "set", "Left Input Mixer Boost", "on"],
        ["amixer", "-c", "0", "set", "Right Input Mixer Boost", "on"],
        ["amixer", "-c", "0", "set", "Left Boost Mixer LINPUT1", "on"],
        ["amixer", "-c", "0", "set", "Right Boost Mixer RINPUT1", "on"],
    ]
    for cmd in mixer_cmds:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            ctrl = " ".join(cmd[4:])
            if result.returncode == 0:
                print(f"  {OK()} {ctrl}")
            else:
                # Control may not exist — that's fine
                stderr = result.stderr.strip()
                if "Unable to find" in stderr or "Invalid" in stderr:
                    pass  # silently skip non-existent controls
                else:
                    print(f"  {WARN()} {ctrl}: {stderr}")
        except Exception:
            pass

    return True


# ============================================================
# Test 3: Raw Capture Test
# ============================================================
def test_raw_capture():
    print(bold("\n═══ TEST 3: Raw ALSA Capture Test ═══"))
    
    try:
        import alsaaudio
    except ImportError:
        print(f"  {FAIL()} — pyalsaaudio not installed!")
        print(f"       Install: pip install pyalsaaudio")
        return False

    devices_to_try = [
        ("plughw:0,0", "BirdGuard default"),
        ("plughw:1,0", "Alternative card 1"),
        ("default",    "ALSA default"),
        ("hw:0,0",     "Direct hardware"),
    ]

    for device, desc in devices_to_try:
        print(f"\n[3] Trying device '{device}' ({desc}):")
        try:
            stream = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL,
                device=device, channels=2,
                rate=16000, format=alsaaudio.PCM_FORMAT_S16_LE,
                periodsize=1024,
            )
            
            # Read a few buffers
            total_bytes = 0
            total_nonzero = 0
            rms_values = []
            
            for i in range(20):  # ~1.3 seconds at 16kHz
                length, data = stream.read()
                if length > 0 and data:
                    total_bytes += len(data)
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    
                    if len(samples) >= 2:
                        ch0 = samples[0::2]
                        ch1 = samples[1::2]
                        rms0 = float(np.sqrt(np.mean(ch0 ** 2)))
                        rms1 = float(np.sqrt(np.mean(ch1 ** 2)))
                        rms_values.append((rms0, rms1))
                        
                        nonzero = np.count_nonzero(samples)
                        total_nonzero += nonzero
                elif length == 0:
                    rms_values.append((0, 0))
                elif length < 0:
                    print(f"    Read error: length={length}")

            stream.close()
            
            if total_bytes == 0:
                print(f"  {FAIL()} — No data received from {device}")
                continue
            
            print(f"  Total bytes read: {total_bytes}")
            print(f"  Non-zero samples: {total_nonzero}")
            
            if rms_values:
                rms0_arr = [r[0] for r in rms_values]
                rms1_arr = [r[1] for r in rms_values]
                
                print(f"  Ch0 RMS — min: {min(rms0_arr):.1f}  "
                      f"max: {max(rms0_arr):.1f}  mean: {np.mean(rms0_arr):.1f}")
                print(f"  Ch1 RMS — min: {min(rms1_arr):.1f}  "
                      f"max: {max(rms1_arr):.1f}  mean: {np.mean(rms1_arr):.1f}")
                
                max_rms = max(max(rms0_arr), max(rms1_arr))
                
                if total_nonzero == 0:
                    print(f"\n  {FAIL()} — ALL SAMPLES ARE ZERO!")
                    print(f"       The device opened but produced only silence.")
                    print(f"       Causes:")
                    print(f"         - Capture channel is muted (run Test 2)")
                    print(f"         - Wrong device node (metadata vs capture)")
                    print(f"         - Codec not initialised (check dmesg)")
                    print(f"         - HAT not seated properly")
                elif max_rms < 10:
                    print(f"\n  {WARN()} — Signal is very weak (max RMS: {max_rms:.1f})")
                    print(f"       This is near-silence. Either:")
                    print(f"         - Gain is too low (increase ADC PCM and Capture)")
                    print(f"         - Wrong device node")
                    print(f"         - Microphones are damaged")
                elif max_rms < 100:
                    print(f"\n  {WARN()} — Signal present but low (max RMS: {max_rms:.1f})")
                    print(f"       BirdGuard threshold is 500.0 — tapping won't trigger it.")
                    print(f"       Try: increase ADC PCM gain, or lower energy_threshold")
                elif max_rms < 500:
                    print(f"\n  {WARN()} — Signal present (max RMS: {max_rms:.1f})")
                    print(f"       BirdGuard threshold is 500.0 — barely reaching it.")
                    print(f"       Recommend: lower energy_threshold to {max_rms * 0.5:.0f}")
                else:
                    print(f"\n  {OK()} — Good signal level (max RMS: {max_rms:.1f})")
                    print(f"       Above BirdGuard threshold of 500.0")
                
                return True
            
        except alsaaudio.ALSAAudioError as e:
            print(f"  {FAIL()} — ALSA error: {e}")
        except Exception as e:
            print(f"  {FAIL()} — Error: {e}")

    print(f"\n  {FAIL()} — No working audio device found!")
    return False


# ============================================================
# Test 4: BirdGuard Audio Pipeline Simulation  
# ============================================================
def test_birdguard_pipeline():
    print(bold("\n═══ TEST 4: BirdGuard Audio Pipeline Simulation ═══"))
    print("  Simulating exactly what AudioMonitor._process() does")
    print("  Tap the microphone repeatedly during this test!\n")

    try:
        import alsaaudio
    except ImportError:
        print(f"  {FAIL()} — pyalsaaudio not installed")
        return False

    # Try to open with BirdGuard's exact settings
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from shared import CFG
        device = CFG.audio_device
        channels = CFG.audio_channels
        rate = CFG.sample_rate
        chunk = CFG.audio_chunk
        threshold = CFG.energy_threshold
        cooldown = CFG.audio_cooldown
    except ImportError:
        device = "plughw:0,0"
        channels = 2
        rate = 16000
        chunk = 1024
        threshold = 500.0
        cooldown = 2.0

    print(f"  Config: device={device} ch={channels} rate={rate} chunk={chunk}")
    print(f"  Threshold: {threshold}  Cooldown: {cooldown}s")
    print(f"  Listening for 8 seconds...\n")

    try:
        stream = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL,
            device=device, channels=channels,
            rate=rate, format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=chunk,
        )
    except alsaaudio.ALSAAudioError as e:
        print(f"  {FAIL()} — Cannot open {device}: {e}")
        return False

    t_start = time.perf_counter()
    last_wake = 0.0
    wake_count = 0
    max_rms_seen = 0.0
    buffer_count = 0
    zero_buffer_count = 0
    
    # Diagnostic counters
    cooldown_blocked = 0
    below_threshold = 0
    
    rms_history = []

    while time.perf_counter() - t_start < 8.0:
        try:
            length, data = stream.read()
        except Exception as e:
            print(f"  Read error: {e}")
            continue

        if length <= 0 or not data:
            continue

        buffer_count += 1
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        
        if len(samples) < 4:
            continue

        ch0 = samples[0::2]
        ch1 = samples[1::2]

        # Check for all-zero
        if np.count_nonzero(samples) == 0:
            zero_buffer_count += 1
            continue

        rms = float(np.sqrt(np.mean(ch0 ** 2)))
        rms_history.append(rms)
        max_rms_seen = max(max_rms_seen, rms)
        
        now = time.perf_counter()

        # Check cooldown gate (same as BirdGuard)
        if now - last_wake < cooldown:
            if rms >= threshold:
                cooldown_blocked += 1
            continue

        # Check threshold
        if rms < threshold:
            below_threshold += 1
            continue

        # WAKE EVENT!
        wake_count += 1
        last_wake = now
        elapsed = now - t_start
        print(f"  [{elapsed:5.1f}s] *** WAKE #{wake_count} *** "
              f"RMS={rms:.0f} (threshold={threshold})")

    stream.close()

    # Summary
    print(f"\n  --- Results ---")
    print(f"  Buffers read     : {buffer_count}")
    print(f"  All-zero buffers : {zero_buffer_count}")
    if zero_buffer_count == buffer_count:
        print(f"  {FAIL()} — Every single buffer was all zeros!")
        print(f"       The microphone is not producing any signal.")
        return False

    print(f"  Max RMS seen     : {max_rms_seen:.1f}")
    print(f"  Wake events      : {wake_count}")
    print(f"  Blocked by cooldown : {cooldown_blocked}")
    print(f"  Below threshold  : {below_threshold}")

    if rms_history:
        arr = np.array(rms_history)
        print(f"  RMS — mean: {np.mean(arr):.1f}  "
              f"p50: {np.median(arr):.1f}  "
              f"p95: {np.percentile(arr, 95):.1f}  "
              f"max: {np.max(arr):.1f}")

    print()
    if wake_count == 0:
        if max_rms_seen == 0:
            print(f"  {FAIL()} — No audio signal at all. Check hardware.")
        elif max_rms_seen < threshold:
            print(f"  {FAIL()} — Audio signal present but max RMS ({max_rms_seen:.0f}) "
                  f"never reached threshold ({threshold})")
            suggested = max(10, max_rms_seen * 0.6)
            print(f"  {bold('FIX OPTIONS:')}")
            print(f"    1. Lower threshold:  energy_threshold = {suggested:.0f}")
            print(f"       (via MQTT: mosquitto_pub -t birdguard/config "
                  f"-m '{{\"energy_threshold\": {suggested:.0f}}}')")
            print(f"    2. Increase gain:    amixer -c 0 set 'ADC PCM' 235")
            print(f"    3. Increase capture: amixer -c 0 set 'Capture' 63")
        else:
            print(f"  {WARN()} — Signal exceeded threshold but was caught by cooldown")
            print(f"       Cooldown = {cooldown}s. If you tapped quickly, only first tap "
                  f"triggers.")
    else:
        print(f"  {OK()} — Audio wake is working! ({wake_count} events detected)")

    return wake_count > 0


# ============================================================
# Test 5: Turret Moving Gate Check
# ============================================================
def test_turret_moving_gate():
    print(bold("\n═══ TEST 5: turret_moving Gate Check ═══"))
    print("  In BirdGuard, AudioMonitor._process() ignores audio when")
    print("  turret_moving.is_set() is True.\n")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from shared import turret_moving, CFG
        
        if turret_moving.is_set():
            print(f"  {FAIL()} — turret_moving is SET!")
            print(f"       Audio processing will be completely skipped.")
            print(f"       This means the turret never finished a move and")
            print(f"       turret_moving.clear() was never called.")
            print(f"       Check smooth_move() and deterrent_sweep() in shared.py")
        else:
            print(f"  {OK()} — turret_moving is NOT set (audio will be processed)")
        
        print(f"\n  Related config:")
        print(f"    audio_cooldown = {CFG.audio_cooldown}s")
        print(f"    (After each wake event, audio is ignored for {CFG.audio_cooldown}s)")
    except ImportError:
        print(f"  {WARN()} — Cannot import shared module (run from BirdGuard directory)")


# ============================================================
# Test 6: arecord Quick Test
# ============================================================
def test_arecord():
    print(bold("\n═══ TEST 6: arecord Direct Capture Test ═══"))
    print("  Recording 3 seconds with arecord to verify hardware path\n")
    
    test_file = "/tmp/birdguard_audio_test.wav"
    
    # Try different device strings
    devices = ["plughw:0,0", "plughw:1,0", "hw:0,0", "default"]
    
    for device in devices:
        cmd = [
            "arecord", "-D", device, "-d", "3",
            "-r", "16000", "-f", "S16_LE", "-c", "2",
            test_file
        ]
        print(f"  Trying: arecord -D {device} ...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and os.path.exists(test_file):
                size = os.path.getsize(test_file)
                print(f"    {OK()} — Recorded {size} bytes to {test_file}")
                
                # Check if it's just silence
                try:
                    import wave
                    with wave.open(test_file, 'rb') as wf:
                        frames = wf.readframes(wf.getnframes())
                        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                        rms = float(np.sqrt(np.mean(samples ** 2)))
                        max_val = float(np.max(np.abs(samples)))
                        print(f"    RMS: {rms:.1f}  Max sample: {max_val:.0f}")
                        if max_val == 0:
                            print(f"    {FAIL()} — File is complete silence!")
                        elif rms < 50:
                            print(f"    {WARN()} — Very quiet recording")
                        else:
                            print(f"    {OK()} — Audio captured successfully!")
                            print(f"    Play it back: aplay {test_file}")
                            return True
                except Exception:
                    pass
            else:
                stderr = result.stderr.strip()
                if stderr:
                    print(f"    {FAIL()} — {stderr[:100]}")
        except subprocess.TimeoutExpired:
            print(f"    {FAIL()} — Timed out")
        except FileNotFoundError:
            print(f"    {FAIL()} — arecord not found")
            return False

    return False


# ============================================================
# Test 7: Continuous RMS Monitor
# ============================================================
def listen_continuous(duration=10):
    print(bold(f"\n═══ Continuous RMS Monitor ({duration}s) ═══"))
    print("  Showing real-time RMS levels. Tap/clap near the mic!\n")
    print(f"  {'Time':>6}  {'Ch0 RMS':>9}  {'Ch1 RMS':>9}  {'Bar (Ch0)':40}  Status")
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*40}  {'─'*10}")

    try:
        import alsaaudio
    except ImportError:
        print(f"  {FAIL()} — pyalsaaudio not installed")
        return

    try:
        from shared import CFG
        device = CFG.audio_device
        threshold = CFG.energy_threshold
    except ImportError:
        device = "plughw:0,0"
        threshold = 500.0

    try:
        stream = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL,
            device=device, channels=2,
            rate=16000, format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=1024,
        )
    except Exception as e:
        print(f"  {FAIL()} — Cannot open {device}: {e}")
        return

    t_start = time.perf_counter()
    max_rms = 0

    while time.perf_counter() - t_start < duration:
        try:
            length, data = stream.read()
        except Exception:
            continue

        if length <= 0 or not data:
            continue

        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        if len(samples) < 4:
            continue

        ch0 = samples[0::2]
        ch1 = samples[1::2]
        rms0 = float(np.sqrt(np.mean(ch0 ** 2)))
        rms1 = float(np.sqrt(np.mean(ch1 ** 2)))
        max_rms = max(max_rms, rms0)

        elapsed = time.perf_counter() - t_start

        # Visual bar (scale: 0-2000 RMS mapped to 40 chars)
        bar_len = min(40, int(rms0 / 50))
        bar = "█" * bar_len

        if rms0 >= threshold:
            status = green("WAKE!")
        elif rms0 >= threshold * 0.5:
            status = yellow("close")
        elif rms0 > 10:
            status = "quiet"
        else:
            status = red("silent")

        print(f"\r  {elapsed:5.1f}s  {rms0:9.1f}  {rms1:9.1f}  {bar:40}  {status}",
              end="", flush=True)

    stream.close()
    print(f"\n\n  Max RMS seen: {max_rms:.1f}")
    print(f"  Threshold   : {threshold}")
    if max_rms < threshold:
        ratio = max_rms / threshold if threshold > 0 else 0
        print(f"  Gap         : {ratio:.1%} of threshold")
        print(f"\n  {bold('RECOMMENDATION:')} Lower energy_threshold to {max(10, max_rms * 0.5):.0f}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="BirdGuard Audio Diagnostic")
    parser.add_argument("--listen", type=int, default=0, metavar="SECONDS",
                        help="Continuous RMS monitor mode (e.g. --listen 10)")
    parser.add_argument("--skip-arecord", action="store_true",
                        help="Skip arecord test")
    args = parser.parse_args()

    print(bold("╔══════════════════════════════════════════════════════╗"))
    print(bold("║     BirdGuard — Audio Wake Diagnostic Tool          ║"))
    print(bold("╚══════════════════════════════════════════════════════╝"))

    if args.listen > 0:
        listen_continuous(args.listen)
        return

    # Run all tests
    test_alsa_devices()
    test_mixer_settings()
    if not args.skip_arecord:
        test_arecord()
    test_raw_capture()
    test_turret_moving_gate()
    test_birdguard_pipeline()

    print(bold("\n═══ SUMMARY ═══"))
    print("""
  Most common causes of audio wake not firing (in order of likelihood):

  1. GAIN TOO LOW — The ReSpeaker WM8960 codec defaults are very quiet.
     Fix: amixer -c 0 set 'ADC PCM' 235
          amixer -c 0 set 'Capture' 63
          amixer -c 0 set 'Capture' cap   (unmute!)

  2. THRESHOLD TOO HIGH — energy_threshold=500 may be too high for your
     environment. Run --listen 10, tap the mic, see what RMS you get,
     and set threshold to 50-60% of your tap RMS.
     Fix: mosquitto_pub -t birdguard/config -m '{"energy_threshold": 200}'

  3. WRONG DEVICE — If ReSpeaker is not card 0, 'plughw:0,0' opens the
     wrong device (or the Pi's built-in audio).
     Fix: check arecord -l, update audio_device in shared.py

  4. TURRET_MOVING STUCK — If turret_moving.is_set() is stuck True,
     audio processing is entirely skipped.
     Fix: check smooth_move()/deterrent_sweep() in shared.py

  5. CAPTURE MUTED — The WM8960 capture can be muted independently.
     Fix: amixer -c 0 set 'Capture' cap

  6. COOLDOWN — After each wake, audio is ignored for 2 seconds.
     If you're tapping rapidly, only the first tap registers.
""")


if __name__ == "__main__":
    main()