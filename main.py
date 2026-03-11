from machine import Pin, PWM
import time, sys


# ==== PIN CONFIG -
SERVO_PAN_PIN  = 14  
SERVO_TILT_PIN = 12  
LASER_PIN      = 28


# ==== Servo setup ====
servo_pan = PWM(Pin(SERVO_PAN_PIN))
servo_tilt = PWM(Pin(SERVO_TILT_PIN))
servo_pan.freq(50)
servo_tilt.freq(50)


# Adjust these for your servos if needed
MIN_DUTY = 2000
MAX_DUTY = 8000


def angle_to_duty(angle):
    angle = max(0, min(180, angle))
    return int(MIN_DUTY + (MAX_DUTY - MIN_DUTY) * angle / 180)


def set_servo(pwm, angle):
    pwm.duty_u16(angle_to_duty(angle))


# ==== Laser setup ====
laser = Pin(LASER_PIN, Pin.OUT)
laser.value(0)


# ==== Command loop ====
# Expected format: PAN<deg>,TILT<deg>,LASER<0/1>
def process_command(line: str):
    line = line.strip().upper()
    if not line:
        return
    parts = line.split(',')


    pan = None
    tilt = None
    las = None


    for p in parts:
        p = p.strip()
        if p.startswith("PAN"):
            pan = int(p[3:])
        elif p.startswith("TILT"):
            tilt = int(p[4:])
        elif p.startswith("LASER"):
            las = int(p[5:])


    if pan is not None:
        set_servo(servo_pan, pan)
    if tilt is not None:
        set_servo(servo_tilt, tilt)
    if las is not None:
        laser.value(1 if las else 0)


# Main loop: read lines from USB serial
print("Pico turret ready")
while True:
    try:
        line = sys.stdin.readline()
        if not line:
            time.sleep(0.01)
            continue
        process_command(line)
    except Exception as e:
        # Optional: print errors back
        print("ERR:", e)