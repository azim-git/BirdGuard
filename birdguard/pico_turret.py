import serial
import time
from serial import SerialException

PICO_PORT = "/dev/pico_turret"   # stable symlink from udev
BAUDRATE = 115200
RECONNECT_DELAY = 2.0            # seconds between reconnect attempts


class PicoTurret:
    def __init__(self, port=PICO_PORT, baud=BAUDRATE):
        self.port = port
        self.baud = baud
        self.ser = None
        self._healthy = True
        self._connect()

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    def _connect(self):
        """Block until serial connection is established."""
        while True:
            try:
                self.ser = serial.Serial(self.port, baudrate=self.baud, timeout=1)
                time.sleep(2.0)  # allow Pico to finish booting
                self._healthy = True
                break
            except SerialException as e:
                print(f"[PicoTurret] Connect failed on {self.port}: {e}")
                self._healthy = False
                time.sleep(RECONNECT_DELAY)

    def _ensure_connected(self):
        if self.ser is None or not self.ser.is_open:
            self._connect()

    def send_command(self, pan=None, tilt=None, laser=None):
        parts = []
        if pan is not None:
            parts.append(f"PAN{int(pan)}")
        if tilt is not None:
            parts.append(f"TILT{int(tilt)}")
        if laser is not None:
            parts.append(f"LASER{int(laser)}")
        if not parts:
            return

        cmd = ",".join(parts) + "\n"
        data = cmd.encode("ascii")

        for _ in range(2):  # try once, then reconnect and retry once
            try:
                self._ensure_connected()
                self.ser.write(data)
                self._healthy = True
                return
            except (SerialException, OSError) as e:
                print(f"[PicoTurret] Write failed: {e}, reconnecting...")
                try:
                    if self.ser is not None:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
                time.sleep(RECONNECT_DELAY)
        # Both attempts failed — mark unhealthy
        self._healthy = False

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()


if __name__ == "__main__":
    turret = PicoTurret()
    try:
        turret.send_command(pan=90, tilt=45, laser=1)
        time.sleep(1)
        turret.send_command(pan=0, tilt=90, laser=0)
    finally:
        turret.close()
