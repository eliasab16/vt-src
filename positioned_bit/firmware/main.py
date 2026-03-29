from machine import Pin, PWM, I2C, WDT
import sys
import select
from time import ticks_us, ticks_diff, ticks_ms

# --- Stall detection tuning ---
STALL_RAW_A  = 1.8   # hard cutoff: stop if current exceeds this (Amps)
STALL_MULT   = 2.8   # soft cutoff: stop if current > baseline * this
STALL_SETTLE = 500   # ms after motor start before stall detection is active
INA_INTERVAL = 10    # ms between INA219 reads

wdt = WDT(timeout=8000)

_poll_out = select.poll()
_poll_out.register(sys.stdout, select.POLLOUT)

def nb_print(msg):
    if _poll_out.poll(0):
        sys.stdout.write(msg + '\n')

# Laser on GP7
laser = Pin(7, Pin.OUT)
laser.value(0)

# INA219 minimal driver — shunt 0.1 Ω, current LSB = 1 mA, Cal = 4096
class INA219:
    def __init__(self, i2c, addr=0x40):
        self._i2c = i2c; self._addr = addr
        self._wreg(0x00, 0x399F)  # 32V / 320mV / continuous
        self._wreg(0x05, 4096)    # calibration
    def _wreg(self, r, v):
        self._i2c.writeto_mem(self._addr, r, bytes([v >> 8, v & 0xFF]))
    def _rreg(self, r):
        d = self._i2c.readfrom_mem(self._addr, r, 2)
        return (d[0] << 8) | d[1]
    @property
    def current(self):
        v = self._rreg(0x04)
        return ((v - 65536) if v > 32767 else v) * 0.001  # Amps

# INA219 on I2C0 (GP0=SDA, GP1=SCL)
i2c0 = I2C(0, sda=Pin(0), scl=Pin(1), freq=400_000)
ina_ok = False
ina    = None
try:
    ina    = INA219(i2c0, addr=0x40)
    ina_ok = True
    nb_print("INA219 ready")
except Exception as e:
    nb_print("INA219 error: " + str(e))

# Stall detection state
bit_on       = False
bit_start_ms = 0
bit_baseline = 0.0
bit_base_ok  = False
ina_last_ms  = 0

# Automation state: 'idle' → 'bit_wait' (1s delay) → 'bit_run'
auto_state       = 'idle'
auto_deadline_ms = 0

# ToF VL53L4CD on I2C1 (GP26=SDA, GP27=SCL)
i2c1 = I2C(1, sda=Pin(26), scl=Pin(27), freq=400_000)
tof_ok = False
tof_dist = -1
try:
    from vl53l4cd import VL53L4CD
    tof = VL53L4CD(i2c1)
    tof.inter_measurement = 0
    tof.timing_budget = 20
    tof.start_ranging()
    tof_ok = True
    nb_print("VL53L4CD ready")
except Exception as e:
    nb_print("VL53L4CD error: " + str(e))

# Bit motor via DRV8871 (GP2=IN1, GP3=IN2)
in1 = PWM(Pin(2))
in2 = PWM(Pin(3))
in1.freq(1000)
in2.freq(1000)
in1.duty_u16(0)
in2.duty_u16(0)

# Y-axis: 28BYJ-48 via ULN2003 (GP10-GP13)
y_pins = [Pin(i, Pin.OUT) for i in (10, 11, 12, 13)]
Y_SEQ = [
    [1,0,0,0],[1,1,0,0],[0,1,0,0],[0,1,1,0],
    [0,0,1,0],[0,0,1,1],[0,0,0,1],[1,0,0,1],
]
y_dir = 0
y_phase = 0
y_last = ticks_us()

# Z-axis: NEMA 11 via TMC2209 (GP4=STEP, GP5=DIR, GP6=EN)
z_step = Pin(4, Pin.OUT)
z_dir  = Pin(5, Pin.OUT)
z_en   = Pin(6, Pin.OUT)
z_en.value(1)
z_moving = 0
z_last = ticks_us()
z_state = 0

# Limit switches (active LOW, pull-up)
y_lower_limit = Pin(16, Pin.IN, Pin.PULL_UP)
z_back_limit  = Pin(17, Pin.IN, Pin.PULL_UP)
y_upper_limit = Pin(18, Pin.IN, Pin.PULL_UP)
z_front_limit = Pin(19, Pin.IN, Pin.PULL_UP)

poll = select.poll()
poll.register(sys.stdin, select.POLLIN)

while True:
    wdt.feed()
    now    = ticks_us()
    now_ms = ticks_ms()

    if poll.poll(0):
        cmd = sys.stdin.read(1)
        if cmd == "1":
            in1.duty_u16(40000); in2.duty_u16(0)
            bit_on = True; bit_start_ms = now_ms; bit_base_ok = False
        elif cmd == "7":
            in1.duty_u16(0); in2.duty_u16(40000)
            bit_on = True; bit_start_ms = now_ms; bit_base_ok = False
        elif cmd == "4":
            in1.duty_u16(0); in2.duty_u16(0)
            bit_on = False; bit_base_ok = False
            auto_state = 'idle'
        elif cmd == "a":
            auto_state = 'bit_wait'
            auto_deadline_ms = now_ms + 1000
            nb_print("AUTO BIT WAIT")
        elif cmd == "2":
            y_dir = -1
        elif cmd == "8":
            y_dir = 1
        elif cmd == "5":
            y_dir = 0
            for p in y_pins: p.value(0)
        elif cmd == "3":
            z_en.value(0); z_dir.value(1); z_moving = 1
        elif cmd == "9":
            z_en.value(0); z_dir.value(0); z_moving = 1
        elif cmd == "6":
            z_moving = 0; z_en.value(1)
        elif cmd == "l":
            laser.value(1 - laser.value())
            nb_print("LASER " + ("ON" if laser.value() else "OFF"))
        elif cmd == "d":
            if tof_ok:
                nb_print("D " + str(tof_dist))
            else:
                nb_print("TOF NOT READY")
        elif cmd == "i":
            if ina_ok:
                try:
                    nb_print("I " + str(round(ina.current, 3)))
                except Exception:
                    nb_print("INA READ ERR")
            else:
                nb_print("INA NOT READY")

    if y_dir == 1 and y_upper_limit.value() == 0:
        y_dir = 0
        for p in y_pins: p.value(0)
        nb_print("Y UPPER LIMIT")

    if y_dir == -1 and y_lower_limit.value() == 0:
        y_dir = 0
        for p in y_pins: p.value(0)
        nb_print("Y LOWER LIMIT")

    if z_moving and z_dir.value() == 0 and z_front_limit.value() == 0:
        z_moving = 0; z_en.value(1); nb_print("Z FRONT LIMIT")

    if z_moving and z_dir.value() == 1 and z_back_limit.value() == 0:
        z_moving = 0; z_en.value(1); nb_print("Z BACK LIMIT")

    if y_dir != 0 and ticks_diff(now, y_last) >= 800:
        phase = Y_SEQ[y_phase]
        for i, p in enumerate(y_pins): p.value(phase[i])
        y_phase = (y_phase + y_dir) % 8
        y_last = now

    if z_moving and ticks_diff(now, z_last) >= 500:
        z_state = 1 - z_state
        z_step.value(z_state)
        z_last = now

    if tof_ok and tof.data_ready:
        tof_dist = tof.distance
        tof.clear_interrupt()

    # Automation: after 1s delay, activate bit motor (direction "7")
    if auto_state == 'bit_wait' and ticks_diff(now_ms, auto_deadline_ms) >= 0:
        in1.duty_u16(0); in2.duty_u16(40000)
        bit_on = True; bit_start_ms = now_ms; bit_base_ok = False
        auto_state = 'bit_run'
        nb_print("AUTO BIT RUN")

    # INA219 stall detection
    if ina_ok and bit_on and ticks_diff(now_ms, ina_last_ms) >= INA_INTERVAL:
        ina_last_ms = now_ms
        try:
            amps = abs(ina.current)
        except Exception:
            amps = 0.0
        if ticks_diff(now_ms, bit_start_ms) >= STALL_SETTLE:
            if not bit_base_ok:
                bit_baseline = amps if amps > 0.05 else 0.1
                bit_base_ok  = True
            elif amps >= STALL_RAW_A or (bit_baseline > 0 and amps >= bit_baseline * STALL_MULT):
                in1.duty_u16(0); in2.duty_u16(0)
                bit_on = False; bit_base_ok = False
                auto_state = 'idle'
                nb_print("BIT STALL")
