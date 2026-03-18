from machine import Pin, PWM, I2C
import sys
import select
from time import ticks_us, ticks_diff

# This script should be running on the pico. It's used to control the motors.

# Laser on GP7 (via PN2222 transistor, 1k base resistor)
laser = Pin(7, Pin.OUT)
laser.value(0)

# ToF sensor VL53L4CD on I2C0 (GP8=SDA, GP9=SCL)
i2c0 = I2C(0, sda=Pin(8), scl=Pin(9), freq=400_000)
tof_ok = False
try:
    from vl53l4cd import VL53L4CD
    tof = VL53L4CD(i2c0)
    tof.inter_measurement = 0
    tof.timing_budget = 20
    tof.start_ranging()
    tof_ok = True
    print("VL53L4CD ready")
except Exception as e:
    print("VL53L4CD error:", e)

# Bit motor: DC worm gear via DRV8871
in1 = PWM(Pin(2))
in2 = PWM(Pin(3))
in1.freq(1000)
in2.freq(1000)
in1.duty_u16(0)
in2.duty_u16(0)

# Y-axis: 28BYJ-48 via ULN2003
y_pins = [Pin(i, Pin.OUT) for i in (10, 11, 12, 13)]
Y_SEQ = [
    [1,0,0,0],[1,1,0,0],[0,1,0,0],[0,1,1,0],
    [0,0,1,0],[0,0,1,1],[0,0,0,1],[1,0,0,1],
]
y_dir = 0
y_phase = 0
y_last = ticks_us()

# Z-axis: NEMA 11 via TMC2209
z_step = Pin(4, Pin.OUT)
z_dir = Pin(5, Pin.OUT)
z_en = Pin(6, Pin.OUT)
z_en.value(1)
z_moving = 0
z_last = ticks_us()
z_state = 0

# ToF cached distance
tof_dist = -1

# Limit switches (active LOW with pull-up)
y_lower_limit = Pin(16, Pin.IN, Pin.PULL_UP)
z_back_limit  = Pin(17, Pin.IN, Pin.PULL_UP)
y_upper_limit = Pin(18, Pin.IN, Pin.PULL_UP)
z_front_limit = Pin(19, Pin.IN, Pin.PULL_UP)

poll = select.poll()
poll.register(sys.stdin, select.POLLIN)

while True:
    now = ticks_us()

    if poll.poll(0):
        cmd = sys.stdin.read(1)
        if cmd == "1":
            in1.duty_u16(40000)
            in2.duty_u16(0)
        elif cmd == "7":
            in1.duty_u16(0)
            in2.duty_u16(40000)
        elif cmd == "4":
            in1.duty_u16(0)
            in2.duty_u16(0)
        elif cmd == "2":
            y_dir = -1
        elif cmd == "8":
            y_dir = 1
        elif cmd == "5":
            y_dir = 0
            for p in y_pins:
                p.value(0)
        elif cmd == "3":
            z_en.value(0)
            z_dir.value(1)
            z_moving = 1
        elif cmd == "9":
            z_en.value(0)
            z_dir.value(0)
            z_moving = 1
        elif cmd == "6":
            z_moving = 0
            z_en.value(1)
        elif cmd == "l":
            laser.value(1 - laser.value())
            print("LASER", "ON" if laser.value() else "OFF")
        elif cmd == "d":
            if tof_ok:
                print("D", tof_dist)
            else:
                print("TOF NOT READY")
    
    # check the limit switches for each axis and stop when triggered 
    if y_dir == 1 and y_upper_limit.value() == 0:
        y_dir = 0
        for p in y_pins:
            p.value(0)
        print("Y UPPER LIMIT")

    if y_dir == -1 and y_lower_limit.value() == 0:
        y_dir = 0
        for p in y_pins:
            p.value(0)
        print("Y LOWER LIMIT")

    if z_moving and z_dir.value() == 0 and z_front_limit.value() == 0:
        z_moving = 0
        z_en.value(1)
        print("Z FRONT LIMIT")

    if z_moving and z_dir.value() == 1 and z_back_limit.value() == 0:
        z_moving = 0
        z_en.value(1)
        print("Z BACK LIMIT")

    if y_dir != 0 and ticks_diff(now, y_last) >= 800:
        phase = Y_SEQ[y_phase]
        for i, p in enumerate(y_pins):
            p.value(phase[i])
        y_phase = (y_phase + y_dir) % 8
        y_last = now

    if z_moving and ticks_diff(now, z_last) >= 500:
        z_state = 1 - z_state
        z_step.value(z_state)
        z_last = now

    # Non-blocking ToF read: grab latest distance if data is ready
    if tof_ok and tof.data_ready:
        tof_dist = tof.distance
        tof.clear_interrupt()
