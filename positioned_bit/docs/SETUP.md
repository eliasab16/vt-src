# Positioned Bit - Setup & Control Reference

## Overview

A Pico-controlled positioning system with 3 axes, a laser, and a ToF distance sensor.
The Mac communicates with the Pico over USB serial, sending single-character commands.

## Components

| Component | Driver | Function |
|-----------|--------|----------|
| DC worm gear motor | DRV8871 | Bit rotation (drill/unscrew) |
| 28BYJ-48 stepper | ULN2003 | Y-axis (up/down) |
| NEMA 11 stepper | TMC2209 | Z-axis (front/back) |
| 650nm laser (<1mW) | PN2222 NPN transistor | Alignment laser |
| VL53L4CD ToF sensor | I2C (built-in) | Distance measurement (0-130cm) |
| 4x limit switches | Direct GPIO (active LOW) | Axis end-stops |

## Pin Map (GPIO numbers)

| GPIO | Function |
|------|----------|
| GP0 | INA219 SDA (I2C0, current sensor - not in script yet) |
| GP1 | INA219 SCL (I2C0) |
| GP2 | DRV8871 IN1 (bit motor) |
| GP3 | DRV8871 IN2 |
| GP4 | TMC2209 STEP (Z-axis) |
| GP5 | TMC2209 DIR |
| GP6 | TMC2209 EN |
| GP7 | Laser control (via PN2222, 1k base resistor) |
| GP8 | VL53L4CD SDA (I2C0) |
| GP9 | VL53L4CD SCL (I2C0) |
| GP10 | ULN2003 IN1 (Y-axis) |
| GP11 | ULN2003 IN2 |
| GP12 | ULN2003 IN3 |
| GP13 | ULN2003 IN4 |
| GP16 | Y-axis lower limit switch |
| GP17 | Z-axis back limit switch |
| GP18 | Y-axis upper limit switch |
| GP19 | Z-axis front limit switch |

**Free pins:** GP14, GP15, GP20-GP28

## Power

- **3V3** - INA219, VL53L4CD, laser positive
- **VBUS** - ULN2003 VCC
- **12V 5A supply** - DRV8871 VM, TMC2209 VM
- **GND** - common ground for all

## Hardware Notes

- DRV8871 has 100uF capacitor + two 1N4007 flyback diodes
- TMC2209 has 100uF capacitor across VM/GND, VREF ~0.6V, VDD wired directly to 3V3
- Laser switched via PN2222 NPN transistor (draws 13.1mA)
- Limit switches wired as: C to GND, NO to GPIO pin (internal pull-up enabled)
- Recommend 10k pull-down on GP7 to prevent laser turning on at boot

## Files on Pico

| Pico filename | Source file |
|---------------|------------|
| `main.py` | `scripts/pico_main.py` |
| `vl53l4cd.py` | `scripts/vl53l4cd.py` |
| `i2c_device.py` | `scripts/i2c_device.py` |

## Deploy to Pico

Copy all three files and reboot:

```
mpremote cp vt-src/positioned_bit/scripts/pico_main.py :main.py + cp vt-src/positioned_bit/scripts/vl53l4cd.py :vl53l4cd.py + cp vt-src/positioned_bit/scripts/i2c_device.py :i2c_device.py + reset
```

If only `pico_main.py` changed:

```
mpremote cp vt-src/positioned_bit/scripts/pico_main.py :main.py + reset
```

## Connect & Control

Open a serial terminal:

```
screen /dev/cu.usbmodem101 115200
```

On boot you should see `VL53L4CD ready`. Then use single keypresses to control:

### Serial Commands

| Key | Action |
|-----|--------|
| `1` | Bit motor forward |
| `4` | Bit motor stop |
| `7` | Bit motor reverse |
| `2` | Y-axis down |
| `5` | Y-axis stop |
| `8` | Y-axis up |
| `3` | Z-axis back |
| `6` | Z-axis stop |
| `9` | Z-axis front |
| `l` | Toggle laser on/off |
| `d` | Read ToF distance (cm) |

Layout on numpad:

```
7(bit rev)  8(Y up)    9(Z front)
4(bit stop) 5(Y stop)  6(Z stop)
1(bit fwd)  2(Y down)  3(Z back)
```

### Feedback Messages

- `LASER ON` / `LASER OFF` - laser toggle confirmation
- `D 12.3` - distance reading in cm (0 = nothing detected)
- `Y UPPER LIMIT` / `Y LOWER LIMIT` - Y-axis hit end-stop, auto-stopped
- `Z FRONT LIMIT` / `Z BACK LIMIT` - Z-axis hit end-stop, auto-stopped
- `TOF NOT READY` - sensor failed to initialize (check wiring)

## Troubleshooting

**screen shows `>>>` instead of running script:**
Press Ctrl+D to soft-reboot. The script runs automatically as `main.py`.

**VL53L4CD error: EIO:**
I2C communication failure. Check SDA/SCL wiring to GP8/GP9, and VIN to 3V3.

**ToF reads 0:**
Normal - means no object detected within range (~130cm max).

**ToF reads vary by decimals:**
Normal sensor noise (~1-2mm). Average readings if you need stability.

**Laser turns on at boot:**
GP7 floats before the script runs. Add a 10k pull-down resistor from GP7 to GND.

**To exit screen:**
Ctrl+A then K, then confirm with Y.
