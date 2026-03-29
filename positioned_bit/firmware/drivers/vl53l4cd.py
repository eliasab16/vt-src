# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2022 Carter Nelson for Adafruit Industries
# SPDX-License-Identifier: MIT
#
# MicroPython port from: https://github.com/AHSPC/VL53L4CD_micropython

import time
import struct
import i2c_device
from micropython import const

_VL53L4CD_VHV_CONFIG_TIMEOUT_MACROP_LOOP_BOUND = const(0x0008)
_VL53L4CD_GPIO_HV_MUX_CTRL = const(0x0030)
_VL53L4CD_GPIO_TIO_HV_STATUS = const(0x0031)
_VL53L4CD_SYSTEM_INTERRUPT = const(0x0046)
_VL53L4CD_RANGE_CONFIG_A = const(0x005E)
_VL53L4CD_RANGE_CONFIG_B = const(0x0061)
_VL53L4CD_INTERMEASUREMENT_MS = const(0x006C)
_VL53L4CD_SYSTEM_INTERRUPT_CLEAR = const(0x0086)
_VL53L4CD_SYSTEM_START = const(0x0087)
_VL53L4CD_RESULT_DISTANCE = const(0x0096)
_VL53L4CD_RESULT_OSC_CALIBRATE_VAL = const(0x00DE)
_VL53L4CD_FIRMWARE_SYSTEM_STATUS = const(0x00E5)
_VL53L4CD_IDENTIFICATION_MODEL_ID = const(0x010F)
_VL53L4CD_I2C_SLAVE_DEVICE_ADDRESS = const(0x0001)


class VL53L4CD:
    """Minimal MicroPython driver for the VL53L4CD ToF sensor."""

    def __init__(self, i2c, address=0x29):
        self._i2c = i2c
        self.i2c_device = i2c_device.I2CDevice(i2c, address)
        model_id, module_type = self.model_info
        if model_id != 0xEB or module_type != 0xAA:
            raise RuntimeError("Wrong sensor ID or type!")
        self._ranging = False
        self._sensor_init()

    def _sensor_init(self):
        init_seq = (
            b"\x12"  # 0x2d
            b"\x00"  # 0x2e
            b"\x00"  # 0x2f
            b"\x11"  # 0x30
            b"\x02"  # 0x31
            b"\x00"  # 0x32
            b"\x02"  # 0x33
            b"\x08"  # 0x34
            b"\x00"  # 0x35
            b"\x08"  # 0x36
            b"\x10"  # 0x37
            b"\x01"  # 0x38
            b"\x01"  # 0x39
            b"\x00"  # 0x3a
            b"\x00"  # 0x3b
            b"\x00"  # 0x3c
            b"\x00"  # 0x3d
            b"\xff"  # 0x3e
            b"\x00"  # 0x3f
            b"\x0f"  # 0x40
            b"\x00"  # 0x41
            b"\x00"  # 0x42
            b"\x00"  # 0x43
            b"\x00"  # 0x44
            b"\x00"  # 0x45
            b"\x20"  # 0x46
            b"\x0b"  # 0x47
            b"\x00"  # 0x48
            b"\x00"  # 0x49
            b"\x02"  # 0x4a
            b"\x14"  # 0x4b
            b"\x21"  # 0x4c
            b"\x00"  # 0x4d
            b"\x00"  # 0x4e
            b"\x05"  # 0x4f
            b"\x00"  # 0x50
            b"\x00"  # 0x51
            b"\x00"  # 0x52
            b"\x00"  # 0x53
            b"\xc8"  # 0x54
            b"\x00"  # 0x55
            b"\x00"  # 0x56
            b"\x38"  # 0x57
            b"\xff"  # 0x58
            b"\x01"  # 0x59
            b"\x00"  # 0x5a
            b"\x08"  # 0x5b
            b"\x00"  # 0x5c
            b"\x00"  # 0x5d
            b"\x01"  # 0x5e
            b"\xcc"  # 0x5f
            b"\x07"  # 0x60
            b"\x01"  # 0x61
            b"\xf1"  # 0x62
            b"\x05"  # 0x63
            b"\x00"  # 0x64
            b"\xa0"  # 0x65
            b"\x00"  # 0x66
            b"\x80"  # 0x67
            b"\x08"  # 0x68
            b"\x38"  # 0x69
            b"\x00"  # 0x6a
            b"\x00"  # 0x6b
            b"\x00"  # 0x6c
            b"\x00"  # 0x6d
            b"\x0f"  # 0x6e
            b"\x89"  # 0x6f
            b"\x00"  # 0x70
            b"\x00"  # 0x71
            b"\x00"  # 0x72
            b"\x00"  # 0x73
            b"\x00"  # 0x74
            b"\x00"  # 0x75
            b"\x00"  # 0x76
            b"\x01"  # 0x77
            b"\x07"  # 0x78
            b"\x05"  # 0x79
            b"\x06"  # 0x7a
            b"\x06"  # 0x7b
            b"\x00"  # 0x7c
            b"\x00"  # 0x7d
            b"\x02"  # 0x7e
            b"\xc7"  # 0x7f
            b"\xff"  # 0x80
            b"\x9b"  # 0x81
            b"\x00"  # 0x82
            b"\x00"  # 0x83
            b"\x00"  # 0x84
            b"\x01"  # 0x85
            b"\x00"  # 0x86
            b"\x00"  # 0x87
        )
        self._wait_for_boot()
        self._write_register(0x002D, init_seq)
        self._start_vhv()
        self.clear_interrupt()
        self.stop_ranging()
        self._write_register(_VL53L4CD_VHV_CONFIG_TIMEOUT_MACROP_LOOP_BOUND, b"\x09")
        self._write_register(0x000B, b"\x00")
        self._write_register(0x0024, b"\x05\x00")
        self.inter_measurement = 0
        self.timing_budget = 50

    @property
    def model_info(self):
        info = self._read_register(_VL53L4CD_IDENTIFICATION_MODEL_ID, 2)
        return info[0], info[1]

    @property
    def distance(self):
        """Distance in centimeters."""
        dist = self._read_register(_VL53L4CD_RESULT_DISTANCE, 2)
        dist = struct.unpack(">H", dist)[0]
        return dist / 10

    def get_distance(self):
        """Clear interrupt, wait for new data, return distance in cm."""
        self.clear_interrupt()
        self.clear_interrupt()
        while not self.data_ready:
            pass
        return self.distance

    @property
    def timing_budget(self):
        osc_freq = struct.unpack(">H", self._read_register(0x0006, 2))[0]
        macro_period_us = 16 * (int(2304 * (0x40000000 / osc_freq)) >> 6)
        macrop_high = struct.unpack(">H", self._read_register(_VL53L4CD_RANGE_CONFIG_A, 2))[0]
        ls_byte = (macrop_high & 0x00FF) << 4
        ms_byte = (macrop_high & 0xFF00) >> 8
        ms_byte = 0x04 - (ms_byte - 1) - 1
        timing_budget_ms = (
            ((ls_byte + 1) * (macro_period_us >> 6)) - ((macro_period_us >> 6) >> 1)
        ) >> 12
        if ms_byte < 12:
            timing_budget_ms >>= ms_byte
        if self.inter_measurement == 0:
            timing_budget_ms += 2500
        else:
            timing_budget_ms *= 2
            timing_budget_ms += 4300
        return int(timing_budget_ms / 1000)

    @timing_budget.setter
    def timing_budget(self, val):
        if self._ranging:
            raise RuntimeError("Must stop ranging first.")
        if not 10 <= val <= 200:
            raise ValueError("Timing budget must be 10-200ms.")
        inter_meas = self.inter_measurement
        if inter_meas != 0 and val > inter_meas:
            raise ValueError("Budget can't exceed inter-measurement period.")
        osc_freq = struct.unpack(">H", self._read_register(0x0006, 2))[0]
        if osc_freq == 0:
            raise RuntimeError("Osc frequency is 0.")
        timing_budget_us = val * 1000
        macro_period_us = int(2304 * (0x40000000 / osc_freq)) >> 6
        if inter_meas == 0:
            timing_budget_us -= 2500
        else:
            timing_budget_us -= 4300
            timing_budget_us //= 2
        ms_byte = 0
        timing_budget_us <<= 12
        tmp = macro_period_us * 16
        ls_byte = int(((timing_budget_us + ((tmp >> 6) >> 1)) / (tmp >> 6)) - 1)
        while ls_byte & 0xFFFFFF00 > 0:
            ls_byte >>= 1
            ms_byte += 1
        ms_byte = (ms_byte << 8) + (ls_byte & 0xFF)
        self._write_register(_VL53L4CD_RANGE_CONFIG_A, struct.pack(">H", ms_byte))
        ms_byte = 0
        tmp = macro_period_us * 12
        ls_byte = int(((timing_budget_us + ((tmp >> 6) >> 1)) / (tmp >> 6)) - 1)
        while ls_byte & 0xFFFFFF00 > 0:
            ls_byte >>= 1
            ms_byte += 1
        ms_byte = (ms_byte << 8) + (ls_byte & 0xFF)
        self._write_register(_VL53L4CD_RANGE_CONFIG_B, struct.pack(">H", ms_byte))

    @property
    def inter_measurement(self):
        reg_val = struct.unpack(">I", self._read_register(_VL53L4CD_INTERMEASUREMENT_MS, 4))[0]
        clock_pll = struct.unpack(">H", self._read_register(_VL53L4CD_RESULT_OSC_CALIBRATE_VAL, 2))[0]
        clock_pll &= 0x3FF
        clock_pll = int(1.065 * clock_pll)
        return int(reg_val / clock_pll)

    @inter_measurement.setter
    def inter_measurement(self, val):
        if self._ranging:
            raise RuntimeError("Must stop ranging first.")
        timing_bud = self.timing_budget
        if val != 0 and val < timing_bud:
            raise ValueError("Inter-measurement can't be less than timing budget.")
        clock_pll = struct.unpack(">H", self._read_register(_VL53L4CD_RESULT_OSC_CALIBRATE_VAL, 2))[0]
        clock_pll &= 0x3FF
        int_meas = int(1.055 * val * clock_pll)
        self._write_register(_VL53L4CD_INTERMEASUREMENT_MS, struct.pack(">I", int_meas))
        self.timing_budget = timing_bud

    def start_ranging(self):
        if self.inter_measurement == 0:
            self._write_register(_VL53L4CD_SYSTEM_START, b"\x21")
        else:
            self._write_register(_VL53L4CD_SYSTEM_START, b"\x40")
        for _ in range(1000):
            if self.data_ready:
                break
            time.sleep(0.001)
        self.clear_interrupt()
        self._ranging = True

    def stop_ranging(self):
        self._write_register(_VL53L4CD_SYSTEM_START, b"\x00")
        self._ranging = False

    def clear_interrupt(self):
        self._write_register(_VL53L4CD_SYSTEM_INTERRUPT_CLEAR, b"\x01")

    @property
    def data_ready(self):
        if (
            self._read_register(_VL53L4CD_GPIO_TIO_HV_STATUS)[0] & 0x01
            == self._interrupt_polarity
        ):
            return True
        return False

    @property
    def _interrupt_polarity(self):
        int_pol = self._read_register(_VL53L4CD_GPIO_HV_MUX_CTRL)[0] & 0x10
        int_pol = (int_pol >> 4) & 0x01
        return 0 if int_pol else 1

    def _wait_for_boot(self):
        for _ in range(1000):
            if self._read_register(_VL53L4CD_FIRMWARE_SYSTEM_STATUS)[0] == 0x03:
                return
            time.sleep(0.001)
        raise TimeoutError("VL53L4CD boot timeout")

    def _start_vhv(self):
        self.start_ranging()
        for _ in range(1000):
            if self.data_ready:
                return
            time.sleep(0.001)
        raise TimeoutError("VL53L4CD VHV timeout")

    def _write_register(self, address, data, length=None):
        if length is None:
            length = len(data)
        with self.i2c_device as i2c:
            i2c.write(struct.pack(">H", address) + data[:length])

    def _read_register(self, address, length=1):
        data = bytearray(length)
        with self.i2c_device as i2c:
            i2c.write(struct.pack(">H", address))
            i2c.readinto(data)
        return data
