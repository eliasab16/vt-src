# Minimal MicroPython shim for Adafruit's CircuitPython I2CDevice
# Used by vl53l4cd.py driver


class I2CDevice:
    def __init__(self, i2c, address):
        self._i2c = i2c
        self._address = address

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def write(self, data):
        self._i2c.writeto(self._address, data)

    def readinto(self, buf):
        self._i2c.readfrom_into(self._address, buf)
