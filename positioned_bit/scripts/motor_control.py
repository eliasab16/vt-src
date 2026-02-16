import serial
import tty
import termios
import sys

ser = serial.Serial('/dev/cu.usbmodem101', 115200, timeout=1)

old_settings = termios.tcgetattr(sys.stdin)
tty.setraw(sys.stdin)

print("Arrow Up = forward, Arrow Down = reverse, q = quit\r")

try:
    while True:
        ch = sys.stdin.read(1)
        if ch == 'q':
            break
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch3 == 'A':
                ser.write(b'F\n')
                print("Forward\r")
            elif ch3 == 'B':
                ser.write(b'B\n')
                print("Reverse\r")
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    ser.close()