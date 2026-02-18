#!/Users/elisd/miniconda3/bin/python
import serial
import tty
import termios
import sys

ser = serial.Serial('/dev/cu.usbmodem101', 115200, timeout=1)

old_settings = termios.tcgetattr(sys.stdin)
tty.setraw(sys.stdin)

print("Y-axis: 1=fwd 4=rev 7=stop | Z-axis: 2=fwd 5=rev 8=stop | Bit: 3=fwd 6=rev 9=stop | q=quit\r")

CMDS = {
    '1': ('1', 'Y fwd'),
    '4': ('4', 'Y stop'),
    '7': ('7', 'Y rev'),
    '2': ('2', 'Z fwd'),
    '5': ('5', 'Z stop'),
    '8': ('8', 'Z rev'),
    '3': ('3', 'Bit fwd'),
    '6': ('6', 'Bit stop'),
    '9': ('9', 'Bit rev'),
}

try:
    while True:
        ch = sys.stdin.read(1)
        if ch in ('q', '\x03'):
            ser.write(b'4\n')
            ser.write(b'5\n')
            ser.write(b'6\n')
            break
        if ch in CMDS:
            cmd, label = CMDS[ch]
            ser.write((cmd + '\n').encode())
            print(label + '\r')
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    ser.close()
