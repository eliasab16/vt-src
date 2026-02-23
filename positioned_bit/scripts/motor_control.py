#!/Users/elisd/miniconda3/bin/python
import serial
import tty
import termios
import sys
import select

ser = serial.Serial('/dev/cu.usbmodem101', 115200, timeout=0.05)

old_settings = termios.tcgetattr(sys.stdin)
tty.setraw(sys.stdin)

print("Bit: 1=fwd 4=stop 7=rev | Y: 2=fwd 5=stop 8=rev | Z: 3=fwd 6=stop 9=rev | q=quit\r")

CMDS = {
    '1': ('1', 'Bit fwd'),
    '4': ('4', 'Bit stop'),
    '7': ('7', 'Bit rev'),
    '2': ('2', 'Y fwd'),
    '5': ('5', 'Y stop'),
    '8': ('8', 'Y rev'),
    '3': ('3', 'Z fwd'),
    '6': ('6', 'Z stop'),
    '9': ('9', 'Z rev'),
}

try:
    while True:
        if select.select([sys.stdin], [], [], 0)[0]:
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

        if ser.in_waiting:
            line = ser.readline().decode().strip()
            if line:
                print("Current: " + line + "    \r")
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    ser.close()
