import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

# Give camera time to initialize
time.sleep(0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # h, w = frame.shape[:2]
    # start = (h - w) // 2
    # frame = frame[start:start+w, :]
    h, w = frame.shape[:2]
    cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 2)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
