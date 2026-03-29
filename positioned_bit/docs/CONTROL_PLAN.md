# End-Effector Screw Driving Control Plan

## System Overview

**Hardware:**
- 2 stepper axes: Y (vertical, up/down) and Z (horizontal, forward/back)
- DC motor: spins the Phillips/slotted bit
- Camera: eye-in-hand, wide-angle (fisheye), mounted beside bit at same Y height, looking forward along Z axis
- ToF sensor (VL53L4CD): close-range depth measurement
- Laser pointer: reference dot visible in camera image
- Limit switches: one at each end of each stepper axis (4 total)
- Grippers: 2 parallel grippers (for holding wire — not part of this plan)

**Task:** After the robot arm has inserted a wire terminal into a circuit breaker, position the screwdriver bit over the terminal screw and tighten it to spec.

**Target breaker:** Baomain DZ47-63 (and similar DIN-rail MCBs). Slotted terminal screw, ~20-25 in-lbs (2.3-2.8 Nm) target torque.

---

## Key Observations from Camera Demo

From the POV video of approaching the DZ47-63:

1. **Barrel distortion is significant** — straight lines on the breaker body curve noticeably near frame edges. Must undistort before any pixel-based measurement.
2. **The screw head is visible and high-contrast** — metallic circle against the white/gray breaker body. Good target for classical CV.
3. **The terminal slot is below the screw** — wire enters from below, screw is above. The bit approaches from above/front.
4. **At close range, the screw fills a significant portion of the frame** — good for fine alignment but easy to lose the screw if offset is large.
5. **The field of view is wide** — good for searching, but pixel-to-mm ratio changes dramatically with distance.

---

## Control Architecture: State Machine

All three approaches below share the same top-level state machine. The approaches differ in **how screw detection and alignment work** (states 3-5).

```
┌──────┐    ┌────────┐    ┌───────────┐    ┌──────────┐    ┌─────────────┐
│ IDLE │───>│ HOMING │───>│ SEARCHING │───>│ ALIGNING │───>│ APPROACHING │
└──────┘    └────────┘    └───────────┘    └──────────┘    └─────────────┘
                                                                  │
          ┌──────────┐    ┌───────────┐    ┌───────────┐          │
          │ COMPLETE │<───│ VERIFYING │<───│TIGHTENING │<─────────┘
          └──────────┘    └───────────┘    └───────────┘
                                                │
                               ┌────────────────┘
                               v
                       ┌───────────────┐
                       │ERROR_RECOVERY │──> IDLE
                       └───────────────┘
```

### State Details

| State | Sensors Used | Actuators | Exit Condition |
|-------|-------------|-----------|----------------|
| **IDLE** | None | None | Command received |
| **HOMING** | Limit switches | Steppers Y, Z | Both axes homed |
| **SEARCHING** | Camera | Stepper Y (scan) | Screw detected in FOV |
| **ALIGNING** | Camera | Steppers Y, Z | Pixel error < threshold (e.g., 5px) |
| **APPROACHING** | Camera + ToF | Stepper Z (forward) | ToF < engagement distance (~5mm) |
| **ENGAGING** | Current sensor | DC motor (slow) + Z stepper (light pressure) | Current spike = bit seated in slot |
| **TIGHTENING** | Current sensor | DC motor (CW) | Current reaches target torque threshold |
| **VERIFYING** | Current sensor | DC motor (brief reverse + re-tighten) | Torque confirmed |
| **COMPLETE** | None | Steppers (retract) | At safe position |
| **ERROR_RECOVERY** | All | All (stop) | Retracted, awaiting re-attempt |

### Homing Sequence (always first)

```
1. Y axis first (move DOWN toward bottom limit switch)
   - Fast move down at 10 mm/s until bottom limit switch triggers
   - Back off up at 1 mm/s until switch releases
   - Set Y = 0 (bottom = home)

2. Z axis (move BACK toward rear limit switch)
   - Fast move back at 10 mm/s until rear limit switch triggers
   - Back off forward at 1 mm/s until switch releases
   - Set Z = 0 (rear = home)

3. Move to "ready" position (center of travel range)
```

### Critical Shared Subsystems

**Camera Undistortion (required for ALL approaches):**
```python
# One-time calibration with a checkerboard
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (W, H), None, None)
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 1, (W, H))
mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_K, (W, H), cv2.CV_32FC1)

# Every frame:
undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
```
This is essential — your fisheye lens will introduce ~10-20px of error at the frame edges without correction.

**Camera-to-Bit Alignment:**
```
Camera is mounted at the same Y height as the bit tip, offset slightly in X to avoid
occlusion. Vertical alignment uses the camera's horizontal center line: when the screw
is on the center line, it is at the same Y as the bit — this is distance-independent
and requires no depth calibration.

1. Home both axes
2. Verify camera is level (no pitch tilt) — center line must represent true horizontal
3. If camera has X offset: calibrate once by centering a target with the bit, then
   noting the pixel offset in the camera image. This offset is constant at all distances.
```

**Current Monitoring (DC Motor):**
```
Hardware: INA219 on DC motor power line, I2C to ESP32/Pico
Sample rate: 200 Hz (5ms interval)
Filter: moving average of 10 samples (50ms window)

Thresholds (calibrate empirically):
  - BASELINE: free-running current (no load)
  - ENGAGEMENT: current > 1.5x BASELINE (bit seated in screw slot)
  - TIGHTENING: current > 2x BASELINE (screw advancing under load)
  - TARGET_TORQUE: current = Kt * I_target (screw at spec)
  - CAM_OUT: current drops to < 0.5x BASELINE during tightening
  - STALL: current > 4x BASELINE for > 100ms (emergency stop)
```

---

## Approach 1: Classical CV + IBVS (Recommended Starting Point)

**Philosophy:** Use traditional computer vision to detect the screw, image-based visual servoing (IBVS) to align, ToF for depth, current sensing for torque. No ML, no training data, fully deterministic.

### Screw Detection Pipeline

```python
def detect_screw(frame):
    # 1. Undistort
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Enhance contrast (screw head is metallic vs plastic body)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 4. Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 5. Detect circles (screw heads)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=50,             # min distance between detected centers
        param1=100,             # Canny high threshold
        param2=30,              # accumulator threshold (lower = more sensitive)
        minRadius=10,           # adjust based on expected screw size at distance
        maxRadius=80
    )

    if circles is not None:
        # Take the strongest detection
        best = circles[0][0]  # (x, y, radius)
        return (int(best[0]), int(best[1])), int(best[2])

    # Fallback: blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 10000
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    if keypoints:
        kp = max(keypoints, key=lambda k: k.size)
        return (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2)

    return None, None
```

### Visual Servoing (ALIGNING state)

```python
# Target: screw center should be at the camera's center line (+ constant X offset if any).
# Because camera is at the same Y as the bit, center line alignment = bit alignment.
# This is distance-independent — no depth calibration needed.
target_px = (320 + dx_pixels, 240)  # camera center + constant X offset

def visual_servo_step(screw_center, target):
    error_x = screw_center[0] - target[0]  # pixels (horizontal — not actuated, for monitoring)
    error_y = screw_center[1] - target[1]  # pixels (vertical — drives Y stepper)

    # Proportional control in pixel space (no mm conversion needed)
    Kp = 0.5  # tune empirically until loop converges smoothly
    step_y = -Kp * error_y

    # Clamp maximum step size for safety
    step_y = max(-2.0, min(2.0, step_y))  # max 2mm per iteration

    command_stepper_move(axis='Y', distance_mm=step_y)

    converged = abs(error_y) < 5  # 5px threshold
    return converged
```

### Approach Phase (APPROACHING state)

```python
def approach_step():
    distance = tof.read_distance_mm()

    if distance > 50:
        speed = 5.0  # mm/s — fast approach
    elif distance > 20:
        speed = 2.0  # mm/s — moderate
    elif distance > 5:
        speed = 0.5  # mm/s — slow final approach
    else:
        return 'ENGAGING'  # close enough

    # Continue visual servoing during approach
    screw_center, _ = detect_screw(capture_frame())
    if screw_center:
        visual_servo_step(screw_center, target_px)

    command_stepper_move(axis='Z', distance_mm=0.5, speed=speed)  # positive = forward toward screw
    return 'APPROACHING'
```

### Pros & Cons

| Pros | Cons |
|------|------|
| No training data needed | Sensitive to lighting changes |
| Fast to implement and iterate | May struggle with reflective screw heads |
| Fully deterministic and debuggable | Needs re-tuning for different breaker models |
| Low compute — runs on ESP32 or Pi Zero | Hough circles can give false positives on similar circular features |
| No GPU needed | Slot orientation detection requires extra work |

### When to Use
- **Start here.** Get the mechanics and control loop working first. If screw detection is reliable >90% of the time, you may never need to move to Approach 2 or 3.

### Estimated Development Time
- Camera calibration (undistortion): 1-2 hours
- Camera-to-bit offset calibration: 1 hour
- Screw detection tuning: 2-4 hours (iterating on Hough params with real images)
- Visual servo loop: 2-3 hours
- State machine + integration: 4-6 hours
- Current monitoring + torque thresholds: 2-3 hours
- **Total: ~2-3 days of focused work to first successful tightening**

---

## Approach 2: ML-Based Screw Detection + Classical Control

**Philosophy:** Replace the classical CV screw detection with a small ML model (YOLOv8-nano) trained on images of your specific breaker. Keep everything else from Approach 1 (IBVS, state machine, current sensing). This gives robust detection while keeping the control loop deterministic.

### Data Collection

```
1. Mount the end-effector, run through approach sequences manually
2. Capture frames at various:
   - Distances (10mm to 100mm)
   - Y/Z offsets (screw at different positions in frame)
   - Lighting conditions
   - Screw states (loose, partially tightened, fully tightened)
3. Target: 150-300 annotated images
4. Label with bounding boxes around screw head
5. Optional: add keypoint annotations for screw center and slot endpoints
```

### Model Training

```python
# Using Ultralytics YOLOv8
from ultralytics import YOLO

# Option A: Object detection (bounding box)
model = YOLO('yolov8n.pt')  # nano model, 3.2M params
model.train(
    data='breaker_screws.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu'  # or 'mps' on your Mac
)

# Option B: Keypoint detection (screw center + slot endpoints)
model = YOLO('yolov8n-pose.pt')
model.train(
    data='breaker_screws_kp.yaml',
    epochs=100,
    imgsz=640,
    kpt_shape=[3, 2]  # 3 keypoints (center, slot_end_1, slot_end_2), 2D
)
```

### Inference Integration

```python
def detect_screw_ml(frame):
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    results = model.predict(frame, conf=0.5, verbose=False)

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]  # highest confidence
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center = (int((x1+x2)/2), int((y1+y2)/2))
        radius = int((x2-x1)/2)
        confidence = float(box.conf[0])
        return center, radius, confidence

    return None, None, 0.0
```

### Slot Orientation Detection (Bonus)

With keypoint detection, the model outputs the screw center AND the two endpoints of the slot. This enables:
- Aligning the bit to the slot orientation before engaging
- Detecting if the screw is a Phillips (cross pattern) vs slotted
- This is very hard to do reliably with classical CV but trivial with a keypoint model

### Pros & Cons

| Pros | Cons |
|------|------|
| Robust to lighting, angle, partial occlusion | Needs 150-300 labeled images |
| Can detect screw state (loose/tight/missing) | ~1-2 hours labeling time |
| Slot orientation detection "for free" with keypoints | Inference latency: ~15-30ms on RPi 4, ~5ms on Jetson |
| Generalizes across similar breaker models | Black box — harder to debug detection failures |
| Confidence score enables quality gating | Model drift if hardware/lighting changes significantly |

### When to Use
- If Approach 1's classical CV fails >10% of the time due to lighting or reflections
- If you need to handle multiple breaker models
- If slot orientation detection is important (bit must align to slot before engaging)

### Estimated Additional Time (on top of Approach 1)
- Data collection: 1-2 hours
- Labeling: 1-2 hours (using Roboflow or CVAT)
- Training: 1-2 hours (100 epochs on CPU/MPS)
- Integration: 1-2 hours (swap `detect_screw` → `detect_screw_ml`)
- **Total additional: ~1 day**

---

## Approach 3: Learned Policy via Imitation Learning

**Philosophy:** Instead of hand-coding the ALIGNING → APPROACHING → ENGAGING → TIGHTENING sequence, learn a policy from human demonstrations. The human teleoperates the end-effector through successful screw-tightening sequences, and a neural network learns to map sensor inputs to motor commands.

### Why Consider This

The transition from APPROACHING to ENGAGING is the hardest part to hand-code:
- The bit must find the screw slot while applying light downward pressure
- If the slot orientation is unknown, the bit must "feel around" for it
- The engagement "feel" that humans develop naturally is hard to specify as thresholds
- An imitation learning policy can capture these subtle behaviors

### Data Collection (Teleoperation)

```
Hardware: Use a simple joystick or keyboard to control:
  - Joystick X → stepper Z velocity (forward/back)
  - Joystick Y → stepper Y velocity (up/down)
  - Trigger/button → DC motor on/off + direction

Recording per demonstration:
  - Camera frames (640x480 @ 10Hz)
  - ToF distance reading
  - Stepper Y position (steps from home)
  - Stepper Z position (steps from home)
  - DC motor current (from INA219)
  - DC motor commanded PWM
  - Timestamp

Target: 50-100 successful demonstrations
  - Vary starting positions (different Y/Z offsets)
  - Vary breaker orientation slightly
  - Include some recovery demonstrations (bit slips, re-engage)
```

### Policy Architecture

```python
import torch
import torch.nn as nn

class ScrewDrivingPolicy(nn.Module):
    """
    Input: camera image + scalar sensors
    Output: stepper Y velocity, stepper Z velocity, DC motor PWM
    """
    def __init__(self):
        super().__init__()
        # Vision encoder (small ResNet or MobileNet backbone)
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2),  nn.ReLU(),   # 318x238
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),   # 157x117
            nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),   # 78x58
            nn.AdaptiveAvgPool2d(4),                      # 4x4x64 = 1024
            nn.Flatten(),
        )
        # Scalar sensor encoder
        self.scalar_encoder = nn.Sequential(
            nn.Linear(4, 32),  # [tof_dist, motor_current, y_pos, z_pos]
            nn.ReLU(),
        )
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(1024 + 32, 256), nn.ReLU(),
            nn.Linear(256, 64),        nn.ReLU(),
            nn.Linear(64, 3),          # [vy, vz, motor_pwm]
            nn.Tanh(),                 # outputs in [-1, 1], scale to actual ranges
        )

    def forward(self, image, scalars):
        vis_feat = self.vision(image)
        scl_feat = self.scalar_encoder(scalars)
        combined = torch.cat([vis_feat, scl_feat], dim=-1)
        return self.policy(combined)
```

### Training

```python
# Behavioral cloning (supervised learning)
from torch.utils.data import DataLoader

dataset = ScrewDrivingDataset('demonstrations/')  # loads recorded episodes
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ScrewDrivingPolicy()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(200):
    for images, scalars, actions in loader:
        pred_actions = model(images, scalars)
        loss = loss_fn(pred_actions, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Safety Wrapper (Critical)

A learned policy MUST run inside safety constraints — it cannot be trusted to respect limits:

```python
class SafePolicyExecutor:
    def __init__(self, policy, safety_limits):
        self.policy = policy
        self.limits = safety_limits

    def step(self, image, scalars):
        action = self.policy(image, scalars)
        vy, vz, motor_pwm = action

        # Hard limits
        vy = clamp(vy, -self.limits.max_vy, self.limits.max_vy)
        vz = clamp(vz, -self.limits.max_vz, self.limits.max_vz)
        motor_pwm = clamp(motor_pwm, -1.0, 1.0)

        # Position limits (from stepper counters)
        if self.y_pos <= self.limits.y_min and vy < 0: vy = 0
        if self.y_pos >= self.limits.y_max and vy > 0: vy = 0
        if self.z_pos <= self.limits.z_min and vz < 0: vz = 0
        if self.z_pos >= self.limits.z_max and vz > 0: vz = 0

        # Current limit (emergency stop if stall detected)
        if scalars.motor_current > self.limits.max_current:
            motor_pwm = 0
            vz = min(vz, 0)  # only allow moving BACK (retract)

        return vy, vz, motor_pwm
```

### Hybrid: Hand-Coded States + Learned Sub-Policies

The most practical version of Approach 3 doesn't replace the entire state machine — it replaces only the hardest states:

```
IDLE → HOMING → SEARCHING → ALIGNING → APPROACHING → ENGAGING → TIGHTENING → VERIFYING → COMPLETE
       ^^^^^^   ^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^
       hand-    classical    LEARNED POLICY (replaces 3 states)  hand-coded
       coded    CV or ML                                         (current
                                                                 thresholds)
```

The learned policy handles the continuous control from first detection to engagement. The hand-coded current monitoring handles tightening (where physics-based thresholds are reliable and safety-critical).

### Pros & Cons

| Pros | Cons |
|------|------|
| Captures subtle "feel" of engagement | Needs 50-100 demonstrations |
| Handles slot-finding behavior naturally | Teleop rig setup takes time |
| Can generalize to different breaker types | Harder to debug than explicit code |
| Most adaptable to unexpected situations | Safety wrapper is essential |
| Reduces hand-tuning of thresholds | Behavioral cloning has distribution shift issues |

### When to Use
- If the ENGAGING state proves difficult to hand-code (bit can't reliably find the slot)
- If you want to handle multiple breaker types without re-tuning
- If you're already collecting data through teleop anyway

### Estimated Additional Time
- Teleop rig software: 1-2 days
- Data collection (50-100 demos): 2-3 hours
- Training + iteration: 1-2 days
- Safety wrapper + integration: 1 day
- **Total additional: ~3-5 days**

---

## Comparison Matrix

| Criteria | Approach 1: Classical CV | Approach 2: ML Detection | Approach 3: Learned Policy |
|----------|------------------------|------------------------|---------------------------|
| **Development time** | 2-3 days | 3-4 days | 5-8 days |
| **Training data needed** | None | 150-300 images | 50-100 demonstrations |
| **Compute requirements** | Pi Zero / ESP32 | Pi 4 / Jetson Nano | Pi 4 + GPU for training |
| **Robustness to lighting** | Low-Medium | High | High |
| **Robustness to breaker variation** | Low | Medium-High | High |
| **Debuggability** | Excellent | Good | Poor |
| **Safety guarantees** | Deterministic | Deterministic control | Needs safety wrapper |
| **Slot finding capability** | Manual/scripted | Orientation from keypoints | Learned from demos |
| **Recommended for** | Proof of concept, single breaker type | Production with multiple breakers | Complex engagement sequences |

---

## Recommended Path Forward

### Phase 1: Foundation (Week 1)
1. **Lens calibration** — print a checkerboard, capture 20+ images at various angles, run `cv2.calibrateCamera()`, save the undistortion maps
2. **Stepper homing** — implement the two-phase homing sequence with your limit switches; verify repeatability by homing 10x and measuring position
3. **Camera-to-bit alignment** — verify camera is level; if X-offset exists, calibrate the constant pixel offset once
4. **Current monitoring** — wire up INA219, write a data logger, characterize the DC motor's current profile: free-running, stall, and at various loads
5. **State machine skeleton** — implement the state machine with stub implementations for each state

### Phase 2: Classical CV (Week 2) — Approach 1
1. **Screw detection** — implement Hough circles + blob detection fallback, tune with real images from the mounted camera
2. **Visual servoing** — implement the IBVS loop, tune Kp, test convergence
3. **Approach sequence** — integrate ToF for depth-controlled descent
4. **Tightening** — implement current-based torque control with the three thresholds
5. **End-to-end test** — run full sequences, log everything, identify failure modes

### Phase 3: Evaluate and Decide (Week 3)
- If classical CV works >90%: **ship it, refine edge cases**
- If detection is unreliable: **move to Approach 2** (add YOLOv8-nano)
- If engagement/slot-finding is the bottleneck: **add Approach 3** for the ENGAGING state only
- Collect data regardless — even if Approach 1 works, logged images are useful if you later need ML

### Phase 4: Hardening
- Add timeout watchdogs to every state
- Add retry logic (cam-out → retry engagement up to 3x)
- Add logging for every state transition and sensor reading
- Characterize torque thresholds for the specific breaker model
- Test with wire terminals installed (changes friction characteristics)

---

## Hardware Wiring Checklist

```
ESP32 / Pico GPIO assignments needed:

Stepper Y (vertical):
  - STEP pin
  - DIR pin
  - ENABLE pin
  - Limit switch BOTTOM (with 100nF debounce cap) — home position
  - Limit switch TOP (with 100nF debounce cap)
  - (Optional: TMC2209 UART TX/RX for StallGuard)

Stepper Z (horizontal):
  - STEP pin
  - DIR pin
  - ENABLE pin
  - Limit switch REAR (with debounce) — home position
  - Limit switch FRONT (with debounce)
  - (Optional: TMC2209 UART TX/RX)

DC Motor:
  - PWM pin → motor driver (DRV8871 or similar)
  - DIR pin → motor driver
  - INA219 SDA (I2C)
  - INA219 SCL (I2C)

Camera:
  - CSI ribbon cable (Pi) or USB (ESP32-S3)

ToF (VL53L4CD):
  - SDA (I2C, can share bus with INA219)
  - SCL (I2C)
  - (Optional: XSHUT for multi-sensor)

Laser:
  - Digital output pin (on/off)

Total GPIO: ~14-16 pins minimum
```

---

## Key Risk Areas and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Bit doesn't find slot** | Can't engage screw | Slow rotation while pressing lightly; current spike = slot found. Or add a "wiggle" routine that oscillates the bit ±5° while descending |
| **Camera barrel distortion** | Alignment error | Calibrate lens, apply undistortion to every frame |
| **Specular reflections on screw** | False detection / lost detection | Add diffuse LED lighting (ring light), or polarizing filter |
| **Cross-threading** | Damaged breaker | Monitor current during rundown phase; abort if current exceeds 2x baseline before the screw should have seated |
| **Stepper missed steps** | Position drift | Use moderate speeds + acceleration ramps; re-home between operations if concerned |
| **DC motor stall damage** | Burned motor/driver | Hardware current limit on driver + software timeout (max 5s of tightening) |
| **ToF minimum range** | Unreliable at <10mm | Switch to current-sensing-only control below 10mm distance |
| **Camera vibration blur** | Blurry images during motion | Stop steppers, capture frame, then move (stop-and-shoot). Or short exposure time |

---

## Appendix: Useful Code Patterns

### Stepper Control (AccelStepper-style, MicroPython)

```python
class StepperAxis:
    def __init__(self, step_pin, dir_pin, enable_pin, limit_home_pin, limit_far_pin,
                 steps_per_mm=1600):  # 200 steps/rev * 8 microsteps, 1mm lead
        self.step = step_pin
        self.dir = dir_pin
        self.enable = enable_pin
        self.limit_home = limit_home_pin  # Y=bottom, Z=rear
        self.limit_far = limit_far_pin    # Y=top, Z=front
        self.steps_per_mm = steps_per_mm
        self.position_steps = 0
        self.homed = False

    def home(self):
        """Two-phase homing: fast approach to home switch, slow back-off."""
        self.enable.on()

        # Phase 1: fast approach toward home limit (Y=bottom, Z=rear)
        self.dir.off()  # direction toward home
        while not self.limit_home.value():
            self.step.on()
            time.sleep_us(200)  # 5kHz step rate = fast
            self.step.off()
            time.sleep_us(200)

        # Phase 2: slow back-off
        self.dir.on()  # reverse direction
        time.sleep_ms(100)
        while self.limit_home.value():
            self.step.on()
            time.sleep_us(1000)  # 1kHz step rate = slow
            self.step.off()
            time.sleep_us(1000)

        self.position_steps = 0
        self.homed = True

    def move_mm(self, distance_mm, speed_mm_s=5.0):
        """Move a given distance in mm at given speed."""
        steps = int(distance_mm * self.steps_per_mm)
        self.dir.value(1 if steps > 0 else 0)
        delay_us = int(1e6 / (speed_mm_s * self.steps_per_mm))

        for _ in range(abs(steps)):
            # Check limits
            if steps < 0 and self.limit_home.value(): break
            if steps > 0 and self.limit_far.value(): break

            self.step.on()
            time.sleep_us(delay_us)
            self.step.off()
            time.sleep_us(delay_us)
            self.position_steps += 1 if steps > 0 else -1
```

### Current Monitor

```python
from ina219 import INA219

class TorqueMonitor:
    def __init__(self, i2c, shunt_ohms=0.1, max_amps=3.2):
        self.ina = INA219(shunt_ohms, max_amps, i2c=i2c)
        self.ina.configure()
        self.baseline = None
        self.history = []
        self.WINDOW = 10

    def calibrate_baseline(self, samples=50):
        """Run motor briefly, measure free-running current."""
        readings = [self.ina.current() for _ in range(samples)]
        self.baseline = sum(readings) / len(readings)

    def read_filtered(self):
        """Read current with moving average filter."""
        self.history.append(self.ina.current())
        if len(self.history) > self.WINDOW:
            self.history.pop(0)
        return sum(self.history) / len(self.history)

    def check_state(self):
        current = self.read_filtered()
        if self.baseline is None:
            return 'UNCALIBRATED', current

        ratio = current / self.baseline
        if ratio > 4.0:
            return 'STALL', current
        elif ratio > 2.5:
            return 'TARGET_TORQUE', current
        elif ratio > 1.5:
            return 'TIGHTENING', current
        elif ratio < 0.3 and len(self.history) == self.WINDOW:
            return 'CAM_OUT', current
        else:
            return 'RUNNING', current
```
