# End-Effector Autonomous Screw Driving: Research Notes

Compiled 2026-03-22. Research covering visual servoing, camera configurations, torque control, stepper homing, ToF+camera fusion, circuit breaker screw anatomy, existing projects, and control architectures.

---

## 1. Visual Servoing for Robotic Screw Driving

### 1.1 Problem Statement

The task is to use computer vision to: (a) locate a screw head in the camera's field of view, (b) compute the offset between the screwdriver bit and the screw, and (c) close the loop to drive the XZ carriage until alignment is achieved.

### 1.2 Visual Servoing Fundamentals

Two primary approaches exist:

**Image-Based Visual Servoing (IBVS):** The control law is computed directly in image pixel space. The robot moves to minimize pixel error between current feature positions and desired feature positions. No 3D model is needed. More robust to calibration errors, but can exhibit unintuitive camera paths and local minima.

**Position-Based Visual Servoing (PBVS):** The camera image is used to estimate the 3D pose of the target, and the control law operates in Cartesian space. Requires a calibrated camera model. Produces straighter Cartesian paths but is more sensitive to calibration errors.

**For a 2-axis XZ gantry (your case):** IBVS is simpler and more appropriate. You only need to minimize the pixel offset between the detected screw center and the desired image center point (where the bit is calibrated to be). This reduces to two PI or PID controllers -- one for X pixel error driving X stepper, one for Y pixel error driving Z stepper.

Reference: [Comparing PBVS and IBVS for Robotic Assembly](https://sites.ecse.rpi.edu/~rjradke/papers/peng-case20.pdf) -- found PBVS slightly faster and more accurate, but IBVS more robust. For peg-in-hole tasks, IBVS achieves higher success rates.

### 1.3 Classical Screw Detection Methods

**Hough Circle Transform:**
- Detects circular features (screw heads) in edge-detected images
- Works well for round-head screws; less effective for slotted flat-head screws in recessed terminals
- OpenCV: `cv2.HoughCircles()` after Canny edge detection
- Sensitive to lighting and contrast; needs preprocessing (Gaussian blur, histogram equalization)

**Template Matching:**
- `cv2.matchTemplate()` with a reference image of the screw head
- Works when screw appearance is consistent (same breaker model)
- Limitations: scale-sensitive, rotation-sensitive, brittle to lighting changes
- Multi-scale template matching can help but adds computation

**Edge Detection + Contour Analysis:**
- Canny edges -> `findContours()` -> filter by area, circularity, aspect ratio
- Good for slotted screws: look for the slot line within a circular region
- Can detect slot orientation for bit alignment

**Blob Detection:**
- `cv2.SimpleBlobDetector` -- detects circular features by area, circularity, convexity
- Used in the ScrewDrivingBot1 project (UR5e, UNSW) for screw hole localization
- Fast and simple; good starting point

### 1.4 ML/Deep Learning Approaches

**YOLO-based detection:**
- YOLOv5, YOLOv8 variants used for real-time screw detection in industrial settings
- FSS-YOLO (based on YOLOv5n): lightweight model for rail fastener screw detection, uses C3Fast module to reduce compute while maintaining accuracy
- YOLOv8-Seg: adds instance segmentation masks for precise screw boundary detection
- Training requires labeled dataset (50-200 annotated images typically sufficient for a constrained domain like circuit breaker screws)

**DCNN Classification:**
- Xception architecture achieved 98.71% accuracy for screw detection in electronics recycling (GitHub: [eyildiz-ugoe/screw_detection](https://github.com/eyildiz-ugoe/screw_detection))
- Transfer learning from ImageNet weights; input sizes 64x64 to 221x221 pixels
- 1843 true positives, 1974 true negatives, only 8 false positives, 42 false negatives

**Transformer-based Visual Servoing:**
- Recent work (2025): DET architecture fuses multiple camera images with joint angles for sub-millimeter precision without manual feature design or markers
- Convergence error: translational accuracy 0.21--0.50 mm, angular 0.07--0.20 degrees

### 1.5 Resolution and Framerate Requirements

**Resolution:**
- Industrial systems typically use 640x480 to 1280x960 pixels
- Common preprocessing: capture at 1280x960, crop central 640x480 for processing
- For a 50mm field of view at 640 pixels width: ~0.08 mm/pixel resolution
- Sub-pixel interpolation can improve effective resolution to ~0.02 mm

**Framerate:**
- Visual servoing control loops: 10-30 Hz is standard
- 10 Hz sufficient for slow approach; 30 Hz better for dynamic servoing
- Camera acquisition can be faster (60 Hz) with processing on alternate frames
- At slow stepper approach speeds (1 mm/s), even 5 Hz is adequate

**Practical recommendation for your system:** A 640x480 camera at 30 fps with classical detection (blob detection or Hough circles) is sufficient. The screw head is a high-contrast feature against the circuit breaker housing.

### 1.6 How Industrial Systems Do It

- **Keyence** and **Cognex** vision systems: use pattern matching with geometric features, not ML
- **Nitto Seiko**: vision-guided positioning where camera locates screw holes, communicates offset to SCARA robot
- **DEPRAG**: multi-camera setups with screwdriver-mounted cameras for final alignment
- **Inbolt**: real-time 3D vision for screw driving, handles part-to-part variation
- Common pattern: coarse positioning with overhead camera, fine alignment with tool-mounted camera

Sources:
- [From Perception to Precision: Vision-Based Mobile Robotic Manipulation for Assembly Screwdriving](https://www.sciencedirect.com/science/article/pii/S0736584525002029)
- [An Automated 4-DOF Robot Screw Fastening Using Visual Servo](https://www.researchgate.net/publication/251989311_An_automated_four-DOF_robot_screw_fastening_using_visual_servo)
- [High-Precision Transformer-Based Visual Servoing](https://arxiv.org/html/2503.04862v2)
- [Nitto Seiko Vision Guided Positioning Case Study](https://nittoseikoamerica.com/resources-casestudies-vision-guided-positioning.html)

---

## 2. Eye-in-Hand vs Eye-to-Hand Camera Configurations

### 2.1 Eye-in-Hand (Camera Mounted on End Effector)

**Pros:**
- Camera moves with the tool -- always sees the work area from the tool's perspective
- Higher resolution on the target as the tool approaches
- Natural for servoing: pixel error directly maps to tool offset
- Can see the bit-to-screw relationship directly
- Simpler calibration for a 2-axis system (camera-to-tool offset is fixed)

**Cons:**
- Image changes as carriage moves (motion blur at high speeds)
- Field of view narrows as you get close -- may lose the target
- Camera adds mass/inertia to the moving carriage
- Wiring/ribbon cable must accommodate carriage motion
- Vibration from stepper motors can blur images

**Calibration for Eye-in-Hand:**
- Need to determine the fixed transform from camera optical center to tool tip
- For a 2-axis XZ system: measure X and Z offsets between camera center point and bit center point
- Can be done with a calibration target: move camera until target is centered, record position; move bit to same target, record position; difference = offset
- OpenCV `calibrateHandEye()` supports Tsai-Lenz and other solvers, but designed for 6-DOF robots; for 2-DOF (XZ) a simpler approach works

### 2.2 Eye-to-Hand (Camera Fixed, Looking at Work Area)

**Pros:**
- No added mass on carriage; no cable management issues
- Constant field of view -- always sees the full workspace
- Can see both the tool and the target simultaneously
- No motion blur from carriage movement
- Simpler mechanical integration

**Cons:**
- Fixed resolution -- cannot zoom in as tool approaches
- May have occlusion issues (tool blocks view of screw)
- Requires calibrating camera-to-workspace AND workspace-to-carriage coordinate transforms
- Parallax errors if camera is not perfectly perpendicular to the work plane

**Calibration for Eye-to-Hand:**
- Need to map pixel coordinates to physical XZ coordinates in the carriage's frame
- Use a checkerboard or ArUco markers at known positions
- Move carriage to known positions, record pixel locations of a fiducial on the tool
- Build a homography or affine transform: pixel (u,v) -> physical (X,Z)
- Must account for lens distortion (calibrate intrinsics first)

### 2.3 Recommendation for Your System

**Use eye-in-hand** for the following reasons:
1. Your camera is already on the end effector
2. You have a 2-axis XZ system -- the camera-to-bit offset is a simple fixed (dx, dz) in mm
3. IBVS is natural: center the screw in the image, then offset by the known camera-to-bit distance
4. The ToF sensor also needs to be near the work surface for accurate readings
5. The laser pointer can serve double duty: visible in the camera image, it provides a known reference point for calibration

**Simple calibration procedure for your 2-axis system:**
1. Home both axes
2. Place a calibration target (printed dot or crosshair) at the work surface
3. Jog X/Z until the target is centered in the camera image -- record position (X_cam, Z_cam)
4. Jog X/Z until the screwdriver bit tip is over the target -- record position (X_bit, Z_bit)
5. Camera-to-bit offset: dx = X_bit - X_cam, dz = Z_bit - Z_cam
6. During operation: center screw in image, then move (dx, dz) to position bit over screw
7. Measure mm-per-pixel: move X by a known amount (e.g., 10 mm), measure pixel shift, compute scale

**Laser pointer calibration trick:**
- The laser dot appears in the camera image at a known offset from the bit
- If the laser is rigidly mounted, its position in the image gives a reference for the bit's position
- This can self-calibrate: the laser dot position in the image should be constant if the camera-laser-bit geometry is rigid

Sources:
- [Accuracy Evaluation of Hand-Eye Calibration Techniques](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0273261)
- [Robot Hand-Eye Calibration Explained (Aivon)](https://www.aivon.com/blog/robotics-artificial-intelligence/robot-hand-eye-calibration-explained/)
- [Practical Guide to 3D Hand-Eye Calibration (Zivid)](https://medium.com/zivid/the-practical-guide-to-3d-hand-eye-calibration-3c29c0148f62)
- [Hand-Eye Calibration for Gantry Robots (Mech-Mind)](https://docs.mech-mind.net/en/suite-software-manual/latest/vision-calibration/calib-truss-reference.html)
- [OpenCV calibrateHandEye](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [GitHub: easy_handeye](https://github.com/IFL-CAMP/easy_handeye)

---

## 3. Torque Control and Stall Detection for DC Motor Screw Driving

### 3.1 Current-Torque Relationship in DC Motors

The fundamental equation for a brushed DC motor:

```
T = Kt * Ia
```

Where:
- T = torque (Nm)
- Kt = motor torque constant (Nm/A), from the motor datasheet
- Ia = armature current (A)

This is a **linear relationship** -- current is directly proportional to torque. By measuring motor current, you directly measure the torque being applied to the screw.

### 3.2 Current Monitoring Approaches

**Shunt Resistor + ADC:**
- Place a low-value resistor (0.1-1.0 ohm) in series with the motor
- Measure voltage drop: V = I * R_sense
- Feed to microcontroller ADC
- Pros: simple, cheap, fast
- Cons: wastes power, needs amplification for low currents, susceptible to noise

**INA219 (I2C Current/Power Monitor):**
- Integrates shunt resistor (0.1 ohm), 12-bit ADC, and I2C interface
- Measures up to 3.2A with 0.8 mA resolution
- Up to 26V bus voltage
- Sampling rate: up to ~1 kHz (limited by I2C speed and conversion time)
- Pros: easy to use, digital output, measures both current and voltage
- Cons: I2C latency (~1ms per reading at 400kHz), limited dynamic range
- **Well-suited for your application** given moderate DC motor currents

**INA226 (Higher Precision Alternative):**
- 16-bit ADC (vs 12-bit on INA219)
- Higher accuracy: 0.1% gain error
- Alert pin for threshold-based interrupts
- Better for detecting small current changes

**Motor Driver with Integrated Current Sensing (e.g., DRV8251A):**
- IPROPI pin outputs a current proportional to motor current via internal current mirror
- Connect to ADC with external resistor to set measurement range
- Enables stall detection without external current sensor
- Also supports current limiting (I_TRIP threshold)

### 3.3 Detecting "Screw is Tight"

**Current Profile During Screw Tightening (Three Phases):**

```
Current
  ^
  |         Phase 3: Tightening
  |         (current rises steeply)
  |        /
  |       /
  |------/          Phase 2: Rundown
  |                 (steady moderate current)
  |  _______________
  | /
  |/ Phase 1: Engagement
  |  (brief high inrush, then settle)
  +---------------------------------> Time/Angle
```

1. **Engagement Phase:** Initial contact between bit and screw. Brief current spike as motor starts. Current settles to a low baseline (friction only, no clamping load).

2. **Rundown/Threading Phase:** Screw turns freely in the threads. Relatively constant, moderate current. Variations from thread friction. Duration depends on thread engagement length.

3. **Tightening/Clamping Phase:** Screw head seats against the conductor/clamp plate. Current rises as torque increases. The rate of rise depends on the joint stiffness. **Target torque = target current = Kt * I_target**

**Detection strategy:**
- Set a current threshold corresponding to the target torque
- When current exceeds threshold for a sustained period (e.g., 50-100 ms), consider the screw tight
- Also monitor the rate of current rise (dI/dt): a steep rise indicates the screw is seating
- Stop the motor when either threshold is reached

### 3.4 Distinguishing Failure Modes

**"Screw is tight" (correct):**
- Current rises smoothly and stays above threshold
- Motor speed drops toward zero (back-EMF decreases)
- Torque signature: monotonic increase followed by plateau

**"Bit slipped off screw" (cam-out):**
- Sudden drop in current to near-zero or free-running level
- Motor speed suddenly increases (load removed)
- Torque signature: was increasing, then sudden drop
- Detection: monitor for rapid current decrease (dI/dt << 0)

**"Cross-threading":**
- Higher-than-normal current during rundown phase
- Irregular current fluctuations (thread interference)
- Current is elevated but screw doesn't advance smoothly
- Detection: current exceeds "rundown torque MAX" threshold during the threading window
- Per Ingersoll Rand's prevailing torque algorithm: divide tightening into 3 zones; if current exceeds threshold in Zone 1 (cut-in), flag cross-threading

**"Stripped threads / stripped drive recess":**
- Current fluctuates rhythmically (bit catching and slipping repeatedly)
- CNN analysis of torque signal can detect this with 99% accuracy after just 330 degrees of rotation
- Pattern: oscillating torque signal rather than steady or increasing

### 3.5 Practical Implementation

**Sampling requirements:**
- Sample current at the midpoint of the PWM ON period (or OFF period) to get average current
- Center-aligned PWM mode simplifies this timing
- Sampling rate: 100-1000 Hz is adequate (100 Hz = one sample per ~3.6 degrees at 60 RPM)

**Inrush current handling:**
- Motor inrush current is much higher than running current
- TI recommends: ignore current readings above stall threshold for t_INRUSH period
- Stall delay should be at least 1.5x the inrush time
- Typical inrush time: 10-50 ms for small DC motors

**Hardware recommendation for your system:**
- INA219 on the DC motor power line, read via I2C from the Pico/ESP32
- Sample at ~200 Hz (5 ms interval)
- Software low-pass filter (moving average, 5-10 samples)
- Three thresholds:
  1. FREE_RUNNING (engagement/rundown baseline)
  2. TIGHTENING_DETECTED (current exceeds 2x baseline)
  3. TARGET_TORQUE (screw fully tight, stop motor)
- Also detect cam-out: current drops below FREE_RUNNING * 0.5 during tightening phase

Sources:
- [TI FAQ: Motor Stall Detection Using Current Sensing](https://e2e.ti.com/support/motor-drivers-group/motor-drivers/f/motor-drivers-forum/1065786/faq-how-to-detect-motor-stall-using-current-sensing)
- [Source Robotics: Torque Control of DC Motor](https://source-robotics.com/blogs/blog/torque-control-of-dc-motor)
- [INA219 Current Sensor Primer (EDN)](https://www.edn.com/ina219-current-sensor-module-primer/)
- [DEPRAG: Tightening Processes in Screwdriving](https://www.deprag.com/en/screwdriving-technology/technical-information/tightening-processes.html)
- [Celo Fasteners: Torque Curve Analysis](https://www.celofasteners.com/en/content/152-torque-curve-analysis)
- [Ingersoll Rand: Cross-Thread Detection with Prevailing Torque](https://irtoolhelp.ingersollrand.com/hc/en-us/articles/360045794653)
- [Atlas Copco: Cross Threads](https://www.atlascopco.com/en-us/itba/expert-hub/in-the-lab/cross-threads)
- [ML-Based Screw Drive State Detection (dataset)](https://github.com/AAAipa/dataset_unfastening)

---

## 4. Stepper Motor Homing and Positioning with Limit Switches

### 4.1 Homing Sequence Best Practices

**Standard Two-Phase Homing Procedure:**

1. **Phase 1 -- Fast approach:** Move toward the limit switch at moderate speed (e.g., 5-10 mm/s). When the switch triggers, stop immediately.

2. **Phase 2 -- Back off slowly:** Reverse direction at slow speed (e.g., 1-2 mm/s) until the switch deactivates. Record this position as home (0,0).

3. **Optional Phase 3 -- Final approach:** Move forward again at very slow speed until the switch just triggers. This gives the most repeatable home position.

**Why two phases:** The fast approach gets you close quickly; the slow back-off gives precision. The switch activation point has some hysteresis, so the deactivation point (backing off) is more repeatable than the activation point (approaching at speed due to deceleration distance).

**Speed settings:**
- "Homing speed towards": 5-10x faster than "homing speed away"
- "Homing speed away": the slow, precise phase
- Keep homing speed low enough that the motor can stop within 1-2 steps of the switch triggering

### 4.2 Limit Switch Types and Considerations

**Mechanical microswitches:**
- Cheap, reliable, definite click
- Activation force: 0.5-2N typically
- Hysteresis: 0.1-0.5 mm (distance between activation and deactivation)
- Debouncing required: hardware (RC filter, 100nF + 10K) or software (2-5 ms delay)
- Wear: rated for 1M+ cycles typically
- **Risk:** can bounce and give false readings if not debounced

**Hall-effect sensors (magnetic proximity):**
- No mechanical contact -- no wear, no bounce
- Mount a small magnet on the carriage; sensor detects proximity
- Activation distance: 2-10 mm (adjustable by magnet strength)
- No debouncing needed
- More repeatable than mechanical switches
- **Recommended** for better repeatability

**Optical endstops:**
- Slot-type optical sensor with flag on carriage
- Very fast response, no bounce
- Can be affected by dust/debris in dirty environments
- Good repeatability

### 4.3 Maintaining Position Accuracy (Open Loop)

**Causes of missed steps:**
1. Acceleration too high (exceeds motor torque at that speed)
2. Speed too high (motor torque drops off above a threshold RPM)
3. Mechanical binding or obstruction
4. Insufficient motor current (driver current limit set too low)
5. Resonance at certain speeds (typically around 100-200 full steps/sec for NEMA 17)

**Prevention strategies:**
- Use acceleration ramps (trapezoidal or S-curve profiles) -- never start at full speed
- Stay well within the motor's torque-speed curve (use 60-70% of rated torque at operating speed)
- Avoid resonance frequencies: use microstepping (1/8 or 1/16) which smooths out resonance
- Adequate motor current: set driver VREF to match motor's rated current
- Minimize friction: lubricate linear guides, ensure alignment

**If using TMC2209 (your driver):**
- StallGuard4 provides sensorless stall detection by monitoring back-EMF
- DIAG pin goes high on stall detection -- can trigger interrupt
- StallGuard works reliably at 400-2000 Hz step frequency
- CoolStep can dynamically reduce current when loads are low, saving power
- Sensorless homing achievable with 3-sigma repeatability of +/- 0.01 mm (per Andrea Favero's study)

### 4.4 Microstepping Accuracy Reality Check

**Key finding from Hackaday testing:**

Microstepping does NOT improve positional accuracy -- it improves smoothness and reduces vibration/noise.

- Full-step accuracy: +/- 5% of step angle (i.e., +/- 0.09 degrees for 1.8-degree motor)
- 1/16 microstepping: same positional error envelope, just smoother motion within it
- Under load, deflection can exceed half a full step (~0.9 degrees)
- The A4988 driver showed the best microstep linearity; DRV8825 showed significant nonlinearity at microstep boundaries

**Practical implications for your system:**
- Use 1/8 or 1/16 microstepping for smooth motion and reduced vibration
- Don't rely on 1/32 or 1/256 microstepping for actual positioning accuracy
- With a typical lead screw (2 mm/rev, 200 steps/rev, 1/16 microstepping):
  - Theoretical resolution: 2 / (200 * 16) = 0.000625 mm (0.625 um)
  - Practical accuracy: +/- 5% of full step = +/- 0.05 mm (50 um)
  - With closed-loop correction: +/- 0.005 mm achievable
- **For screw driving, +/- 0.5 mm accuracy is sufficient** -- well within open-loop stepper capability

### 4.5 Homing on Every Power Cycle

- Always home on power-up -- stepper motors have no absolute position memory
- Home sequence: Z first (move up to clear workspace), then X
- After homing, set software position counters to 0
- Implement soft limits in software to prevent crashing into physical limits during normal operation

Sources:
- [DroneBot Workshop: Stepper Motor with Hall Effect Homing](https://dronebotworkshop.com/stepper-motor-hall-effect/)
- [Pololu: Setting Up Limit Switches and Homing](https://www.pololu.com/docs/0J71/4.14)
- [CuriousScientist: AccelStepper Homing with Limit Switch](https://curiousscientist.tech/blog/accelstepper-tb6600-homing)
- [Brainy-Bits: Setting HOME Position at Startup](https://www.brainy-bits.com/post/how-to-set-the-home-position-of-a-stepper-at-startup)
- [Machine Design: Why Open-Loop Steppers Lose Steps](https://www.machinedesign.com/motors-drives/article/21833271/why-open-loop-steppers-lose-steps-and-how-to-solve-the-problem)
- [EDN: Closed-Loop Stepper Control](https://www.edn.com/no-more-missed-steps-unlocking-precision-with-closed-loop-stepper-control/)
- [Hackaday: How Accurate Is Microstepping Really?](https://hackaday.com/2016/08/29/how-accurate-is-microstepping-really/)
- [GitHub: stepper_sensorless_homing (TMC2209)](https://github.com/AndreaFavero71/stepper_sensorless_homing)
- [TMC2209 Datasheet](https://www.analog.com/en/products/tmc2209.html)
- [Klipper: TMC Drivers Documentation](https://www.klipper3d.org/TMC_Drivers.html)

---

## 5. ToF Sensor + Camera Fusion for Z-Axis Positioning

### 5.1 ToF Sensor Characteristics (VL53L0X / VL53L1X / VL53L4CD)

**VL53L0X:**
- Range: 30-1200 mm (1200 mm max practical, 2000 mm theoretical)
- Update rate: up to 50 Hz
- Accuracy: +/- 3% at best, +/- 10% in poor conditions
- At 100 mm: accuracy is +/- 3 to 10 mm
- Noise increases significantly above 380 mm
- Field of view: ~25 degrees cone

**VL53L1X (next gen):**
- Range: 30-4000 mm
- Up to 50 Hz update rate
- Better ambient light rejection
- Programmable ROI (region of interest)
- Better accuracy than VL53L0X at same range

**VL53L4CD (your sensor, based on your codebase):**
- Range: 1-1300 mm
- Optimized for short-range (up to 200 mm)
- Higher accuracy at close range than VL53L0X/L1X
- Up to 100 Hz update rate
- Better for your close-range approach application

### 5.2 Improving ToF Accuracy

**Measurement budget (timing budget):**
- Higher budget = more integration time = less noise
- 20 ms: variance 0.057 (noisy)
- 50 ms: variance 0.011 (5x improvement)
- 60 ms: 0.010 (diminishing returns beyond 50 ms)
- Recommendation: use 50 ms budget for approach phase (20 Hz update)

**Averaging/filtering:**
- Moving average of 5-10 readings reduces noise significantly
- Median filter (take median of 5 readings) rejects outliers
- Kalman filter for optimal fusion with motion model

**Calibration:**
- Individual sensors have systematic offset errors (can be +/- 5-10 mm)
- Calibrate at a known distance: measure offset, apply correction
- Offset can vary with target reflectance and color

### 5.3 Fusion Strategy: Camera + ToF

**Role of each sensor:**
- **Camera:** provides precise XZ alignment (lateral positioning) via image-based servoing
- **ToF:** provides Z-depth (distance from tool to work surface) for approach planning
- **Together:** camera handles "where to go laterally," ToF handles "how far away is the surface"

**Approach Sequence Using Fusion:**

```
Phase 1: SEARCH (camera only)
  - ToF reading > 100 mm (far from surface)
  - Camera scans for screw head
  - Move XZ to center screw in frame

Phase 2: APPROACH (camera + ToF)
  - ToF: 100 mm -> 20 mm, controlled descent
  - Camera: continuous servoing to maintain XZ alignment
  - Speed proportional to ToF distance (slow down as you get closer)
  - Camera provides increasingly precise alignment as resolution improves

Phase 3: FINAL POSITIONING (camera + ToF)
  - ToF: 20 mm -> 5 mm, very slow approach
  - Camera: sub-pixel alignment corrections
  - ToF confirms when bit is at expected engagement distance

Phase 4: ENGAGEMENT (ToF + current sensing)
  - ToF: < 5 mm, bit should be touching screw
  - Start motor rotation
  - Monitor current for engagement confirmation
  - ToF readings may become unreliable this close (below minimum range)
```

**Registration/Calibration between camera and ToF:**
- Mount ToF adjacent to camera on the same bracket
- ToF gives distance along its axis (approximately the camera's optical axis)
- No complex fusion math needed -- they measure orthogonal things:
  - Camera: X,Z position in the lateral plane
  - ToF: depth (distance to surface along approach axis)

### 5.4 Alternative: Using Camera for Depth Estimation

If the ToF has issues at very close range (<30 mm), the camera can estimate depth via:
- **Focus-based:** Blur changes with distance (requires a lens with narrow depth of field)
- **Size-based:** Known screw head size -> apparent size in pixels -> distance
- **Stereo:** Not applicable with single camera, but the laser dot position shift can give crude depth info

Sources:
- [VL53L0X Datasheet (ST)](https://www.st.com/en/imaging-and-photonics-solutions/vl53l0x.html)
- [Improving VL53L0X Accuracy (Paynter)](https://www.fpaynter.com/2022/11/improving-vl53l0x-measurement-accuracy-precision/)
- [ToF Cameras in Industrial Robots (ToFSensors)](https://tofsensors.com/blogs/tof-sensor-knowledge/tof-cameras-in-industrial-robots-from-positioning-to-avoidance)
- [Sensor Fusion for Depth Estimation including ToF (IEEE)](https://ieeexplore.ieee.org/document/6375030/)
- [High-Resolution Depth Maps Based on ToF-Stereo Fusion](https://arxiv.org/abs/2107.14688)
- [Novel Hand-Eye Calibration Based on ToF Camera (Frontiers)](https://www.frontiersin.org/articles/10.3389/fpls.2022.1099033/full)

---

## 6. Circuit Breaker Screw Terminal Anatomy

### 6.1 Screw Terminal Design

Residential circuit breakers (15A, 20A -- Square D QO, Homeline, Eaton BR, Siemens QP) use a **pressure plate / clamp plate** design:
- A screw presses a clamp plate down onto the inserted wire
- The screw sits in a threaded hole in the breaker body
- The wire is inserted into a rectangular opening below the screw

The screw is typically recessed in a small well/pocket in the breaker housing, making it partially enclosed. This is important for your vision system -- the screw head may not be fully visible from all angles.

### 6.2 Screw Specifications

**Screw thread size:**
- Most residential branch breakers (15A, 20A): **#8-32** thread
- Larger breakers (30A+): **#10-32** thread
- Main breakers (100A+): **1/4"-20** or larger, often hex socket head

**Screw head type:**
- Most common: **combination slot/square (Robertson)** drive
  - Has both a slotted groove AND a square recess
  - Accepts flat-blade screwdriver, Robertson (#2 square), or ECX combination bits
  - ECX 1: #1 square + 1/4" (6.35 mm) slotted blade
  - ECX 2: #2 square + 9/32" (7.14 mm) slotted blade
- Some older breakers: pure slotted drive
- Some commercial breakers: hex socket (Allen)

**Screw head dimensions (per ASME B18.6.3):**

For **#8** screws:
- Head diameter: 0.332" (8.4 mm)
- Slot width: 0.054" (1.37 mm) for flat/round head; 0.029"-0.045" range for 82-degree flat head
- Slot depth: 0.036"-0.045" (0.9-1.1 mm)

For **#10** screws:
- Head diameter: 0.385" (9.8 mm)
- Slot width: 0.060" (1.52 mm) for flat/round head; 0.034"-0.050" range for 82-degree flat head
- Slot depth: 0.042"-0.053" (1.1-1.3 mm)

### 6.3 Torque Specifications

**Residential 15-20A branch breakers:**
- Square D QO: **20-25 in-lbs** (2.3-2.8 Nm) -- embossed on breaker side
- Square D Homeline: **22-35 in-lbs** (2.5-4.0 Nm)
- General: 20 in-lbs (2.3 Nm) is the standard per UL 486A-B for #10 slotted head screws with 12 AWG wire

**Per UL 486A-B Table 21:**
- Size 10 slotted head screw, 12 AWG wire: **20 in-lbs** (2.3 Nm)
- Slot width ≤ 1.2 mm (0.047"), slot length ≤ 6.4 mm (0.25")

**Larger breakers:**
- 30-60A: 35-45 in-lbs (4.0-5.1 Nm)
- 100A main: ~250 in-lbs (28 Nm) -- requires different tooling

### 6.4 Practical Implications for Your System

**Screwdriver bit:**
- Use a 6 mm (1/4") slotted bit or ECX 1 combination bit
- Bit width must be slightly less than the slot width (~1.2-1.5 mm engagement)
- A flat-blade bit is simplest mechanically

**Torque requirement:**
- Target: 20-25 in-lbs (2.3-2.8 Nm) for 15-20A breakers
- Your DC motor + gearing must produce this torque at the bit
- With a geared DC motor (e.g., 100:1 gear ratio), a motor producing 0.025 Nm can deliver 2.5 Nm at the bit
- Account for gear efficiency (~50-70%): need ~0.04-0.05 Nm from the motor

**Visual detection challenges:**
- Screw is often partially recessed in the breaker housing
- Combination slot/square head creates a cross-shaped feature (not just a circle)
- Lighting is critical -- direct illumination may cause reflections from the shiny screw head
- Consider ring light or angled LED illumination to avoid specular reflection

Sources:
- [Schneider Electric: QO Breaker Torque Requirements](https://www.se.com/us/en/faqs/FA128927/)
- [E-Hazard: Torque Talk](https://e-hazard.com/lets-talk-torque-again/)
- [UL 486A-B Torque Tables (UpCodes)](https://up.codes/viewer/california/ca-electric-code-2019/chapter/annex_i_/annex-i-recommended-tightening-torque-tables-from-ul-standard-486a-b)
- [IAEI Magazine: Inspecting Electrical Connections for Proper Torque](https://iaeimagazine.org/2015/januaryfebruary-2015/inspecting-electrical-connections-for-proper-torque/)
- [Klein Tools 612-4: Terminal Block Screwdriver](https://www.kleintools.com/catalog/fixed-blade-screwdrivers/terminal-block-screwdriver-18-inch-cabinet-tb-din)
- [Wiha Slotted Screwdriver and Bit Sizes](https://www.wihatools.com/pages/slotted-sizes)
- [ASME B18.6.3 Flat Head Screw Dimensions (Engineers Edge)](https://www.engineersedge.com/flat_head_screw.htm)
- [Carling Tech: Circuit Breaker Termination](https://www.carlingtech.com/circuit-termination)

---

## 7. Existing Open-Source and Research Projects

### 7.1 GitHub Repositories

**ScrewDrivingBot1 (DaviddNie/ScrewDrivingBot1)**
- Platform: UR5e collaborative robot
- Vision: depth camera on end effector
- Detection: blob/centroid detection in Python for screw hole localization
- Control: ROS 2 Humble, MoveIt for trajectory planning
- Architecture: brain module coordinates vision and movement services
- End effector: Arduino-controlled motor for screw driving
- Won 2024 UNSW MTRN4231 competition
- https://github.com/DaviddNie/ScrewDrivingBot1

**screw_detection (eyildiz-ugoe/screw_detection)**
- Task: detect screws in images for electronics disassembly/recycling
- Architecture: Xception, ResNet101V2, ResNeXt101, InceptionResNetV2, DenseNet201
- Best accuracy: 98.71% (Xception)
- Uses TensorFlow 2.x, transfer learning from ImageNet
- Input sizes: 64x64 to 221x221 pixels
- Supports standalone detection and ROS integration
- https://github.com/eyildiz-ugoe/screw_detection

**dataset_unfastening (AAAipa/dataset_unfastening)**
- Dataset of 1000+ unfastening operations
- Torque and angle measurements from manual and robotic test benches
- Screw types: M4, M5, M6, M8 with Torx, Phillips, External/Internal Hex
- Two classes: releasable vs non-releasable (stripped)
- Hardware: Bosch Rexroth Nexo, DEPRAG AST40
- JSON format, angle resolution: 5.320 degrees
- https://github.com/AAAipa/dataset_unfastening

**stepper_sensorless_homing (AndreaFavero71)**
- TMC2209 + RP2040-Zero + NEMA 17
- Sensorless homing using StallGuard4
- Achieved 3-sigma repeatability of +/- 0.01 mm
- 1/8 microstepping (1600 pulses/rev)
- MicroPython + PIO implementation
- Reliable in 400-2000 Hz step frequency range
- https://github.com/AndreaFavero71/stepper_sensorless_homing

### 7.2 Key Research Papers

**"From Perception to Precision: Vision-Based Mobile Robotic Manipulation for Assembly Screwdriving" (2025)**
- Monocular RGB pipeline: object segmentation, pose estimation, CAD-based screw hole localization
- 0.21-0.50 mm translational accuracy, 0.07-0.20 degree angular accuracy
- 100% success rate over 400 screw insertions
- https://www.sciencedirect.com/science/article/pii/S0736584525002029

**"Autonomous Robotic Screwdriving for High-Mix Manufacturing" (2025)**
- Passively compliant rotary tools + 3D vision + force sensing
- Handles part-to-part variation without re-programming
- https://www.sciencedirect.com/science/article/abs/pii/S0736584525002261

**"Machine Learning Based Screw Drive State Detection" (2022)**
- CNN detects stripped screw drives after only 330 degrees (< 1 revolution)
- 99% accuracy classifying releasable vs stripped
- Torque signal features used for classification
- https://www.sciencedirect.com/science/article/pii/S0278612522001248

**"Robust Fastener Detection Based on Force and Vision" (2023)**
- Combines force sensing and vision for screw detection in robotic screwing/unscrewing
- Multi-modal approach improves reliability
- https://www.mdpi.com/1424-8220/23/9/4527

**"An Automated Four-DOF Robot Screw Fastening Using Visual Servo" (2011)**
- Classic paper: 4-DOF robot + dual cameras for visual servo screw fastening
- Two cameras measure position error in real time
- https://www.researchgate.net/publication/251989311

**"High-Precision Transformer-Based Visual Servoing" (2025)**
- DET architecture for sub-millimeter alignment of tiny objects
- No manual feature design or fiducial markers needed
- https://arxiv.org/html/2503.04862v2

### 7.3 Commercial Systems for Reference

| System | Approach | Key Feature |
|--------|----------|-------------|
| Robotiq SD-100 | UR cobot + vacuum + compliant sleeve | Auto screw feeding, force-based insertion |
| OnRobot Screwdriver | Integrated sensor suite | Detects pick, length, proper insertion |
| DEPRAG Robotic | Multi-spindle + vision | Thread monitoring, torque/angle control |
| Nitto Seiko | Vision-guided positioning | Camera locates holes, offsets communicated |
| Visumatic SCARA | 4-axis SCARA + auto feeder | High-speed, dedicated screwdriving |
| Mountz MD-Series | DC servo + auto shutoff clutch | Precision torque control |

Sources:
- [GitHub Topics: Screwdriving](https://github.com/topics/screwdriving)
- [Robotiq Screwdriving Solution](https://robotiq.com/solutions/screwdriving)
- [OnRobot Screwdriver](https://onrobot.com/en/products/onrobot-screwdriver)
- [DEPRAG Robotic Screwdriving](https://www.depragusa.com/Robotic-Screwdriving.html)
- [igus: Screwdriving Robots Revolution](https://toolbox.igus.com/8667/screwdriving-robots)

---

## 8. Control Architectures for Multi-Stage Precision Tasks

### 8.1 State Machine Approach

**Best for:** Your system's complexity level (5-10 states, well-defined sequence)

**State machine for your screw driving task:**

```
IDLE -> HOMING -> SEARCHING -> ALIGNING -> APPROACHING -> ENGAGING -> TIGHTENING -> VERIFYING -> COMPLETE
  |                                                                                                  |
  +----------- ERROR_RECOVERY <--- (any state can transition here on failure) ----<------------------+
```

**States in detail:**

```
1. IDLE
   - System powered on, waiting for command
   -> HOMING

2. HOMING
   - Execute limit switch homing sequence for X, Z
   - Transitions: success -> SEARCHING, failure -> ERROR_RECOVERY

3. SEARCHING
   - Camera active, scanning for screw head
   - Move X axis to sweep field of view if needed
   - Transitions: screw detected -> ALIGNING, timeout -> ERROR_RECOVERY

4. ALIGNING
   - IBVS loop: minimize pixel error between screw center and target
   - Camera + ToF active, steppers controlled by visual servo
   - Transitions: pixel error < threshold -> APPROACHING, lost target -> SEARCHING

5. APPROACHING
   - Move Z axis down toward work surface
   - ToF monitors distance, camera maintains XZ alignment
   - Speed proportional to remaining distance
   - Transitions: ToF < engagement_distance -> ENGAGING, lost alignment -> ALIGNING

6. ENGAGING
   - Start DC motor rotation (slow, CCW to find slot, then CW)
   - Apply light Z pressure via stepper
   - Monitor current for engagement signature
   - Transitions: engagement_confirmed -> TIGHTENING, timeout -> ERROR_RECOVERY

7. TIGHTENING
   - DC motor running CW at tightening speed
   - Monitor current continuously
   - Transitions: target_torque_reached -> VERIFYING, cam_out -> ENGAGING, cross_thread -> ERROR_RECOVERY

8. VERIFYING
   - Stop motor
   - Optional: reverse slightly, check current, re-tighten
   - Retract Z axis
   - Transitions: verified -> COMPLETE, failed -> ERROR_RECOVERY

9. COMPLETE
   - Log success
   - Retract to safe position
   -> IDLE

10. ERROR_RECOVERY
    - Stop all motors
    - Retract Z to safe height
    - Log error type and state where failure occurred
    - Transitions: retracted -> IDLE (await operator intervention)
```

**Pros of state machine:**
- Simple to implement and debug
- Low overhead, no libraries needed
- Easy to add logging at each transition
- Deterministic behavior
- Well-suited for 8-10 states

**Cons:**
- Gets unwieldy beyond ~15 states
- Adding new behaviors requires modifying transition table
- Hard to reuse sub-sequences

### 8.2 Behavior Tree Approach

**Better for:** If you plan to expand the system (multiple screw locations, different breaker types, error recovery sub-trees)

**Structure for your task:**

```
Root (Sequence)
├── Home Axes (Action)
├── Find Screw (Fallback)
│   ├── Detect Screw in FOV (Condition)
│   └── Scan for Screw (Action)
├── Align to Screw (Action, repeats until converged)
│   └── Visual Servo Loop
├── Approach Surface (Sequence)
│   ├── Check ToF Distance (Condition)
│   ├── Move Z Down (Action)
│   └── Maintain XZ Alignment (Action)
├── Engage and Tighten (Sequence)
│   ├── Start Motor (Action)
│   ├── Monitor Current (Decorator: repeat until success/failure)
│   │   ├── Check Target Torque (Condition)
│   │   └── Check Cam Out (Condition -> Retry Engagement)
│   └── Stop Motor (Action)
└── Retract (Action)
```

**Python libraries:**
- **py_trees** (Python, by Daniel Stonier): mature, well-documented, good for prototyping
  - `pip install py_trees`
  - Supports blackboard for shared data
  - Visualization tools
  - ROS2 integration available (py_trees_ros)
  - https://github.com/splintered-reality/py_trees

- **BehaviorTree.CPP** (C++, by Faconti/Colledanchise): production-grade, XML-based
  - Better performance, stronger typing
  - Groot2 GUI for visual editing
  - Overkill for your application unless you're already in C++

**Pros of behavior trees:**
- Modular -- easy to add new behaviors as subtrees
- Fallback nodes handle error recovery naturally
- Can visualize the tree structure
- Reactive -- can interrupt and re-evaluate conditions

**Cons:**
- More upfront design effort
- Blackboard data management more complex
- Library dependency
- Overkill for a simple sequential task

### 8.3 Hybrid Approach (Recommended)

**Use a state machine for the top-level sequence, with specialized sub-routines for complex phases:**

```python
class ScrewDrivingStateMachine:
    states = [IDLE, HOMING, SEARCHING, ALIGNING, APPROACHING,
              ENGAGING, TIGHTENING, VERIFYING, COMPLETE, ERROR]

    def run(self):
        while True:
            if self.state == HOMING:
                self.state = self.do_homing()        # Simple sequential
            elif self.state == SEARCHING:
                self.state = self.do_search()         # Camera + scan pattern
            elif self.state == ALIGNING:
                self.state = self.do_visual_servo()   # PID control loop
            elif self.state == APPROACHING:
                self.state = self.do_approach()       # ToF + camera fusion
            elif self.state == TIGHTENING:
                self.state = self.do_tighten()        # Current monitoring loop
            ...
```

Each sub-routine handles its own internal logic (PID loops, sensor fusion, threshold monitoring) and returns the next state.

**Practical tips:**
- Run the main state machine loop at ~50-100 Hz
- Each state handler is responsible for reading its relevant sensors and commanding its actuators
- Use a shared data structure (dict or dataclass) for sensor readings, not global variables
- Log every state transition with timestamp for debugging
- Implement a watchdog timer: if any state takes too long, force transition to ERROR_RECOVERY
- For the visual servo loop (ALIGNING state): run as a tight inner loop at camera framerate

### 8.4 Control Loop Architecture

```
+-------------------+     +------------------+     +------------------+
|   State Machine   | --> | Visual Servo     | --> | Stepper          |
|   (10-50 Hz)      |     | Controller       |     | Motion Control   |
|                   |     | (10-30 Hz)       |     | (1-10 kHz)       |
+-------------------+     +------------------+     +------------------+
        |                        ^                        ^
        v                        |                        |
+-------------------+     +------------------+     +------------------+
|   Sensor Fusion   | --> | Camera           |     | TMC2209 UART     |
|   (combines ToF,  |     | Processing       |     | StallGuard,      |
|    camera, current)|     | (OpenCV)         |     | Step/Dir         |
+-------------------+     +------------------+     +------------------+
        ^
        |
+-------------------+
|   Current Monitor |
|   (INA219, 200Hz) |
+-------------------+
```

**Communication between layers:**
- State machine reads sensor fusion output, decides actions
- Visual servo controller receives "align to target" command, outputs stepper velocity commands
- Stepper motion control executes velocity/position commands via step/dir pulses
- Current monitor runs independently, sets flags for state machine (STALL, CAM_OUT, TARGET_REACHED)

Sources:
- [State Machines vs Behavior Trees (Polymath Robotics)](https://www.polymathrobotics.com/blog/state-machines-vs-behavior-trees)
- [Behavior Trees and State Machines in Robotics Applications (IEEE TSE)](https://dl.acm.org/doi/abs/10.1109/TSE.2023.3269081)
- [py_trees Documentation](https://py-trees.readthedocs.io/en/devel/introduction.html)
- [BehaviorTree.CPP](https://www.behaviortree.dev/)
- [Toyota Research: Task Behavior Engine](https://github.com/ToyotaResearchInstitute/task_behavior_engine)
- [Behavior Trees in Industrial Applications](https://arxiv.org/html/2403.19602v1)
- [Hierarchical and State-based Architectures for Robot Behavior](https://arxiv.org/pdf/1809.11067)
- [Autonomous Robot Task Execution Using PDDL and Behavior Trees](https://pmc.ncbi.nlm.nih.gov/articles/PMC11504948/)

---

## Summary of Key Recommendations for Your System

### Architecture Decision
Use a **state machine** with 8-10 states. Your task is well-defined and sequential. A behavior tree adds complexity without proportional benefit at this scale.

### Vision Pipeline
Start with **classical CV** (blob detection or Hough circles + contour analysis) for screw detection. At 640x480, 30fps, this is computationally light enough for an ESP32 companion or Raspberry Pi. Only move to YOLO/ML if classical methods prove insufficient.

### Camera Configuration
**Eye-in-hand** with a simple calibration: measure the fixed (dx, dz) offset between camera center and bit tip. Use the laser pointer dot in the camera image as a secondary reference.

### Torque Control
Use an **INA219** on the DC motor power line. Implement three-threshold detection: baseline (free running), tightening detected, target torque reached. Monitor dI/dt for cam-out detection (sudden current drop).

### Steppers
**TMC2209** with 1/8 or 1/16 microstepping. Mechanical limit switches for homing (hall-effect if budget allows). Two-phase homing: fast approach + slow back-off. Open-loop positioning is more than accurate enough (+/- 0.05 mm with good mechanics vs. the +/- 0.5 mm you need).

### Approach Planning
Use **ToF for Z-depth** during approach, **camera for XZ lateral alignment**. Speed proportional to distance. Switch from vision-guided to current-guided control at engagement distance.

### Target Torque
**20-25 in-lbs (2.3-2.8 Nm)** for 15-20A residential breakers. Use a 6mm slotted bit or ECX combination bit. Verify the torque value embossed on the specific breaker.
