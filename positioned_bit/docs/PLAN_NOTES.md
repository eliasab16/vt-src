# Control Plan Review Notes

Notes to batch-apply to CONTROL_PLAN.md.

---

## Homing
- Home Y axis to the bottom (not top)
- Home Z axis to the back (not rear/forward as currently written)

## Axis Terminology

- Y axis = vertical (up/down) — currently called "Z" in the plan
- Z axis = horizontal (forward/back, bit travel) — currently called "X" in the plan
- Update all references throughout the plan

## Camera Placement

- Mount camera to the side of the bit, at the same Y height (same horizontal axis as bit tip)
- Camera looks forward along Z (approach axis), not down from above
- Vertical alignment via center line overlay (or similar approach): when the screw is on the camera's horizontal center line, it's at the same Y as the bit — distance-independent, no calibration needed
- May offset camera slightly in X to avoid bit occluding the screw; this adds a small constant offset, calibrated once
- Camera must be level vertically (no pitch tilt); horizontal yaw toward the bit is fine
