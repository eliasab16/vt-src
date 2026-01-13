# Camera Configuration Reference

## XVLA Policy Camera Mappings

When using XVLA policies with LeRobot, map robot cameras to policy input names:

```json
{
  "observation.images.wrist_top": "observation.images.image",
  "observation.images.overhead_top": "observation.images.image2",
  "observation.images.wrist_bottom": "observation.images.empty_camera_0"
}
```

| Robot Camera | Policy Input |
|--------------|--------------|
| `wrist_top` | `image` (primary) |
| `overhead_top` | `image2` (secondary) |
| `wrist_bottom` | `empty_camera_0` (unused) |

## Bounding Box Detection Padding

Per-camera padding values for wire detection visualization:

| Camera | Padding |
|--------|---------|
| `overhead_top` | 10px |
| `wrist_top` | 15px |
| `wrist_bottom` | 15px |

Padding expands the detected bounding box on all sides to avoid occluding the detected object.
