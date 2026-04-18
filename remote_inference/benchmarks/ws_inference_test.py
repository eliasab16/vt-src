"""Send a dummy observation over WebSocket and measure inference round-trip time."""

import sys
import time
import json
import numpy as np
from websockets.sync.client import connect

if len(sys.argv) < 2:
    print("Usage: python ws_inference_test.py ws://<IP>:<PORT>/ws [num_requests]")
    sys.exit(1)

URL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# Simulate a 640x480 camera frame, then apply the same resize-and-encode pipeline
# that the real client (remote_inference.image_codec.encode_image) would use.
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
from remote_inference.image_codec import encode_image

dummy_img_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
encoded = encode_image(dummy_img_bgr)
print(f"Encoded image size: {len(encoded) / 1024:.1f} KB (per camera)")

dummy_request = json.dumps({
    "state": [0.0] * 8,
    "images": {
        "cam_1": encoded,
        "cam_2": encoded,
        "cam_3": encoded,
    },
    "task": "example task description",
    "timestamp": 0.0,
})
print(f"Total request size: {len(dummy_request) / 1024:.1f} KB\n")

with connect(URL, open_timeout=120, ping_timeout=120) as ws:
    print(f"Connected to {URL}")
    print(f"Sending {N} inference requests...\n")

    for i in range(N):
        t0 = time.perf_counter()
        ws.send(dummy_request)
        response_text = ws.recv()
        total_ms = (time.perf_counter() - t0) * 1000

        response = json.loads(response_text)

        if "error" in response:
            print(f"  [{i+1}/{N}] ERROR: {response['error']}")
            continue

        inference_ms = response.get("inference_time_ms", 0)
        overhead_ms = total_ms - inference_ms  # network + (de)serialization + pre/post-processing
        actions_shape = f"{len(response['actions'])}x{len(response['actions'][0])}"

        print(f"  [{i+1}/{N}] total: {total_ms:.0f}ms | model: {inference_ms:.0f}ms | overhead: {overhead_ms:.0f}ms | actions: {actions_shape}")

print("\nLegend:")
print("  model    = GPU forward pass (predict_action_chunk)")
print("  overhead = network RTT + JSON + base64 + preprocessing + postprocessing")

print("\nDone.")
