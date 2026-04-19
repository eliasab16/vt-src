"""End-to-end test for RemoteInferenceClient.

Simulates a 30Hz main robot control loop without actual hardware:
- Fakes observations (random images, zero state)
- Calls get_action() each frame and prints received actions
- Validates the full pipeline: setup, WebSocket, inference thread, queue

Usage:
    python client_e2e_test.py <server-ip> <server-tcp-port> <policy-path> [duration_s]

Example:
    python client_e2e_test.py 203.57.40.240 10049 eliasab16/pi05_wire_tip_p1_4k 10
"""

import sys
import time
import pathlib
import numpy as np

# Make remote_inference importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from remote_inference.client import RemoteInferenceClient

if len(sys.argv) < 4:
    print("Usage: python client_e2e_test.py <ip> <tcp_port> <policy_path> [duration_s]")
    sys.exit(1)

IP = sys.argv[1]
PORT = sys.argv[2]
POLICY_PATH = sys.argv[3]
DURATION_S = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0

SERVER_URL = f"ws://{IP}:{PORT}"
FPS = 30
FRAME_PERIOD = 1.0 / FPS

# --- Setup ---
print(f"Connecting to {SERVER_URL}...")
from lerobot.policies.rtc.configuration_rtc import RTCConfig
rtc_config = RTCConfig(enabled=True, execution_horizon=20)
client = RemoteInferenceClient(server_url=SERVER_URL, fps=FPS, rtc_config=rtc_config)

print(f"Loading policy: {POLICY_PATH}")
client.setup(
    policy_path=POLICY_PATH,
    action_dim=8,
    camera_names=["wrist_top", "wrist_bottom", "overhead"],
    task="example task description",
    device="cuda",
)

print("Starting inference thread...")
client.start()

# --- Main loop simulation ---
fake_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
n_frames = int(DURATION_S * FPS)
frames_with_fresh_action = 0
frames_holding_last = 0
last_action = None  # hold-last-position fallback

print(f"\nRunning {n_frames} frames at {FPS}Hz ({DURATION_S}s)...\n")

for i in range(n_frames):
    t0 = time.perf_counter()

    # Update observation (main thread -> inference thread via shared slot)
    client.update_observation(
        state=[0.0] * 8,
        images={"wrist_top": fake_img, "wrist_bottom": fake_img, "overhead": fake_img},
        task="example task description",
    )

    # Pop next action from queue (may be None if queue is empty)
    action = client.get_action()
    if action is not None:
        last_action = action  # remember for fallback
        frames_with_fresh_action += 1
        if i % 30 == 0:
            print(f"[frame {i:4d}] fresh action, first 3 values={action[:3].tolist()}")
    elif last_action is not None:
        # Hold-last-position: queue is empty but we have a previous action
        action = last_action
        frames_holding_last += 1
        if i % 30 == 0:
            print(f"[frame {i:4d}] holding last action (queue empty)")
    else:
        # No action ever received — robot would stay at current position (no-op)
        if i % 30 == 0:
            print(f"[frame {i:4d}] no action yet (warming up)")

    # Maintain 30Hz
    elapsed = time.perf_counter() - t0
    sleep_time = FRAME_PERIOD - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

# --- Teardown ---
print("\nStopping...")
client.stop()

print(f"\n--- Summary ---")
print(f"Frames total:                {n_frames}")
print(f"Frames with fresh action:    {frames_with_fresh_action} ({frames_with_fresh_action / n_frames * 100:.1f}%)")
print(f"Frames holding last action:  {frames_holding_last} ({frames_holding_last / n_frames * 100:.1f}%)")
print("Done.")
