"""
This class:
1. runs locally,
2. encodes observations,
3. sends them to the server over WebSocket in a background thread,
4. receives action chunks,
5. and makes them available to the main robot control loop at 30Hz.

Manages two threads:
1. Main thread (30Hz): robot control loop. Pops one action per frame from the queue, sends to robot.
2. Inference thread (background): continuously reads the latest observation, sends to server, merges the returned chunk into the queue.
"""

import threading
import time
import torch
from torch import Tensor

from websockets.sync.client import connect as ws_connect
import requests

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.latency_tracker import LatencyTracker

from remote_inference.protocol import (
    SetupRequest,
    SetupResponse,
    InferenceRequest,
    InferenceResponse,
    SetupStatus,
)
from remote_inference.image_codec import encode_image



class RemoteInferenceClient:
    def __init__(self, server_url, fps, rtc_config: RTCConfig | None = None):
        self.server_url = server_url
        self.fps = fps
        self.rtc_config = rtc_config
        self.ws = None
        self._running = threading.Event()
        self._inference_thread: threading.Thread | None = None
        self._latest_obs: dict | None = None
        self._obs_lock = threading.Lock()  # protects _latest_obs

    def setup(self, policy_path, action_dim, camera_names, task, device):
        setup_request = SetupRequest(
            policy_path=policy_path,
            action_dim=action_dim,
            camera_names=camera_names,
            task=task,
            device=device,
            compile_model=False,  # for now, we let the server decide whether or not to compile the model
        )

        http_url = self.server_url.replace("ws://", "http://").replace("wss://", "https://")
        timeout = 300 # seconds
        try:
            response = requests.post(
                f"{http_url}/setup",
                data=setup_request.model_dump_json(),
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()
            setup_response = SetupResponse.model_validate_json(response.text)
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Setup timed out after {timeout}s. Server may still be loading.")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Cannot connect to {http_url}. Is the server running? ({e})")
        except Exception as e:
            raise RuntimeError(f"Setup failed: {type(e).__name__}: {e}")

        if setup_response.status != SetupStatus.READY:
            raise RuntimeError(f"Setup failed on server: {setup_response.message}")

        self.chunk_size = setup_response.chunk_size
        print(f"Server setup successful. Chunk size: {self.chunk_size}")

    def start(self):
        if self._running.is_set():
            print("Inference thread already running.")
            return
        
        self.ws = ws_connect(f"{self.server_url}/ws", open_timeout=30)
        self._running.set()
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def stop(self):
        self._running.clear()
        if self._inference_thread:
            self._inference_thread.join(timeout=5)
        if self.ws:
            self.ws.close()
            self.ws = None

    def update_observation(self, state, images, task):
        # called by main thread each frame. Encodes images, stores latest in a shared slot.
        self.state = state
        self.images = images
        self.task = task

    def get_action(self) -> Tensor | None:
        pass

    def clear_queue(self):
        pass

    def _inference_loop(self):
        while self._running.is_set():
            time.sleep(0.01)  # placeholder