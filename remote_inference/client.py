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

import json
import threading
import time
import torch
from torch import Tensor

from websockets.sync.client import connect as ws_connect
import requests
from websockets.exceptions import ConnectionClosed

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.latency_tracker import LatencyTracker

from remote_inference.protocol import (
    SetupRequest,
    SetupResponse,
    InferenceRequest,
    InferenceRTCRequest,
    InferenceResponse,
    SetupStatus,
)
from remote_inference.image_codec import encode_image



class RemoteInferenceClient:
    def __init__(
            self,
            server_url,
            fps,
            rtc_config: RTCConfig | None = None,
            min_queue_threshold: int = 8,
            max_queue_threshold: int = 15,
            safety_margin_frames: int = 10,
    ):
        self.server_url = server_url
        self.fps = fps
        self.rtc_config = rtc_config
        self.ws = None
        self._running = threading.Event()
        self._inference_thread: threading.Thread | None = None
        self._latest_obs: dict | None = None
        self._obs_lock = threading.Lock()  # protects _latest_obs
        self._action_queue: ActionQueue | None = None
        self._latency_tracker = LatencyTracker()
        self._safety_margin_frames = safety_margin_frames  # ~0.33s at 30Hz buffer to absorb latency spikes
        self._min_queue_threshold = min_queue_threshold  # always keep at least 8 actions queued (~0.27s)
        self._max_queue_threshold = max_queue_threshold  # never buffer more than 15 actions (~0.5s) — keeps policy reactive to fresh obs
        self._warmup_latency_cutoff_s = 1.0  # ignore inferences slower than this in latency tracker

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

        self._action_queue = ActionQueue(
            cfg=self.rtc_config if self.rtc_config is not None else RTCConfig(),
        )

    def start(self):
        if self._running.is_set():
            print("Inference thread already running.")
            return
        
        # ping_interval=None disables client-side keepalive pings — otherwise a slow
        # first-time warmup inference (30-60s) would cause the connection to be killed
        # before the response arrives.
        self.ws = ws_connect(
            f"{self.server_url}/ws",
            open_timeout=200,
            ping_interval=None,
        )
        self._running.set()
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def stop(self):
        self._running.clear()
        if self.ws:
            self.ws.close()
        if self._inference_thread:
            self._inference_thread.join(timeout=5)
        self.ws = None

    # called by main thread each frame. Encodes images, stores latest in a shared slot.
    def update_observation(self, state, images, task):
        with self._obs_lock:
            self._latest_obs = {
                "state": state,
                "images": images,
                "task": task,
            }

    def get_action(self) -> Tensor | None:
        if self._action_queue is None:
            return None
        return self._action_queue.get()


    def clear_queue(self):
        if self._action_queue is not None:
            self._action_queue.clear()

    def _build_request(self, obs: dict) -> InferenceRequest | InferenceRTCRequest:
        """Build the appropriate request type based on RTC config."""
        base_fields = {
            "state": obs["state"],
            "images": {name: encode_image(img) for name, img in obs["images"].items()},
            "task": obs["task"],
            "timestamp": time.perf_counter(),
        }

        if self.rtc_config is None or not self.rtc_config.enabled:
            return InferenceRequest(**base_fields)
        
        leftover = self._action_queue.get_left_over()
        if leftover is None or len(leftover) == 0:
            # First call — no previous chunk to align to. Use non-RTC endpoint.
            return InferenceRequest(**base_fields)

        return InferenceRTCRequest(
            **base_fields,
            prev_chunk_left_over=leftover.tolist(),
            inference_delay=int((self._latency_tracker.percentile(0.5)) * self.fps),
            execution_horizon=self.rtc_config.execution_horizon,
        )


    def _inference_loop(self):
        """Background thread: pull latest obs, send to server, merge returned chunk into queue."""
        iteration = 0
        skipped_throttle = 0          # throttle skips since last log line
        t_prev_inference_end = None   # wall-clock end of previous inference (for inter-inference dt)
        while self._running.is_set():
            # 1. Grab latest observation (atomic swap — pop it so we don't re-send)
            with self._obs_lock:
                obs = self._latest_obs
                self._latest_obs = None

            if obs is None:
                time.sleep(0.005)  # nothing to process, short sleep
                continue

            # Throttle: skip inference if queue has enough actions to cover the next inference cycle.
            # Threshold adapts to observed p95 latency + safety margin, clamped to [min, max].
            p95_latency_s = self._latency_tracker.percentile(0.95) or 0.2  # default 200ms if no samples
            raw_threshold = int(p95_latency_s * self.fps) + self._safety_margin_frames
            threshold = max(self._min_queue_threshold, min(raw_threshold, self._max_queue_threshold))
            qsize_at_gate = self._action_queue.qsize()
            if qsize_at_gate > threshold:
                skipped_throttle += 1
                time.sleep(0.005)
                continue

            iteration += 1
            t_loop_start = time.perf_counter()
            dt_since_prev_ms = (t_loop_start - t_prev_inference_end) * 1000 if t_prev_inference_end is not None else 0.0

            # 2. Build InferenceRequest (encoding happens here, inside the inference thread)
            t_encode_start = time.perf_counter()
            inference_request = self._build_request(obs)

            encode_ms = (time.perf_counter() - t_encode_start) * 1000

            # 3. Send + receive
            action_index_before_inference = self._action_queue.get_action_index() if self._action_queue else None
            qsize_pre_request = self._action_queue.qsize()
            t_rtt_start = time.perf_counter()
            try:
                self.ws.send(inference_request.model_dump_json())
                try:
                    response_text = self.ws.recv()
                except ConnectionClosed:
                    break  # stop() closed the WebSocket
                payload = json.loads(response_text)
                if "error" in payload:
                    print(f"[inference thread] server error: {payload['error']}")
                    continue
                response = InferenceResponse.model_validate(payload)
            except Exception as e:
                print(f"[inference thread] request/response error: {type(e).__name__}: {e}")
                continue
            rtt_ms = (time.perf_counter() - t_rtt_start) * 1000
            overhead_ms = rtt_ms - response.inference_time_ms
            # Exclude warmup outliers (first few CUDA-compile inferences) from latency tracker
            if rtt_ms / 1000 < self._warmup_latency_cutoff_s:
                self._latency_tracker.add(rtt_ms / 1000)

            # 4. Merge received action chunk into ActionQueue
            real_delay = self._action_queue.get_action_index() - action_index_before_inference if action_index_before_inference is not None else 0
            processed_actions = torch.tensor(response.actions)
            original_actions = torch.tensor(response.original_actions)
            qsize_pre_merge = self._action_queue.qsize()
            chunk_len = processed_actions.shape[0] if processed_actions.dim() > 1 else 1
            self._action_queue.merge(
                original_actions=original_actions,
                processed_actions=processed_actions,
                real_delay=real_delay,
                action_index_before_inference=action_index_before_inference,
            )
            qsize_post_merge = self._action_queue.qsize()

            t_prev_inference_end = time.perf_counter()

            print(
                f"[inference #{iteration}] "
                f"dt_inter={dt_since_prev_ms:.0f}ms | "
                f"encode={encode_ms:.0f}ms | "
                f"model={response.inference_time_ms:.0f}ms | "
                f"overhead={overhead_ms:.0f}ms | "
                f"total={rtt_ms:.0f}ms | "
                f"real_delay={real_delay} | "
                f"thresh={threshold} | "
                f"q_gate={qsize_at_gate} pre={qsize_pre_request} premerge={qsize_pre_merge} post={qsize_post_merge} "
                f"delta={qsize_post_merge - qsize_pre_merge:+d} | "
                f"chunk={chunk_len} | "
                f"skips={skipped_throttle}",
                flush=True,
            )
            skipped_throttle = 0
