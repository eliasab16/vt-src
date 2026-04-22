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

import csv
import json
import threading
import time
from collections import deque
import torch
from torch import Tensor

_LOG_SEPARATOR = "─" * 80

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
            queue_threshold: int = 15,
            log_actions_csv: str | None = None,
            joint_names: list[str] | None = None,
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
        self._queue_threshold = queue_threshold
        self._warmup_latency_cutoff_s = 1.0  # ignore inferences slower than this in latency tracker

        # Chunk-tagging bookkeeping: parallel to ActionQueue, tracks which actions
        # belong to which chunk so we can log chunk boundaries / transitions.
        self._chunk_counter = 0
        self._pending_chunks: deque = deque()
        self._chunks_lock = threading.Lock()
        self._t_start: float | None = None  # set in start()

        # Per-action CSV logging (one row per popped action).
        self._log_actions_csv_path = log_actions_csv
        self._joint_names = joint_names
        self._csv_file = None
        self._csv_writer = None
        self._csv_lock = threading.Lock()

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
        self._t_start = time.perf_counter()

        # Open per-action CSV if configured. Header is written lazily on first
        # row so we can size motor columns to the actual action_dim.
        if self._log_actions_csv_path is not None:
            self._csv_file = open(self._log_actions_csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            print(f"[client] logging per-action positions to {self._log_actions_csv_path}", flush=True)

        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def stop(self):
        self._running.clear()
        if self.ws:
            self.ws.close()
        if self._inference_thread:
            self._inference_thread.join(timeout=5)
        self.ws = None
        with self._csv_lock:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

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
        action = self._action_queue.get()
        if action is None:
            return None

        # Track consumption of the head chunk; log transition when it's exhausted.
        chunk_id_for_csv = None
        action_idx_for_csv = None
        chunk_fire_ms_for_csv = None
        with self._chunks_lock:
            if self._pending_chunks:
                head = self._pending_chunks[0]
                chunk_id_for_csv = head["chunk_id"]
                action_idx_for_csv = head["chunk_len"] - head["remaining"]
                chunk_fire_ms_for_csv = head["fire_time_ms"]
                head["remaining"] -= 1

                if head["remaining"] == 0:
                    next_chunk = self._pending_chunks[1] if len(self._pending_chunks) > 1 else None
                    if next_chunk is not None and self._t_start is not None:
                        t_rel = (time.perf_counter() - self._t_start) * 1000
                        first_new_action = next_chunk["actions"][0]
                        print(_LOG_SEPARATOR, flush=True)
                        print(
                            f"[transition @ t={t_rel:.0f}ms]  chunk_{head['chunk_id']} → chunk_{next_chunk['chunk_id']}",
                            flush=True,
                        )
                        print(
                            f"  last action (chunk_{head['chunk_id']}):  {[f'{v:+.3f}' for v in action.tolist()]}",
                            flush=True,
                        )
                        print(
                            f"  first action (chunk_{next_chunk['chunk_id']}): {[f'{v:+.3f}' for v in first_new_action.tolist()]}",
                            flush=True,
                        )
                    self._pending_chunks.popleft()

        # Per-action CSV row (one row per popped action, for post-hoc analysis).
        if self._csv_writer is not None and self._t_start is not None:
            with self._csv_lock:
                if self._csv_writer is None:  # double-check inside lock (stop() may have closed it)
                    return action
                t_ms = (time.perf_counter() - self._t_start) * 1000
                action_list = action.tolist()
                # Lazy header: write once we know the action_dim.
                if self._csv_file.tell() == 0:
                    motor_cols = (
                        self._joint_names
                        if self._joint_names is not None and len(self._joint_names) == len(action_list)
                        else [f"m{i}" for i in range(len(action_list))]
                    )
                    self._csv_writer.writerow(
                        ["t_ms", "chunk_id", "action_idx", "chunk_fire_ms", "age_ms", *motor_cols]
                    )
                age_ms = t_ms - chunk_fire_ms_for_csv if chunk_fire_ms_for_csv is not None else ""
                self._csv_writer.writerow([
                    f"{t_ms:.2f}",
                    chunk_id_for_csv if chunk_id_for_csv is not None else "",
                    action_idx_for_csv if action_idx_for_csv is not None else "",
                    f"{chunk_fire_ms_for_csv:.2f}" if chunk_fire_ms_for_csv is not None else "",
                    f"{age_ms:.2f}" if age_ms != "" else "",
                    *[f"{v:.4f}" for v in action_list],
                ])

        return action


    def clear_queue(self):
        if self._action_queue is not None:
            self._action_queue.clear()
        with self._chunks_lock:
            self._pending_chunks.clear()

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
        while self._running.is_set():
            # 1. Grab latest observation (atomic swap — pop it so we don't re-send)
            with self._obs_lock:
                obs = self._latest_obs
                self._latest_obs = None

            if obs is None:
                time.sleep(0.005)
                continue

            # Throttle: skip inference until the queue drops to the configured threshold.
            if self._action_queue.qsize() > self._queue_threshold:
                time.sleep(0.005)
                continue

            # Snapshot what the main thread is currently executing, so we can
            # later report which action-within-chunk was being executed at fire time.
            with self._chunks_lock:
                if self._pending_chunks:
                    head = self._pending_chunks[0]
                    exec_chunk_id = head["chunk_id"]
                    exec_action_idx = head["chunk_len"] - head["remaining"]
                    exec_chunk_len = head["chunk_len"]
                else:
                    exec_chunk_id, exec_action_idx, exec_chunk_len = None, None, None

            t_fire = time.perf_counter()
            t_fire_rel_ms = (t_fire - self._t_start) * 1000 if self._t_start is not None else 0.0

            # 2. Build + send inference request
            inference_request = self._build_request(obs)
            action_index_before_inference = self._action_queue.get_action_index() if self._action_queue else None
            try:
                self.ws.send(inference_request.model_dump_json())
                try:
                    response_text = self.ws.recv()
                except ConnectionClosed:
                    break
                payload = json.loads(response_text)
                if "error" in payload:
                    print(f"[inference thread] server error: {payload['error']}", flush=True)
                    continue
                response = InferenceResponse.model_validate(payload)
            except Exception as e:
                print(f"[inference thread] request/response error: {type(e).__name__}: {e}", flush=True)
                continue
            rtt_ms = (time.perf_counter() - t_fire) * 1000
            if rtt_ms / 1000 < self._warmup_latency_cutoff_s:
                self._latency_tracker.add(rtt_ms / 1000)

            # 3. Merge into ActionQueue
            real_delay = self._action_queue.get_action_index() - action_index_before_inference if action_index_before_inference is not None else 0
            processed_actions = torch.tensor(response.actions)
            original_actions = torch.tensor(response.original_actions)
            self._action_queue.merge(
                original_actions=original_actions,
                processed_actions=processed_actions,
                real_delay=real_delay,
                action_index_before_inference=action_index_before_inference,
            )

            # 4. Register this chunk for main-thread chunk-transition tracking
            with self._chunks_lock:
                self._chunk_counter += 1
                new_chunk_id = self._chunk_counter
                chunk_len_int = processed_actions.shape[0]
                self._pending_chunks.append({
                    "chunk_id": new_chunk_id,
                    "remaining": chunk_len_int,
                    "chunk_len": chunk_len_int,
                    "fire_time_ms": t_fire_rel_ms,
                    "actions": processed_actions.clone(),
                })

            # 5. Log fire event
            exec_desc = (
                f"executing chunk_{exec_chunk_id} action {exec_action_idx}/{exec_chunk_len}"
                if exec_chunk_id is not None
                else "queue empty"
            )
            print(_LOG_SEPARATOR, flush=True)
            print(
                f"[fire chunk_{new_chunk_id} @ t={t_fire_rel_ms:.0f}ms]  rtt={rtt_ms:.0f}ms  {exec_desc}",
                flush=True,
            )
