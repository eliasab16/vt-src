import time
from fastapi import FastAPI, WebSocket
import torch
from remote_inference.image_codec import decode_image_to_tensor
import traceback
import json

from remote_inference.protocol import (
    HealthResponse,
    SetupRequest,
    SetupResponse,
    InferenceRequest,
    InferenceRTCRequest,
    InferenceResponse,
    SetupStatus,
)

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors


class InferenceServer:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.device = None
        self.loaded_policy_path = None  # tracks which policy is currently loaded

    @property
    def chunk_size(self) -> int | None:
        if self.policy is None:
            return None    
        return self.policy.config.chunk_size

    @property
    def ready(self) -> bool:
        return all([
            self.policy is not None,
            self.preprocessor is not None,
            self.postprocessor is not None,
            self.device is not None,
        ])

    @property
    def uptime_s(self) -> float:
        return time.perf_counter() - self.start_time
    
    def setup(self, request: SetupRequest) -> SetupResponse:
        # Skip reload if the same policy is already loaded on the same device
        if (
            self.ready
            and self.loaded_policy_path == request.policy_path
            and self.device == request.device
        ):
            return SetupResponse(
                status=SetupStatus.READY,
                message=f"Policy already loaded: {request.policy_path} on {request.device}",
                chunk_size=self.chunk_size,
            )

        try:
            # 1. Load and setup policy
            policy = PI05Policy.from_pretrained(request.policy_path)
            # tells the policy whether or not to use "torch.compile" (important for mps compatibility)
            policy.config.compile_model = request.compile_model
            policy.to(request.device)
            policy.eval()

            # 2. Load processors (+ device override with requested device)
            device_override = {"device": request.device}
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config,
                pretrained_path=request.policy_path,
                preprocessor_overrides={
                    "device_processor": device_override,
                    # TODO: if we need camera renamings in the future for other models, do it here
                },
                postprocessor_overrides={"device_processor": device_override},
            )

            # 3. Warmup CUDA kernels — runs a non-RTC and an RTC dummy inference so the
            # first real request is fast instead of eating a 20s+ kernel compilation hit.
            try:
                self._warmup(policy, preprocessor, request.camera_names, request.action_dim, request.device)
            except Exception as warmup_err:
                traceback.print_exc()
                return SetupResponse(
                    status=SetupStatus.ERROR,
                    message=f"Warmup failed: {type(warmup_err).__name__}: {warmup_err}",
                )

            # 4. now we set all the server states (after all the above steps succeeded without exceptions)
            self.policy = policy
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor
            self.device = request.device
            self.loaded_policy_path = request.policy_path

            return SetupResponse(status=SetupStatus.READY, message=f"Loaded {request.policy_path} on {request.device}", chunk_size=self.chunk_size)
        except Exception as e:
            traceback.print_exc()
            return SetupResponse(status=SetupStatus.ERROR, message=f"{type(e).__name__}: {e}")

    def _warmup(self, policy, preprocessor, camera_names, action_dim, device):
        """Run dummy inferences to pre-compile CUDA kernels (non-RTC + RTC paths).

        Both kernel paths are compiled here so the first real /infer request
        doesn't trigger a ~20s kernel compilation hit (which otherwise drains
        the client's action queue during warmup).
        """
        # Use the robot's actual dim (not policy.config.max_*_dim) — normalizer stats
        # are sized to the real dim, so passing the padded max breaks broadcasting.
        state_dim = action_dim
        chunk_size = policy.config.chunk_size

        # Build dummy observation. Image shape mirrors what the client sends:
        # aspect-preserving, non-square — so resize_with_pad_torch runs (same path as real requests).
        obs = {
            "observation.state": torch.zeros(1, state_dim, dtype=torch.float32).to(device),
            "task": "warmup",
            "robot_type": "",
        }
        for cam_name in camera_names:
            obs[f"observation.images.{cam_name}"] = torch.zeros(
                1, 3, 168, 224, dtype=torch.float32
            ).to(device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            processed_obs = preprocessor(obs)

            # Non-RTC kernel path
            _ = policy.predict_action_chunk(processed_obs)

            # RTC kernel path — fake prev_chunk_left_over so the RTC path compiles too
            fake_leftover = torch.zeros(
                1, chunk_size - 5, action_dim, dtype=torch.float32
            ).to(device)
            _ = policy.predict_action_chunk(
                processed_obs,
                prev_chunk_left_over=fake_leftover,
                inference_delay=5,
                execution_horizon=20,
            )
        # Reset any internal policy state that accumulated during warmup
        # (e.g., action queues, attention caches the policy might maintain).
        policy.reset()

        elapsed_s = time.perf_counter() - t0
        print(f"Warmup complete ({elapsed_s:.1f}s) — non-RTC + RTC kernels compiled.")

    def inference(self, request: InferenceRequest | InferenceRTCRequest) -> InferenceResponse:
        # 1. Build observation dict
        obs = {}
        obs["observation.state"] = torch.tensor(request.state, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs["task"] = request.task
        obs["robot_type"] = "" # for now we hardcode this since our initial model only supports one robot type; in the future we can add it as a field in the request if needed
        for cam_name, img_b64 in request.images.items():
            obs[f"observation.images.{cam_name}"] = decode_image_to_tensor(img_b64).to(self.device)


        # 2. Optional RTC kwargs - only when request is RTC-typed with a leftover
        rtc_kwargs = {}
        if isinstance(request, InferenceRTCRequest) and request.prev_chunk_left_over is not None:
            rtc_kwargs["prev_chunk_left_over"] = (
                torch.tensor(request.prev_chunk_left_over, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            rtc_kwargs["inference_delay"] = request.inference_delay
            rtc_kwargs["execution_horizon"] = request.execution_horizon

        with torch.inference_mode():
            # 3. Preprocess
            obs = self.preprocessor(obs)
            # 4. Inference
            t0 = time.perf_counter()
            action_chunk = self.policy.predict_action_chunk(obs, **rtc_kwargs)
            t1 = time.perf_counter()
            inference_time_ms = (t1 - t0) * 1000

            # 5. save original actions (for RTC leftover tracking)
            original_actions = action_chunk.squeeze(0).detach().cpu()

            # 6. Postprocess each step individually
            _, chunk_size, _ = action_chunk.shape
            processed_actions = []
            for i in range(chunk_size):
                single_action = action_chunk[:, i, :]        # shape (1, action_dim)
                processed = self.postprocessor(single_action)  # shape (1, action_dim)
                processed_actions.append(processed)
            processed_tensor = torch.stack(processed_actions, dim=1)  # shape (1, chunk_size, action_dim)
            processed_tensor = processed_tensor.squeeze(0).detach().cpu()  # shape (chunk_size, action_dim)

        return InferenceResponse(
            actions=processed_tensor.tolist(),
            original_actions=original_actions.tolist(),
            inference_time_ms=inference_time_ms
        )


app = FastAPI(title="Remote Inference Server")
server = InferenceServer()

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint to report server status."""
    return HealthResponse(
        ready=server.ready,
        device=server.device or "none",
        uptime_s=server.uptime_s,
    )

@app.post("/setup", response_model=SetupResponse)
async def setup(req: SetupRequest) -> SetupResponse:
    return server.setup(req)

@app.websocket("/ws")
async def websocket_inference(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                if "prev_chunk_left_over" in payload:
                    # if the request has RTC fields, validate as InferenceRTCRequest
                    request = InferenceRTCRequest.model_validate(payload)
                else:
                    request = InferenceRequest.model_validate(payload)
                response = server.inference(request)

                await ws.send_text(response.model_dump_json())
            except Exception as e:
                traceback.print_exc()
                await ws.send_text(f'{{"error": "{type(e).__name__}: {e}"}}')
    except Exception:
        traceback.print_exc()
        pass  # client disconnected
