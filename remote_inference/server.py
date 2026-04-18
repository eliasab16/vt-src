import time
from fastapi import FastAPI

from remote_inference.protocol import (
    HealthResponse,
    SetupRequest,
    SetupResponse,
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

            # 3. now we set all the server states (after all the above steps succeeded without exceptions)
            self.policy = policy
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor
            self.device = request.device

            return SetupResponse(status="ready", message=f"Loaded {request.policy_path} on {request.device}", chunk_size=self.chunk_size)
        except Exception as e:
            return SetupResponse(status="error", message=f"{type(e).__name__}: {e}")


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