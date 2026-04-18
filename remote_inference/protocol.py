from pydantic import BaseModel

# client tells server which policy to load
class SetupRequest(BaseModel):
    policy_path: str # hf repo path, e.g. "lerobot/pi0_base"
    action_dim: int = 8 # default for so110 arm
    camera_names: list[str] # ["wrist_top", "wrist_bottom", "overhead"]
    task: str # task name in natural language
    device: str = "cuda"
    compile_model: bool = False

# server acknowledges with model metadata
class SetupResponse(BaseModel):
    status: str # "error", "ready"
    message: str # human readable
    chunk_size: int | None = None # client must specify this

# ask for action chunk given an observation
class InferenceRequest(BaseModel):
    state: list[float] # for joint positions
    images: dict[str, str] # camera name ==> base64-encoded jpeg image
    task: str
    timestamp: float

# same as InferenceRequest but with additional fields for real-time control (RTC)
class InferenceRTCRequest(InferenceRequest):
    prev_chunk_left_over: list[list[float]] | None = None # (unconsumed actions from previous chunk; shape is [timesteps, action_dim])
    inference_delay: int | None = None # (how many steps will be consumed during this inference; client computes from latency tracker)
    execution_horizon: int = 20 # how many steps to predict/return in this chunk

# Here we're using two action lists because RTC needs the pre-postprocessor actions (the raw model output) for its guidance math.
# See the ActionQueue.merge() signature in rtc/action_queue.py which takes both original_actions and processed_actions.
class InferenceResponse(BaseModel):
    actions: list[list[float]] # shape [chunk_size, action_dim], post-postprocessor — ready for robot
    original_actions: list[list[float]] # same shape, pre-postprocessor — needed for RTC leftover tracking
    inference_time_ms: float

class HealthResponse(BaseModel):
    ready: bool
    device: str
    uptime_s: float
