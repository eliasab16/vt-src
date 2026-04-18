"""Round-trip tests for protocol models."""

# to run the tests: pytest remote_inference/protocol_test.py

import base64
import time

from remote_inference.protocol import (
    InferenceRequest,
    InferenceRTCRequest,
    InferenceResponse,
    SetupRequest,
    SetupResponse,
    HealthResponse,
)


def test_inference_request_round_trip():
    """InferenceRequest should survive JSON serialize -> deserialize unchanged."""
    fake_jpeg = base64.b64encode(b"fake_jpeg_bytes").decode("ascii")

    req = InferenceRequest(
        state=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        images={
            "wrist_top": fake_jpeg,
            "wrist_bottom": fake_jpeg,
            "overhead": fake_jpeg,
        },
        task="pick up the object",
        timestamp=time.perf_counter(),
    )

    json_str = req.model_dump_json()
    req2 = InferenceRequest.model_validate_json(json_str)

    assert req == req2, f"Round-trip in {test_inference_request_round_trip.__name__} failed: {req} != {req2}"


def test_infer_rtc_request_round_trip():
    """InferenceRTCRequest should survive JSON serialize -> deserialize unchanged."""
    fake_jpeg = base64.b64encode(b"fake_jpeg_bytes").decode("ascii")

    req = InferenceRTCRequest(
        state=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        images={
            "wrist_top": fake_jpeg,
            "wrist_bottom": fake_jpeg,
            "overhead": fake_jpeg,
        },
        task="pick up the object",
        timestamp=time.perf_counter(),
        prev_chunk_left_over=[[0.1] * 8, [0.2] * 8],
        inference_delay=3,
        execution_horizon=20,
    )

    json_str = req.model_dump_json()
    req2 = InferenceRTCRequest.model_validate_json(json_str)

    assert req == req2, f"Round-trip in {test_infer_rtc_request_round_trip.__name__} failed: {req} != {req2}"


def test_infer_rtc_request_optional_fields():
    """InferenceRTCRequest optional fields should default to None."""
    fake_jpeg = base64.b64encode(b"fake_jpeg_bytes").decode("ascii")

    req = InferenceRTCRequest(
        state=[0.0] * 8,
        images={"wrist_top": fake_jpeg},
        task="test",
        timestamp=0.0,
    )

    assert req.prev_chunk_left_over is None
    assert req.inference_delay is None

    json_str = req.model_dump_json()
    req2 = InferenceRTCRequest.model_validate_json(json_str)
    assert req == req2


def test_inference_response_round_trip():
    """InferenceResponse should survive JSON serialize -> deserialize unchanged."""
    req = InferenceResponse(
        actions=[[0.1] * 8] * 10,
        original_actions=[[0.2] * 8] * 10,
        inference_time_ms=42.5,
    )

    json_str = req.model_dump_json()
    req2 = InferenceResponse.model_validate_json(json_str)

    assert req == req2, f"Round-trip in {test_inference_response_round_trip.__name__} failed: {req} != {req2}"


def test_setup_request_round_trip():
    """SetupRequest should survive JSON serialize -> deserialize unchanged."""
    req = SetupRequest(
        policy_path="lerobot/pi0_base",
        action_dim=8,
        camera_names=["wrist_top", "wrist_bottom", "overhead"],
        task="pick up the object",
        device="cuda",
        compile_model=False,
    )

    json_str = req.model_dump_json()
    req2 = SetupRequest.model_validate_json(json_str)

    assert req == req2, f"Round-trip in {test_setup_request_round_trip.__name__} failed: {req} != {req2}"


def test_setup_response_round_trip():
    """SetupResponse should survive JSON serialize -> deserialize unchanged."""
    req = SetupResponse(
        status="ready",
        message="model loaded",
        chunk_size=10,
    )

    json_str = req.model_dump_json()
    req2 = SetupResponse.model_validate_json(json_str)

    assert req == req2, f"Round-trip in {test_setup_response_round_trip.__name__} failed: {req} != {req2}"


def test_setup_response_null_chunk_size():
    """SetupResponse chunk_size should default to None."""
    req = SetupResponse(status="error", message="failed to load")

    assert req.chunk_size is None

    json_str = req.model_dump_json()
    req2 = SetupResponse.model_validate_json(json_str)
    assert req == req2


def test_health_response_round_trip():
    """HealthResponse should survive JSON serialize -> deserialize unchanged."""
    req = HealthResponse(
        ready=True,
        device="cuda",
        uptime_s=123.4,
    )

    json_str = req.model_dump_json()
    req2 = HealthResponse.model_validate_json(json_str)

    assert req == req2, f"Round-trip in {test_health_response_round_trip.__name__} failed: {req} != {req2}"
