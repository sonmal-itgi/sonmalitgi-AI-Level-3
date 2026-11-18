import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ai_server import metrics
from ai_server.main import app, evaluate
from ai_server.model import Prediction


def test_receive_fps_uses_server_intervals(monkeypatch):
    clock = iter([10, 11, 11.125, 11.25, 11.5])
    monkeypatch.setattr(metrics.time, "monotonic", lambda: next(clock))
    attempt = metrics.AttemptMetrics()
    for _ in range(3):
        attempt.receive_frame()
    snapshot = attempt.snapshot()
    assert snapshot["frame_count"] == 3
    assert snapshot["received_fps"] == snapshot["average_received_fps"] == 8
    assert snapshot["elapsed_seconds"] == 1.5
    assert snapshot["pass_seconds"] is None


@pytest.mark.parametrize("prediction_id, expected", [("SEN0012", "PASS"), ("SEN0013", "FAIL")])
def test_evaluation_metrics_keep_pass_policy(monkeypatch, prediction_id, expected):
    class Model:
        mode = "test"
        supported_ids = ["SEN0012", "SEN0013"]

        def predict_landmarks(self, frames):
            return Prediction(prediction_id, .99, True)

    class Socket:
        headers = {"origin": "http://localhost:5173"}
        url = SimpleNamespace(path="/ws/v1/word/evaluate")
        client_state = SimpleNamespace(name="CONNECTED")

        def __init__(self):
            self.sent = []
            self.messages = [
                dict(type="start", input_mode="landmarks", target_id="SEN0012",
                     attempt_id=str(uuid4())),
                *[dict(type="landmarks", schema="mediapipe-hands-v1", timestamp_ms=i * 125,
                       left_hand=[], right_hand=[]) for i in range(18)],
                dict(type="stop"),
            ]

        async def accept(self):
            pass

        async def close(self):
            pass

        async def send_json(self, message):
            self.sent.append(message)

        async def receive(self):
            return dict(type="websocket.receive", text=json.dumps(self.messages.pop(0)))

    async def run():
        model = Model()
        for name in ("model", "word_model", "unit5_model"):
            monkeypatch.setattr(app.state, name, model, raising=False)
        monkeypatch.setattr(app.state, "unit5_words", None, raising=False)
        monkeypatch.setattr(app.state, "model_lock", asyncio.Lock(), raising=False)
        socket = Socket()
        await evaluate(socket)
        final = socket.sent[-1]
        assert final["type"] == "final"
        assert final["result"] == expected
        measurement = final["metrics"]
        assert measurement["frame_count"] == 18
        assert measurement["window_seconds"] == 2.125
        assert measurement["window_frames"] == 18
        assert measurement["raw_predicted_id"] == prediction_id
        assert (measurement["pass_seconds"] is not None) == (expected == "PASS")
        assert measurement["inference_ms"] >= 0
        assert measurement["queue_ms"] >= 0

    asyncio.run(run())
