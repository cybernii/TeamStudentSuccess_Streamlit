"""
Integration tests for the Student Risk Prediction API.

Run with:
    .venv/bin/python -m pytest tests/test_integration.py -q
"""

import asyncio
import json
import time

import pytest

from api.app import app

VALID_PAYLOAD = {
    "avg_score": 45.0,
    "completion_rate": 0.35,
    "total_clicks": 150,
    "studied_credits": 60,
    "num_of_prev_attempts": 1,
    "module_BBB": False,
    "module_CCC": False,
    "module_DDD": False,
    "module_EEE": False,
    "module_FFF": False,
    "module_GGG": False,
    "gender": "M",
    "region": "South East Region",
    "highest_education": "A Level or Equivalent",
    "imd_band": "50-60%",
    "age_band": "0-35",
    "disability": "N",
}


def request_api(method: str, path: str, payload=None):
    body = b""
    headers = []
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers.append((b"content-type", b"application/json"))
        headers.append((b"content-length", str(len(body)).encode("utf-8")))

    async def run_request():
        messages = []
        sent_body = False

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("utf-8"),
                "query_string": b"",
                "headers": headers,
                "client": ("testclient", 123),
                "server": ("testserver", 80),
            },
            receive,
            send,
        )
        return messages

    messages = asyncio.run(run_request())
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    data = json.loads(response_body.decode("utf-8")) if response_body else None
    return status, data


def test_api_health():
    status, data = request_api("GET", "/health")
    assert status == 200
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_valid_prediction():
    status, data = request_api("POST", "/predict", VALID_PAYLOAD)
    assert status == 200
    assert data["prediction"] in [0, 1]
    assert data["risk_level"] in ["low", "medium", "high"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_at_risk"] <= 1.0


def test_minimum_values():
    min_payload = {
        **VALID_PAYLOAD,
        "avg_score": 0.0,
        "completion_rate": 0.0,
        "total_clicks": 0,
        "studied_credits": 0,
        "num_of_prev_attempts": 0,
    }
    status, _ = request_api("POST", "/predict", min_payload)
    assert status == 200


def test_maximum_values():
    max_payload = {
        **VALID_PAYLOAD,
        "avg_score": 100.0,
        "completion_rate": 1.0,
        "total_clicks": 30000,
        "studied_credits": 700,
        "num_of_prev_attempts": 6,
    }
    status, _ = request_api("POST", "/predict", max_payload)
    assert status == 200


def test_missing_field():
    incomplete_payload = {key: value for key, value in VALID_PAYLOAD.items() if key != "avg_score"}
    status, _ = request_api("POST", "/predict", incomplete_payload)
    assert status == 422


def test_empty_request():
    status, _ = request_api("POST", "/predict", {})
    assert status == 422


@pytest.mark.parametrize("bad_value", ["not_a_number", None])
def test_wrong_data_type(bad_value):
    bad_payload = {**VALID_PAYLOAD, "avg_score": bad_value}
    status, _ = request_api("POST", "/predict", bad_payload)
    assert status == 422


def test_response_time():
    start = time.perf_counter()
    status, _ = request_api("POST", "/predict", VALID_PAYLOAD)
    elapsed = time.perf_counter() - start
    assert status == 200
    assert elapsed < 5.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
