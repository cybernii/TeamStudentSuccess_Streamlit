"""
Assignment 03 - Integration Test Suite
AIE1014 | Onyekachi Odunze

Run with: python tests/test_integration.py
(API must be running: python -m uvicorn api.main:app --reload)
"""

import requests
import time
import sys
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_URL = "http://localhost:8000"

# Track results across all tests
passed = 0
failed = 0

# A realistic "at-risk" student payload matching the API schema exactly
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


def run_test(test_fn):
    """Execute a single test function and record pass/fail."""
    global passed, failed
    try:
        test_fn()
        passed += 1
    except AssertionError as e:
        print(f"  ❌ FAILED - {e}")
        failed += 1
    except Exception as e:
        print(f"  ❌ ERROR  - {e}")
        failed += 1


# Test 1: Health check
def test_api_health():
    print("Test 1: API health check")
    response = requests.get(f"{API_URL}/health", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data.get("status") == "healthy", f"Expected status=healthy, got {data}"
    assert data.get("model_loaded") is True, "Model not loaded - check api/model.pkl"
    print("  ✅ API is healthy and model is loaded")


# Test 2: Valid prediction - happy path
def test_valid_prediction():
    print("Test 2: Valid prediction (happy path)")
    response = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD, timeout=10)
    assert response.status_code == 200, \
        f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "prediction" in data, f"Response missing 'prediction' key. Got: {data}"
    assert "risk_level" in data, f"Response missing 'risk_level' key. Got: {data}"
    assert "confidence" in data, f"Response missing 'confidence' key. Got: {data}"
    assert data["prediction"] in [0, 1], f"Prediction must be 0 or 1, got {data['prediction']}"
    assert data["risk_level"] in ["low", "medium", "high"], \
        f"risk_level must be low/medium/high, got {data['risk_level']}"
    print(f"  ✅ Prediction: {data['prediction']} | Risk: {data['risk_level']} | Confidence: {data['confidence']:.1%}")


# Test 3: Minimum boundary values
def test_minimum_values():
    print("Test 3: Minimum boundary values")
    min_payload = {**VALID_PAYLOAD,
                   "avg_score": 0.0,
                   "completion_rate": 0.0,
                   "total_clicks": 0,
                   "studied_credits": 0,
                   "num_of_prev_attempts": 0}
    response = requests.post(f"{API_URL}/predict", json=min_payload, timeout=10)
    assert response.status_code == 200, \
        f"Minimum values rejected with status {response.status_code}: {response.text}"
    print("  ✅ Minimum values accepted")


# Test 4: Maximum boundary values
def test_maximum_values():
    print("Test 4: Maximum boundary values")
    max_payload = {**VALID_PAYLOAD,
                   "avg_score": 100.0,
                   "completion_rate": 1.0,
                   "total_clicks": 30000,
                   "studied_credits": 700,
                   "num_of_prev_attempts": 6}
    response = requests.post(f"{API_URL}/predict", json=max_payload, timeout=10)
    assert response.status_code == 200, \
        f"Maximum values rejected with status {response.status_code}: {response.text}"
    print("  ✅ Maximum values accepted")


# Test 5: Missing required field
def test_missing_field():
    print("Test 5: Missing required field (avg_score omitted)")
    incomplete_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "avg_score"}
    response = requests.post(f"{API_URL}/predict", json=incomplete_payload, timeout=10)
    # FastAPI/Pydantic returns 422 for validation errors when a required field is absent
    assert response.status_code in [400, 422], \
        f"Expected 400 or 422 for missing field, got {response.status_code}"
    print(f"  ✅ Missing field rejected with status {response.status_code}")


# Test 6: Empty request body
def test_empty_request():
    print("Test 6: Empty request body")
    response = requests.post(f"{API_URL}/predict", json={}, timeout=10)
    assert response.status_code in [400, 422], \
        f"Expected 400 or 422 for empty body, got {response.status_code}"
    print(f"  ✅ Empty request rejected with status {response.status_code}")


# Test 7: Wrong data type for numeric field
def test_wrong_data_type():
    print("Test 7: Wrong data type (string where float expected)")
    bad_payload = {**VALID_PAYLOAD, "avg_score": "not_a_number"}
    response = requests.post(f"{API_URL}/predict", json=bad_payload, timeout=10)
    assert response.status_code in [400, 422, 500], \
        f"Expected error status for wrong type, got {response.status_code}"
    print(f"  ✅ Wrong type rejected with status {response.status_code}")


# Test 8: Response time under 5 seconds
def test_response_time():
    print("Test 8: Response time under 5 seconds")
    start = time.time()
    response = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD, timeout=10)
    elapsed = time.time() - start
    assert elapsed < 5.0, f"Response took {elapsed:.2f}s - exceeds 5s limit"
    print(f"  ✅ Response time: {elapsed:.3f}s")


def run_all_tests():
    print("\n" + "=" * 55)
    print("  INTEGRATION TEST SUITE - AIE1014 Assignment 03")
    print("  Student: Onyekachi Odunze")
    print("=" * 55 + "\n")

    tests = [
        test_api_health,
        test_valid_prediction,
        test_minimum_values,
        test_maximum_values,
        test_missing_field,
        test_empty_request,
        test_wrong_data_type,
        test_response_time,
    ]

    for test in tests:
        run_test(test)
        print()

    print("=" * 55)
    print(f"  RESULTS: {passed} passed  |  {failed} failed")
    print("=" * 55 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
