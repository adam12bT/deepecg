"""Exercise a running SleepOps API with a bundled held-out EEG window."""

import argparse
import json
from urllib.request import Request, urlopen

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    with urlopen(base + "/health/ready", timeout=10) as response:
        ready = json.load(response)
    with np.load("artifacts/demo_windows.npz", allow_pickle=False) as examples:
        payload = {"features": examples["windows"][len(examples["windows"]) // 2].tolist()}
    request = Request(base + "/v1/predict", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        prediction = json.load(response)
    assert ready["status"] == "ready" and ready["release_id"] == prediction["release_id"]
    assert len(prediction["asleep_probabilities"]) == 10
    if prediction["state"] == "asleep":
        assert len(prediction["forecast"]["bins"]) == 13
    else:
        assert prediction["forecast"] is None
    with urlopen(base + "/metrics", timeout=10) as response:
        assert b'sleepops_prediction_requests_total{outcome="success"}' in response.read()
    print(f"HTTP smoke passed for {ready['release_id']} ({prediction['state']})")


if __name__ == "__main__":
    main()
