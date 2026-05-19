"""Load test: clients fire JPEGs at /detect.

Run a local stress test:
    locust -f locustfile.py --host http://localhost:8000 --users 10 --spawn-rate 2

Open http://localhost:8089 for the web UI, or add `--headless -t 60s` for CLI.
"""
import io
import random

import numpy as np
from locust import HttpUser, between, task
from PIL import Image


def make_jpeg(size: int = 640) -> bytes:
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=75)
    return buf.getvalue()


_IMAGES = [make_jpeg(640) for _ in range(5)]


class DetectUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def detect(self) -> None:
        img = random.choice(_IMAGES)
        self.client.post(
            "/detect",
            files={"file": ("bench.jpg", img, "image/jpeg")},
            name="POST /detect",
        )

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
