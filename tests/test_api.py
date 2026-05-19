import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _fake_jpeg(width: int = 320, height: int = 240) -> bytes:
    arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "yolov8" in body["model"]
    assert body["device"] in {"cpu", "mps", "cuda"}


def test_classes_includes_person(client):
    r = client.get("/classes")
    assert r.status_code == 200
    body = r.json()
    assert "0" in body["classes"] or 0 in body["classes"]


def test_detect_happy_path_returns_schema(client):
    img = _fake_jpeg()
    r = client.post("/detect", files={"file": ("test.jpg", img, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "detections" in body
    assert "latency_ms" in body
    assert body["image"]["width"] == 320
    assert isinstance(body["detections"], list)


def test_detect_rejects_non_image(client):
    r = client.post(
        "/detect", files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert r.status_code == 415


def test_detect_rejects_empty_upload(client):
    r = client.post("/detect", files={"file": ("test.jpg", b"", "image/jpeg")})
    assert r.status_code == 422


def test_detect_unknown_class_filter(client):
    img = _fake_jpeg()
    r = client.post(
        "/detect",
        files={"file": ("test.jpg", img, "image/jpeg")},
        params={"classes": "not_a_class"},
    )
    assert r.status_code == 422


def test_detect_class_filter_by_name(client):
    img = _fake_jpeg()
    r = client.post(
        "/detect",
        files={"file": ("test.jpg", img, "image/jpeg")},
        params={"classes": "person,car"},
    )
    assert r.status_code == 200
