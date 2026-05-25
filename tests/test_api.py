import io
from unittest.mock import patch

import httpx
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


# ---------------------------------------------------------------------------
# /detect/url — patch httpx.AsyncClient to avoid real network calls
# ---------------------------------------------------------------------------


class _MockAsyncClient:
    """Mimics `async with httpx.AsyncClient(timeout=…) as c: await c.get(url)`."""

    def __init__(self, handler, **_kwargs):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, url, **_kwargs):
        req = httpx.Request("GET", url)
        resp = self._handler(req)
        # binding the request is required for raise_for_status() to work
        if isinstance(resp, httpx.Response):
            resp.request = req
        return resp


def _patch_httpx(handler):
    return patch(
        "app.main.httpx.AsyncClient",
        lambda **kwargs: _MockAsyncClient(handler, **kwargs),
    )


def test_detect_url_happy_path(client):
    img = _fake_jpeg(width=400, height=300)

    def handler(_req):
        return httpx.Response(200, content=img, headers={"content-type": "image/jpeg"})

    with _patch_httpx(handler):
        r = client.post("/detect/url", params={"url": "https://example.com/img.jpg"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"]["width"] == 400


def test_detect_url_returns_502_on_404(client):
    def handler(_req):
        return httpx.Response(404)

    with _patch_httpx(handler):
        r = client.post(
            "/detect/url", params={"url": "https://example.com/missing.jpg"}
        )
    assert r.status_code == 502
    assert "fetch failed" in r.json()["detail"].lower()


def test_detect_url_returns_502_on_timeout(client):
    def handler(req):
        raise httpx.ConnectTimeout("simulated", request=req)

    with _patch_httpx(handler):
        r = client.post(
            "/detect/url", params={"url": "https://slow.example.com/img.jpg"}
        )
    assert r.status_code == 502


def test_detect_url_returns_415_when_body_is_not_an_image(client):
    """3xx redirects aren't followed by httpx default, so the body the client
    gets back is whatever the server sent — often HTML. Anything that isn't
    a decodable image must surface as 415, not crash."""

    def handler(_req):
        return httpx.Response(
            200,
            content=b"<html>not an image</html>",
            headers={"content-type": "text/html"},
        )

    with _patch_httpx(handler):
        r = client.post(
            "/detect/url", params={"url": "https://example.com/page.html"}
        )
    assert r.status_code == 415
