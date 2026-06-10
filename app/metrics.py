"""Prometheus instrumentation.

Counters cover request volume + per-class detection counts; the histogram
captures end-to-end latency per endpoint with buckets that match the
realistic operating range for YOLOv8 small-batch inference.
"""
import time

from fastapi import Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response


REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "cv_inference_requests_total",
    "Total HTTP requests handled, labelled by endpoint, method and status class.",
    labelnames=("endpoint", "method", "status"),
    registry=REGISTRY,
)

LATENCY = Histogram(
    "cv_inference_latency_seconds",
    "End-to-end wall-clock latency per endpoint, including decode + model + serialize.",
    labelnames=("endpoint",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    registry=REGISTRY,
)

DETECTIONS = Counter(
    "cv_inference_detections_total",
    "Number of bounding boxes returned, labelled by class name.",
    labelnames=("class_name",),
    registry=REGISTRY,
)

MODEL_INFO = Gauge(
    "cv_inference_model_info",
    "Active model + device. Value is always 1; the labels carry the info.",
    labelnames=("model", "device"),
    registry=REGISTRY,
)


def set_model_info(model: str, device: str) -> None:
    MODEL_INFO.labels(model=model, device=device).set(1)


def record_detections(class_names: list[str]) -> None:
    for cn in class_names:
        DETECTIONS.labels(class_name=cn).inc()


def _status_class(status_code: int) -> str:
    # 200 → "2xx", 404 → "4xx" — keeps cardinality low for Prometheus.
    return f"{status_code // 100}xx"


async def prometheus_middleware(request: Request, call_next):
    """Time the request and increment counters. Routes that error before
    reaching this middleware (e.g. invalid host) won't be counted — fine,
    they're not interesting for SLO tracking."""
    started = time.perf_counter()
    endpoint = request.url.path
    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUESTS.labels(
            endpoint=endpoint,
            method=request.method,
            status=_status_class(response.status_code),
        ).inc()
        return response
    except Exception:
        elapsed = time.perf_counter() - started
        LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUESTS.labels(endpoint=endpoint, method=request.method, status="5xx").inc()
        raise


def metrics_response() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
