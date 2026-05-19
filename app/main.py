import io
import uuid
from contextlib import asynccontextmanager

import cv2
import httpx
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from .config import settings
from .inference import ModelService
from .schemas import ClassesResponse, DetectionResponse, ErrorResponse, HealthResponse

_state: dict[str, ModelService] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    _state["model"] = ModelService(
        weights=settings.model_weights,
        device=settings.device,
        max_image_side=settings.max_image_side,
    )
    yield
    _state.clear()


app = FastAPI(
    title="CV Inference API",
    description=(
        "REST wrapper around YOLOv8 for object detection. Send an image, get "
        "JSON detections with bounding boxes, class names and confidences."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_model() -> ModelService:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return model


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing api key")


def _parse_classes(classes_q: str | None, model: ModelService) -> list[int] | None:
    if not classes_q:
        return None
    out: list[int] = []
    for token in classes_q.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            out.append(int(token))
            continue
        for cid, cname in model.class_names.items():
            if cname == token:
                out.append(cid)
                break
        else:
            raise HTTPException(status_code=422, detail=f"unknown class: {token}")
    return out or None


def _decode_image(raw: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=415, detail=f"cannot decode image: {exc}") from exc
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


@app.get("/health", response_model=HealthResponse)
def health(model: ModelService = Depends(get_model)) -> HealthResponse:
    return HealthResponse(status="ok", model=model.weights, device=model.device)


@app.get("/classes", response_model=ClassesResponse)
def classes(model: ModelService = Depends(get_model)) -> ClassesResponse:
    return ClassesResponse(model=model.weights, classes=model.class_names)


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        401: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    dependencies=[Depends(require_api_key)],
)
async def detect(
    file: UploadFile = File(..., description="JPEG/PNG/WebP image"),
    conf: float = Query(default=settings.default_conf, ge=0.0, le=1.0),
    classes_filter: str | None = Query(
        default=None,
        alias="classes",
        description="Comma-separated COCO class ids or names (e.g. `person,car`).",
    ),
    request: Request = None,  # type: ignore[assignment]
    model: ModelService = Depends(get_model),
) -> DetectionResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="empty upload")
    image = _decode_image(raw)
    cls_filter = _parse_classes(classes_filter, model)
    detections, meta, latency_ms = model.detect(image, conf=conf, classes=cls_filter)
    return DetectionResponse(
        request_id=str(uuid.uuid4()),
        model=model.weights,
        device=model.device,
        image=meta,
        detections=detections,
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/detect/url",
    response_model=DetectionResponse,
    dependencies=[Depends(require_api_key)],
)
async def detect_url(
    url: str = Query(..., description="Public image URL"),
    conf: float = Query(default=settings.default_conf, ge=0.0, le=1.0),
    classes_filter: str | None = Query(default=None, alias="classes"),
    model: ModelService = Depends(get_model),
) -> DetectionResponse:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"fetch failed: {exc}") from exc
    image = _decode_image(resp.content)
    cls_filter = _parse_classes(classes_filter, model)
    detections, meta, latency_ms = model.detect(image, conf=conf, classes=cls_filter)
    return DetectionResponse(
        request_id=str(uuid.uuid4()),
        model=model.weights,
        device=model.device,
        image=meta,
        detections=detections,
        latency_ms=round(latency_ms, 2),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.__class__.__name__, detail=str(exc.detail)).model_dump(),
    )
