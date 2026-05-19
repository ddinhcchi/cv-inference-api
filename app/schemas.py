from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    class_id: int = Field(..., description="COCO class index")
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BBox


class ImageMeta(BaseModel):
    width: int
    height: int
    resized_width: int
    resized_height: int


class DetectionResponse(BaseModel):
    request_id: str
    model: str
    device: str
    image: ImageMeta
    detections: list[Detection]
    latency_ms: float


class ClassesResponse(BaseModel):
    model: str
    classes: dict[int, str]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
