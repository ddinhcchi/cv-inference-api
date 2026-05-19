import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .schemas import BBox, Detection, ImageMeta


def _resolve_device(preferred: str) -> str:
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ModelService:
    def __init__(self, weights: str, device: str, max_image_side: int = 1920):
        self.device = _resolve_device(device)
        self.weights = weights
        self.model = YOLO(weights)
        self.class_names: dict[int, str] = self.model.names
        self.max_image_side = max_image_side
        self._warmup()

    def _warmup(self) -> None:
        dummy = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
        for _ in range(2):
            self.model.predict(dummy, device=self.device, verbose=False)

    def _resize_if_needed(self, bgr: np.ndarray) -> tuple[np.ndarray, ImageMeta]:
        h, w = bgr.shape[:2]
        long_side = max(h, w)
        if long_side <= self.max_image_side:
            return bgr, ImageMeta(width=w, height=h, resized_width=w, resized_height=h)
        scale = self.max_image_side / long_side
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, ImageMeta(width=w, height=h, resized_width=new_w, resized_height=new_h)

    def detect(
        self,
        image_bgr: np.ndarray,
        conf: float,
        classes: list[int] | None = None,
    ) -> tuple[list[Detection], ImageMeta, float]:
        resized, meta = self._resize_if_needed(image_bgr)
        t0 = time.perf_counter()
        results = self.model.predict(
            resized,
            device=self.device,
            conf=conf,
            classes=classes,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        out: list[Detection] = []
        if not results:
            return out, meta, latency_ms
        boxes = results[0].boxes
        if boxes is None or boxes.cls is None:
            return out, meta, latency_ms

        scale_x = meta.width / meta.resized_width
        scale_y = meta.height / meta.resized_height
        for cls_t, conf_t, xyxy_t in zip(boxes.cls, boxes.conf, boxes.xyxy):
            cid = int(cls_t.item())
            x1, y1, x2, y2 = xyxy_t.tolist()
            out.append(
                Detection(
                    class_id=cid,
                    class_name=self.class_names.get(cid, str(cid)),
                    confidence=float(conf_t.item()),
                    bbox=BBox(
                        x1=int(x1 * scale_x),
                        y1=int(y1 * scale_y),
                        x2=int(x2 * scale_x),
                        y2=int(y2 * scale_y),
                    ),
                )
            )
        return out, meta, latency_ms
