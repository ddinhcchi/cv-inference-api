# ⚡ CV Inference API

Production-ready FastAPI service wrapping YOLOv8 for object detection. Send an image, get JSON detections — works for any system that can `POST` a file (web apps, n8n flows, mobile, Postman, `curl`).

![swagger](demo/swagger.jpg)

> Swagger UI screenshot placeholder. After cloning, run the server and capture the `/docs` page.

---

## Why this project

Most ML repos hand you a notebook and stop. A real client wants something they can `curl`. This service is what you'd actually ship to their staging environment: typed Pydantic schemas, a `/health` for their load balancer, optional `X-API-Key` auth, CORS, configurable confidence and class filter, multipart **or** URL input, and a multi-stage Docker image with the model baked in.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + active model + device |
| `GET` | `/classes` | All COCO class ids → names |
| `POST` | `/detect` | Multipart image upload → detections |
| `POST` | `/detect/batch` | 2–N images in one request → per-image detections |
| `POST` | `/detect/url?url=…` | Detect from a public image URL |
| `GET` | `/metrics` | Prometheus text format — counters, latency histogram, model info |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc reference |

All `POST` endpoints accept `?conf=0.5` and `?classes=person,car` query params.

---

## Quick start

```bash
git clone <this-repo>
cd cv-inference-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional — defaults work
uvicorn app.main:app --reload
```

Then:

```bash
curl -fsS http://127.0.0.1:8000/health
# {"status":"ok","model":"yolov8n.pt","device":"cpu"}

# Detect everything in an image
curl -s -X POST http://127.0.0.1:8000/detect \
     -F "file=@samples/street.jpg" \
     | jq '.detections[] | {class_name, confidence}'

# Only return people and cars, raise the threshold to 0.6
curl -s -X POST "http://127.0.0.1:8000/detect?conf=0.6&classes=person,car" \
     -F "file=@samples/street.jpg" \
     | jq '.detections | length'

# Equivalent using class ids instead of names (person=0, car=2)
curl -s -X POST "http://127.0.0.1:8000/detect?classes=0,2" \
     -F "file=@samples/street.jpg"

# Detect from a public URL (no upload)
curl -s -X POST "http://127.0.0.1:8000/detect/url?url=https://ultralytics.com/images/bus.jpg" \
     | jq '{count: (.detections | length), latency_ms}'

# Batch: 5 images in one round-trip → per-image detection arrays
curl -s -X POST http://127.0.0.1:8000/detect/batch \
     -F "files=@a.jpg" -F "files=@b.jpg" -F "files=@c.jpg" \
     -F "files=@d.jpg" -F "files=@e.jpg" \
     | jq '{batch_size, inference_latency_ms, total_latency_ms,
            per_image: [.results[] | {filename, count: (.detections | length)}]}'

# With API key (when API_KEY is set in .env)
curl -X POST http://127.0.0.1:8000/detect \
     -H "X-API-Key: $API_KEY" \
     -F "file=@samples/street.jpg"
```

Or open http://127.0.0.1:8000/docs and use the Swagger UI directly.

### Example response

```json
{
  "request_id": "f8a4...",
  "model": "yolov8n.pt",
  "device": "mps",
  "image": {"width": 1920, "height": 1080, "resized_width": 1920, "resized_height": 1080},
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "bbox": {"x1": 612, "y1": 248, "x2": 745, "y2": 612}
    }
  ],
  "latency_ms": 23.47
}
```

---

## Benchmark — Apple M4 (10-core, 16 GB)

Real load test with `locust`, 4 concurrent users, 30 s, 640×640 JPEG payloads. Reproduce with `locust -f locustfile.py --host http://127.0.0.1:8000 --users 4 --headless -t 30s`.

| Device | p50 | p95 | p99 | throughput | failures |
|---|---:|---:|---:|---:|---:|
| **MPS (Apple GPU)** | **24 ms** | 42 ms | 63 ms | ~11 req/s | 0 |
| CPU | 41 ms | 86 ms | 140 ms | ~10 req/s | 0 |

Throughput is gated by the test's `wait_time` (0.1–0.5 s per user) — not by the API. Headroom on MPS is significantly higher; bump `--users` to find your saturation point on your hardware.

### Batch vs sequential (5 × 640 JPEGs, M4 MPS, post-warmup)

| Mode | Wall-clock | Inference only |
|---|---:|---:|
| 5 × sequential `POST /detect` | ~56 ms total (11.2 ms/img) | — |
| 1 × `POST /detect/batch` of 5 | **~45 ms total (9 ms/img)** | 33 ms |

Batching is ~1.24× faster on M4 — most of the win comes from amortising HTTP, decode and Pydantic serialisation across one request, not from raw inference parallelism. On true GPU servers (CUDA + larger batch) the speedup widens substantially. The *first* batched call pays an MPS kernel-compile cost (~1 s); subsequent calls run at the steady-state numbers above.

---

## Auth (optional)

Set `API_KEY=somelongsecret` in `.env`. Clients must then send `X-API-Key: somelongsecret`. Leaving `API_KEY` empty disables auth entirely.

```bash
curl -X POST http://127.0.0.1:8000/detect \
     -H "X-API-Key: somelongsecret" \
     -F "file=@photo.jpg"
```

---

## Docker

Multi-stage build, image ≈ 1.4 GB. Model weights baked in so first request is instant.

```bash
docker build -t cv-inference-api .
docker run --rm -p 8000:8000 cv-inference-api
```

Or with compose + `.env`:

```bash
docker compose up --build
```

The container runs CPU inference (Docker Desktop on macOS cannot pass through Apple GPU). On a CUDA host you can add the `--gpus all` flag and set `DEVICE=cuda`.

---

## Code layout

| File | Responsibility |
|---|---|
| [`app/main.py`](app/main.py) | FastAPI app, routes, CORS, API-key dependency |
| [`app/inference.py`](app/inference.py) | `ModelService` — loads YOLO once, handles resize + predict |
| [`app/schemas.py`](app/schemas.py) | Pydantic response models |
| [`app/config.py`](app/config.py) | `pydantic-settings`-driven env vars |
| [`tests/test_api.py`](tests/test_api.py) | 7 pytest cases: happy path, bad input, class filter |
| [`locustfile.py`](locustfile.py) | Load test scenario |
| [`Dockerfile`](Dockerfile) | Multi-stage CPU image |

---

## Tests

```bash
pytest tests/ -v
# 16 passed in ~7s (model load dominates first run)
```

---

## Observability — Prometheus `/metrics`

The service exposes four metric families on `/metrics` in standard Prometheus text format. Point your scraper at the endpoint and you get request volume, latency, per-class detection counts and a model-info label out of the box.

| Metric | Type | Labels |
|---|---|---|
| `cv_inference_requests_total` | counter | `endpoint`, `method`, `status` (status class `2xx`/`4xx`/`5xx`) |
| `cv_inference_latency_seconds` | histogram | `endpoint` — buckets: 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2s, 5s |
| `cv_inference_detections_total` | counter | `class_name` — increments by 1 per bbox returned |
| `cv_inference_model_info` | gauge | `model`, `device` — always 1, labels carry the info |

Sample scrape excerpt after a few real requests:

```
cv_inference_requests_total{endpoint="/detect",method="POST",status="2xx"} 12
cv_inference_latency_seconds_bucket{endpoint="/detect",le="0.05"} 9
cv_inference_latency_seconds_bucket{endpoint="/detect",le="0.1"} 12
cv_inference_detections_total{class_name="person"} 17
cv_inference_detections_total{class_name="car"} 4
cv_inference_model_info{model="yolov8n.pt",device="mps"} 1.0
```

Drop this Prometheus scrape job in and you've got SLO dashboards in 5 minutes:

```yaml
- job_name: cv-inference-api
  metrics_path: /metrics
  static_configs:
    - targets: ['cv-api.internal:8000']
```

Status-class labels keep cardinality bounded (3 instead of every distinct HTTP code) — Prometheus best-practice.

---

## Design notes

- **Large images are downscaled** before inference (configurable via `MAX_IMAGE_SIDE`). Bounding boxes are scaled back to original-image coordinates in the response so clients never have to.
- **Singleton model** loaded inside FastAPI's `lifespan` context — no per-request reload, no cold starts after warm-up.
- **Failure modes are explicit**: 415 for non-image upload, 422 for empty / bad params, 502 for failed URL fetch, 401 for bad API key.

---

## Security — secret scanning

The repo wires up [`gitleaks`](https://github.com/gitleaks/gitleaks) via [`pre-commit`](https://pre-commit.com) so it's impossible to accidentally commit an `API_KEY` value or any other credential. The hook runs locally on every `git commit` and blocks the commit if anything matches.

```bash
brew install gitleaks pre-commit   # macOS — Linux users: pipx install both
pre-commit install                  # installs the .git/hooks/pre-commit shim
pre-commit run --all-files          # scan everything already in the index
gitleaks detect --source . --verbose # scan the full git history
```

[`.gitleaks.toml`](.gitleaks.toml) extends the default ruleset with two allowlists:

- `.env.example` and `README.md` — they intentionally contain placeholder credentials
- The literal `X-API-Key: $API_KEY` curl example and `API_KEY=somelongsecret` doc snippet — these are documentation patterns, not real secrets

For deployments, also enable [GitHub Push Protection](https://docs.github.com/en/code-security/secret-scanning/push-protection-for-repositories-and-organizations) as a second line of defence — it scans on the server side, so even a developer who skipped `pre-commit install` can't leak credentials.

---

## License

MIT
