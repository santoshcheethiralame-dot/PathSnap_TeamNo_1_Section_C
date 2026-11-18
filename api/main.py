import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Folder whose sub-folders are the trips to serve, one folder per trip with the
# four files inside. Set PATHSNAP_TRIPS_DIR to point at the evaluator folder on
# the day; the default is the student-test folder next to the repository.
TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR") or PROJECT_ROOT.parent / "Student folder" / "student_test")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.predict import predict_trip
from api.scoring import score_trip


LOG_DIR = PROJECT_ROOT / "api" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("pathsnap_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "requests.log",
        encoding="utf-8"
    )

    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.propagate = False


@asynccontextmanager
async def lifespan(_app):
    logger.info("Serving %d trips from %s", len(trip_ids()), TRIPS_DIR)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="PathSnap API",
    description="GPS Route Reconstruction and Map Matching",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration = (time.perf_counter() - start_time) * 1000

        logger.info(
            "%s %s - Status: %s - %.2f ms",
            request.method,
            request.url.path,
            response.status_code,
            duration
        )

        return response

    except Exception:
        logger.exception(
            "Request failed: %s %s",
            request.method,
            request.url.path
        )
        raise


class PredictionRequest(BaseModel):
    trip_id: str


def trip_ids():
    """Trip folders that hold a track file, sorted by id."""
    if not TRIPS_DIR.is_dir():
        return []
    return sorted(d.name for d in TRIPS_DIR.iterdir() if d.is_dir() and (d / f"{d.name}.track").is_file())


@app.get("/health")
def health():
    return {"status": "ok", "trips_available": len(trip_ids())}


@app.get("/trips")
def get_trips():
    try:
        ids = trip_ids()
        return {
            "count": len(ids),
            "trips": ids
        }

    except Exception:
        logger.exception("Unable to load student-test trips.")
        raise HTTPException(
            status_code=500,
            detail="Unable to load student-test trips."
        )


@app.post("/predict")
def predict(request: PredictionRequest):
    trip_id = request.trip_id

    if not trip_id or Path(trip_id).name != trip_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid trip ID."
        )

    trip_dir = TRIPS_DIR / trip_id

    if not (trip_dir / f"{trip_id}.track").is_file():
        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    try:
        started = time.perf_counter()
        # Run prediction once
        result = predict_trip(trip_dir)

        # Score the existing prediction
        scores = score_trip(trip_dir, prediction=result)

        # Add numerical metrics
        result.update(scores)

        logger.info(
            "predict trip=%s method=%s points=%d route_arcs=%d top1=%s continuity=%s scored=%s time=%.2fs",
            trip_id, result["method"], result["n_points"], len(result["route_arcs"]),
            _fmt(scores["top1_accuracy"]), _fmt(scores["route_continuity"]), scores["scored"],
            time.perf_counter() - started,
        )
        return result

    except Exception:
        logger.exception("Prediction or scoring failed for trip: %s", trip_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction or scoring failed. Check the backend terminal."
        )


def _fmt(value):
    return "n/a" if value is None else f"{value:.4f}"
