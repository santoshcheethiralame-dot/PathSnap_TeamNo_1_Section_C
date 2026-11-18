import sys
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "Student folder"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.trips import list_trips
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


app = FastAPI(
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def get_trips():
    try:
        trip_dirs = list_trips(DATA_ROOT, "student_test")
        trip_ids = [Path(path).name for path in trip_dirs]

        return {
            "count": len(trip_ids),
            "trips": trip_ids
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

    trip_dir = DATA_ROOT / "student_test" / trip_id

    if not trip_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    try:
        # Run prediction once
        result = predict_trip(trip_dir)

        # Score the existing prediction
        scores = score_trip(trip_dir, prediction=result)

        # Add numerical metrics
        result.update(scores)

        return result

    except Exception:
        logger.exception("Prediction or scoring failed for trip: %s", trip_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction or scoring failed. Check the backend terminal."
        )
