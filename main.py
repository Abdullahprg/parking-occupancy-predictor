"""
FastAPI layer for the Parking Occupancy Predictor.

Usage:
    python3 parking_predictor.py setup-dev      # or fetch-real + load-real
    uvicorn main:app --reload
    http://127.0.0.1:8000/docs
"""

from typing import Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from parking_predictor import (
    DAYS,
    BaselineModel,
    predict_for,
    row_count,
)

app = FastAPI(
    title="Parking Occupancy Predictor API",
    description=(
        "Predicts parking lot occupancy for a given day of week and hour, "
        "using a baseline grouped-average model trained on historical occupancy data."
    ),
    version="1.0.0",
    contact={"name": "Johnny"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: BaselineModel | None = None


@app.on_event("startup")
def load_model():
    global _model
    _model = BaselineModel() if row_count() > 0 else None


# ---------------------------------------------------------------------------
# Response schemas (these are what make the /docs page look clean and
# self-documenting instead of showing raw untyped JSON)
# ---------------------------------------------------------------------------
class PredictionMeta(BaseModel):
    sample_count: int = Field(..., description="How many historical rows backed this prediction")
    used_fallback_average: bool = Field(..., description="True if no exact match was found and the overall average was used instead")
    best_time_predicted_occupancy_percent: float = Field(..., description="Predicted occupancy at the best time of day")


class PredictionResponse(BaseModel):
    occupancy_percent: float = Field(..., example=63.1, description="Predicted occupancy as a percentage (0-100)")
    wait_minutes: int = Field(..., example=3, description="Estimated wait time in minutes to find a spot")
    best_time: str = Field(..., example="8:00 PM", description="The least-busy hour on this day, formatted 12-hour")
    _meta: PredictionMeta


class BestTimeResponse(BaseModel):
    day: str = Field(..., example="Saturday")
    best_hour: int = Field(..., example=9, description="Best hour in 24h format (0-23)")
    predicted_occupancy_percent: float = Field(..., example=43.1)
    hourly_breakdown: Dict[int, float] = Field(..., description="Predicted occupancy % for every hour of the day")


class DaysResponse(BaseModel):
    days: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"], summary="API status")
def root():
    """Basic health check and endpoint index."""
    return {
        "message": "Parking Occupancy Predictor API",
        "status": "ok" if _model is not None else "no data loaded",
        "docs": "/docs",
    }


@app.get("/days", response_model=DaysResponse, tags=["Info"], summary="Valid day names")
def get_days():
    """Returns the list of valid day-of-week strings accepted by other endpoints."""
    return {"days": DAYS}


@app.get(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict occupancy for a day and hour",
)
def predict(
    day: str = Query(..., examples=["Monday"], description="Day of week (case-insensitive)"),
    hour: int = Query(..., ge=0, le=23, examples=[14], description="Hour of day, 0-23"),
):
    """
    Predicts parking occupancy for the given day and hour using the baseline
    grouped-average model, plus an estimated wait time and the best time to go.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="No data loaded yet. Run setup-dev or fetch-real + load-real, then restart the API.")

    day = day.strip().capitalize()
    if day not in DAYS:
        raise HTTPException(status_code=400, detail=f"day must be one of {DAYS}")

    return predict_for(day, hour, _model)


@app.get(
    "/best-time",
    response_model=BestTimeResponse,
    tags=["Predictions"],
    summary="Find the least-busy hour on a given day",
)
def best_time(day: str = Query(..., examples=["Saturday"], description="Day of week (case-insensitive)")):
    """Returns the predicted best (least busy) hour for the given day, plus the full 24-hour breakdown."""
    if _model is None:
        raise HTTPException(status_code=503, detail="No data loaded yet. Run setup-dev or fetch-real + load-real, then restart the API.")

    day = day.strip().capitalize()
    if day not in DAYS:
        raise HTTPException(status_code=400, detail=f"day must be one of {DAYS}")

    hourly = _model.predict_all_hours(day)
    best_hour = min(hourly, key=hourly.get)
    return {
        "day": day,
        "best_hour": best_hour,
        "predicted_occupancy_percent": hourly[best_hour],
        "hourly_breakdown": hourly,
    }