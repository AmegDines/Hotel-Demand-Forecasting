import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.features import FEATURE_COLS, build_features

app = FastAPI(title="Hotel Room-Demand Forecasting API", version="1.0.0")

try:
    ARTIFACT = joblib.load("models/demand_forecaster.joblib")
except FileNotFoundError:
    ARTIFACT = None

class ForecastRequest(BaseModel):
    stay_date: str = Field(..., example="2025-12-25")
    room_category: str = Field(..., example="Deluxe")
    room_capacity: int = Field(..., gt=0, example=45)
    otb_7d_prior: int = Field(..., ge=0, example=31)
    otb_14d_prior: int = Field(..., ge=0, example=25)
    avg_booking_lead_days: float = Field(..., ge=0, example=21.5)
    is_holiday: int = Field(..., ge=0, le=1, example=1)
    occ_lag_7d: float = Field(..., ge=0, example=33.0)
    occ_lag_14d: float = Field(..., ge=0, example=30.0)
    occ_roll_mean_7d: float = Field(..., ge=0, example=31.4)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": ARTIFACT is not None}

@app.post("/predict")
def predict_demand(req: ForecastRequest):
    if ARTIFACT is None:
        raise HTTPException(status_code=503, detail="Model artifact not found. Run training first.")

    row = pd.DataFrame([req.model_dump()])
    featured = build_features(row)
    X = featured[FEATURE_COLS]

    cap = req.room_capacity
    pred = float(np.clip(ARTIFACT["point_model"].predict(X)[0], 0, cap))
    low = float(np.clip(ARTIFACT["lower_model"].predict(X)[0], 0, pred))
    high = float(np.clip(ARTIFACT["upper_model"].predict(X)[0], pred, cap))

    spread = high - low
    if req.is_holiday == 1 or spread > cap * 0.25:
        uncertainty_note = (
            "Higher forecast uncertainty due to holiday volatility or wide historical pickup variance. "
            "Recommend revenue manager review before adjusting dynamic rates."
        )
    else:
        uncertainty_note = (
            "Normal booking variance within the 80% quantile interval. "
            "Safe for baseline staffing and inventory planning with human oversight."
        )

    return {
        "stay_date": req.stay_date,
        "room_category": req.room_category,
        "predicted_room_nights": round(pred, 1),
        "predicted_occupancy_pct": round((pred / cap) * 100, 1),
        "prediction_interval_80pct": {
            "lower_bound": round(low, 1),
            "upper_bound": round(high, 1),
        },
        "uncertainty_explanation": uncertainty_note,
    }
