import pandas as pd
from fastapi.testclient import TestClient
from src.app import app
from src.features import build_features

def test_no_cross_category_leakage():
    sample = pd.DataFrame([
        {
            "stay_date": f"2025-01-{d:02d}",
            "room_category": "Standard",
            "room_capacity": 80,
            "occupied_rooms": 50 + d,
            "otb_7d_prior": 40,
            "otb_14d_prior": 35,
            "avg_booking_lead_days": 15.0,
            "is_holiday": 0,
        }
        for d in range(1, 10)
    ] + [
        {
            "stay_date": "2025-01-01",
            "room_category": "Suite",
            "room_capacity": 20,
            "occupied_rooms": 10,
            "otb_7d_prior": 7,
            "otb_14d_prior": 5,
            "avg_booking_lead_days": 20.0,
            "is_holiday": 0,
        }
    ])

    out = build_features(sample)
    suite_row = out[out["room_category"] == "Suite"].iloc[0]
    assert pd.isna(suite_row["occ_lag_7d"])

def test_api_prediction_endpoint():
    client = TestClient(app)
    payload = {
        "stay_date": "2025-12-25",
        "room_category": "Deluxe",
        "room_capacity": 45,
        "otb_7d_prior": 31,
        "otb_14d_prior": 25,
        "avg_booking_lead_days": 21.5,
        "is_holiday": 1,
        "occ_lag_7d": 33.0,
        "occ_lag_14d": 30.0,
        "occ_roll_mean_7d": 31.4,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["predicted_room_nights"] <= 45
    assert body["prediction_interval_80pct"]["lower_bound"] <= body["predicted_room_nights"]
