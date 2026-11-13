# Hotel Occupancy & Room-Demand Forecasting

Decision-support forecasting system predicting daily occupied room-nights by room category (`Standard`, `Deluxe`, `Suite`, `Executive`) for a 7-day operational horizon.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 1. Generate Reproducible Dataset
```bash
python data/generate_data.py
```

## 2. Train & Evaluate Models
Trains a 7-day Seasonal Naive baseline, Ridge Regression, and LightGBM (point + 10th/90th quantile models) using chronological splits (`2023-2024` train, `H1 2025` validation, `H2 2025` held-out test).
```bash
python -m src.train
```

## 3. Run Inference API
```bash
uvicorn src.app:app --reload --port 8000
```

Sample request:
```bash
curl -X POST "http://127.0.0" \
  -H "Content-Type: application/json" \
  -d '{
    "stay_date": "2025-12-25",
    "room_category": "Deluxe",
    "room_capacity": 45,
    "otb_7d_prior": 31,
    "otb_14d_prior": 25,
    "avg_booking_lead_days": 21.5,
    "is_holiday": 1,
    "occ_lag_7d": 33.0,
    "occ_lag_14d": 30.0,
    "occ_roll_mean_7d": 31.4
  }'
```

## 4. Run Tests
```bash
pytest
```
