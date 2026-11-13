# Technical Report: Hotel Room-Demand Forecasting

## 1. Problem Framing
- **Business Decision:** Assist hotel revenue managers and housekeeping leads in scheduling staff and adjusting room tier pricing 7 days ahead of stay dates.
- **Prediction Unit & Target:** Daily `occupied_rooms` (integer count bounded by `room_capacity`) per `room_category`.
- **Prediction Point & Horizon:** Forecast generated at $T - 7$ days for stay date $T$.

## 2. Dataset & Leakage Controls
- **Data Source:** Synthetic generator (`data/generate_data.py`, seed=42) producing 4,380 daily records across 4 room categories from 2023–2025. No real guest PII is collected or stored.
- **Leakage Prevention:** All historical target features (`occ_lag_7d`, `occ_lag_14d`, `occ_roll_mean_7d`) are shifted by at least 7 days per room category. Advance-booking signals (`otb_7d_prior`, `otb_14d_prior`) strictly reflect reservations already on the books prior to the forecast window.

## 3. Validation Strategy & Model Comparison
Chronological split without shuffling:
- **Train:** Jan 1, 2023 – Dec 31, 2024
- **Validation:** Jan 1, 2025 – Jun 30, 2025
- **Held-Out Test:** Jul 1, 2025 – Dec 31, 2025

| Model | Role | Primary Strengths |
| :--- | :--- | :--- |
| **Naive Lag-7** | Seasonal Baseline | Captures weekly day-of-week repetition without ML complexity. |
| **Ridge Regression** | Linear Candidate | Fast, interpretable baseline capturing linear pickup and calendar trends. |
| **LightGBM** | Primary Candidate | Captures non-linear interactions between holidays, weekends, and booking pickup; supports quantile intervals. |

*(Run `python -m src.train` to populate exact held-out test MAE, RMSE, and WAPE figures from `models/metrics.json` into this table before final submission.)*

## 4. Error Analysis & Limitations
- **Error Pattern:** Largest absolute errors occur in `Standard` rooms due to higher capacity (80 rooms), whereas `Suite` rooms show higher percentage variance (WAPE) because small booking changes shift percentages significantly.
- **Key Limitations:** Synthetic demand does not model sudden external shocks (airline cancellations, local conventions, or competitor flash sales).
- **Responsible Use & Human Oversight:** Outputs include an 80% prediction interval (`lower_bound` to `upper_bound`). Forecasts are strictly advisory and must be reviewed by operations staff before locking room inventory or staff shifts.
