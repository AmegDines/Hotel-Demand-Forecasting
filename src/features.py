import numpy as np
import pandas as pd

FEATURE_COLS = [
    "room_category",
    "room_capacity",
    "day_of_week",
    "is_weekend",
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_holiday",
    "otb_7d_prior",
    "otb_14d_prior",
    "otb_pickup_7d",
    "avg_booking_lead_days",
    "occ_lag_7d",
    "occ_lag_14d",
    "occ_roll_mean_7d",
]

def build_features(df):
    data = df.copy()
    data["stay_date"] = pd.to_datetime(data["stay_date"])
    data = data.sort_values(["room_category", "stay_date"]).reset_index(drop=True)

    data["day_of_week"] = data["stay_date"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([4, 5]).astype(int)
    data["month"] = data["stay_date"].dt.month
    doy = data["stay_date"].dt.dayofyear
    data["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.0)
    data["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.0)
    data["otb_pickup_7d"] = data["otb_7d_prior"] - data["otb_14d_prior"]

    if "occupied_rooms" in data.columns:
        grp = data.groupby("room_category")["occupied_rooms"]
        data["occ_lag_7d"] = grp.shift(7)
        data["occ_lag_14d"] = grp.shift(14)
        data["occ_roll_mean_7d"] = grp.shift(7).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)

    return data
