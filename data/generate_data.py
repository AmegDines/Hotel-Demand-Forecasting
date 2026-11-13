import os
import numpy as np
import pandas as pd

def create_dataset(seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
    room_configs = {
        "Standard": {"capacity": 80, "base": 52, "weekend_boost": 14},
        "Deluxe": {"capacity": 45, "base": 28, "weekend_boost": 9},
        "Suite": {"capacity": 20, "base": 11, "weekend_boost": 5},
        "Executive": {"capacity": 15, "base": 9, "weekend_boost": 2},
    }

    holidays = {
        (1, 1), (1, 26), (5, 1), (8, 15), (10, 2), (12, 24), (12, 25), (12, 31)
    }

    rows = []
    for dt in dates:
        dow = dt.dayofweek
        is_weekend = 1 if dow in (4, 5) else 0
        is_holiday = 1 if (dt.month, dt.day) in holidays else 0
        season_factor = 1.0 + 0.18 * np.sin(2 * np.pi * (dt.dayofyear - 30) / 365.0)
        trend_factor = 1.0 + 0.04 * ((dt.year - 2023) + dt.dayofyear / 365.0)

        for cat, cfg in room_configs.items():
            expected = (
                cfg["base"] * season_factor * trend_factor
                + is_weekend * cfg["weekend_boost"]
                + is_holiday * (cfg["capacity"] * 0.15)
            )
            noise = np.random.normal(0, cfg["capacity"] * 0.06)
            actual_occupied = int(np.clip(round(expected + noise), 0, cfg["capacity"]))

            otb_7d = int(np.clip(round(actual_occupied * np.random.uniform(0.68, 0.88)), 0, cfg["capacity"]))
            otb_14d = int(np.clip(round(otb_7d * np.random.uniform(0.75, 0.92)), 0, cfg["capacity"]))
            avg_lead_days = round(np.random.uniform(12.0, 28.0) + is_holiday * 6.5, 1)

            rows.append({
                "stay_date": dt.strftime("%Y-%m-%d"),
                "room_category": cat,
                "room_capacity": cfg["capacity"],
                "occupied_rooms": actual_occupied,
                "otb_7d_prior": otb_7d,
                "otb_14d_prior": otb_14d,
                "avg_booking_lead_days": avg_lead_days,
                "is_holiday": is_holiday,
            })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/hotel_occupancy.csv", index=False)
    print(f"Saved {len(df)} records to data/hotel_occupancy.csv")

if __name__ == "__main__":
    create_dataset()
