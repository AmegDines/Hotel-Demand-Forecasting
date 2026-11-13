import json
import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features import FEATURE_COLS, build_features

def wape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)))

def evaluate_predictions(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(root_mean_squared_error(y_true, y_pred)), 3),
        "WAPE": round(wape(y_true, y_pred) * 100, 2),
    }

def run_training():
    raw = pd.read_csv("data/hotel_occupancy.csv")
    df = build_features(raw).dropna().reset_index(drop=True)

    train_df = df[df["stay_date"] < "2025-01-01"]
    val_df = df[(df["stay_date"] >= "2025-01-01") & (df["stay_date"] < "2025-07-01")]
    test_df = df[df["stay_date"] >= "2025-07-01"]

    X_train, y_train = train_df[FEATURE_COLS], train_df["occupied_rooms"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["occupied_rooms"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["occupied_rooms"]

    cat_cols = ["room_category"]
    num_cols = [c for c in FEATURE_COLS if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    ridge_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", Ridge(alpha=1.0, random_state=42)),
    ])
    ridge_pipe.fit(X_train, y_train)

    lgbm_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )),
    ])
    lgbm_pipe.fit(X_train, y_train)

    lgbm_lower = Pipeline([
        ("prep", preprocessor),
        ("model", LGBMRegressor(
            objective="quantile",
            alpha=0.1,
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )),
    ])
    lgbm_lower.fit(X_train, y_train)

    lgbm_upper = Pipeline([
        ("prep", preprocessor),
        ("model", LGBMRegressor(
            objective="quantile",
            alpha=0.9,
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )),
    ])
    lgbm_upper.fit(X_train, y_train)

    val_results = {
        "Naive_Lag7": evaluate_predictions(y_val, val_df["occ_lag_7d"]),
        "Ridge": evaluate_predictions(y_val, np.clip(ridge_pipe.predict(X_val), 0, val_df["room_capacity"])),
        "LightGBM": evaluate_predictions(y_val, np.clip(lgbm_pipe.predict(X_val), 0, val_df["room_capacity"])),
    }

    test_preds = np.clip(lgbm_pipe.predict(X_test), 0, test_df["room_capacity"])
    test_results = {
        "Naive_Lag7": evaluate_predictions(y_test, test_df["occ_lag_7d"]),
        "Ridge": evaluate_predictions(y_test, np.clip(ridge_pipe.predict(X_test), 0, test_df["room_capacity"])),
        "LightGBM": evaluate_predictions(y_test, test_preds),
    }

    category_breakdown = {}
    test_eval = test_df.copy()
    test_eval["pred"] = test_preds
    for cat, grp in test_eval.groupby("room_category"):
        category_breakdown[cat] = evaluate_predictions(grp["occupied_rooms"], grp["pred"])

    os.makedirs("models", exist_ok=True)
    artifact = {
        "point_model": lgbm_pipe,
        "lower_model": lgbm_lower,
        "upper_model": lgbm_upper,
        "feature_cols": FEATURE_COLS,
    }
    joblib.dump(artifact, "models/demand_forecaster.joblib")

    report = {
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "validation_metrics": val_results,
        "held_out_test_metrics": test_results,
        "test_metrics_by_category": category_breakdown,
    }
    with open("models/metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run_training()
