import os
import pandas as pd
import matplotlib.pyplot as plt

def run_eda():
    df = pd.read_csv("data/hotel_occupancy.csv")
    df["stay_date"] = pd.to_datetime(df["stay_date"])
    df["occupancy_rate"] = (df["occupied_rooms"] / df["room_capacity"]) * 100
    df["day_name"] = df["stay_date"].dt.day_name()

    print("=== DATASET SHAPE & DATE RANGE ===")
    print("Rows, Columns:", df.shape)
    print("Start Date:", df["stay_date"].min().date(), "| End Date:", df["stay_date"].max().date())

    print("\n=== DATA QUALITY CHECKS ===")
    print("Missing values per column:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    over_capacity = (df["occupied_rooms"] > df["room_capacity"]).sum()
    negative_vals = (df["occupied_rooms"] < 0).sum()
    print("Over-capacity records:", over_capacity)
    print("Negative occupancy records:", negative_vals)

    print("\n=== SUMMARY BY ROOM CATEGORY ===")
    summary = df.groupby("room_category").agg(
        capacity=("room_capacity", "first"),
        mean_occupied=("occupied_rooms", "mean"),
        std_occupied=("occupied_rooms", "std"),
        mean_occ_pct=("occupancy_rate", "mean"),
        mean_otb_7d=("otb_7d_prior", "mean"),
    ).round(2)
    print(summary)

    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(8, 4))
    summary["mean_occ_pct"].plot(kind="bar", color="#2b5c8f")
    plt.title("Average Occupancy Rate (%) by Room Category")
    plt.ylabel("Occupancy (%)")
    plt.xlabel("Room Category")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reports/figures/category_occupancy.png", dpi=150)
    plt.close()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_avg = df.groupby("day_name")["occupancy_rate"].mean().reindex(day_order)

    plt.figure(figsize=(9, 4))
    dow_avg.plot(kind="line", marker="o", color="#d95f02", linewidth=2)
    plt.title("Average Occupancy Rate (%) by Day of Week")
    plt.ylabel("Occupancy (%)")
    plt.xlabel("Day of Week")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/figures/weekday_pattern.png", dpi=150)
    plt.close()

    std_df = df[(df["room_category"] == "Standard") & (df["stay_date"] >= "2025-10-01")]
    plt.figure(figsize=(10, 4))
    plt.plot(std_df["stay_date"], std_df["occupied_rooms"], label="Actual Occupied Rooms", color="#1b9e77")
    plt.plot(std_df["stay_date"], std_df["otb_7d_prior"], label="On-The-Books (7 Days Prior)", linestyle="--", color="#7570b3")
    plt.title("Standard Rooms: Actual Occupancy vs. 7-Day Prior Bookings (Q4 2025)")
    plt.ylabel("Room Nights")
    plt.xlabel("Stay Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/otb_vs_actual.png", dpi=150)
    plt.close()

    print("\nSaved charts to reports/figures/")

if __name__ == "__main__":
    run_eda()
