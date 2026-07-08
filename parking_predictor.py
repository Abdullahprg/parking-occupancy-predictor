"""
Usage:
    python parking_predictor.py setup-dev     # loads small synthetic test data (no internet needed)
    python parking_predictor.py fetch-real    # pulls REAL Seattle parking data (needs internet)
    python parking_predictor.py load-real     # cleans + loads the real data into SQLite
    python parking_predictor.py predict       # interactive predictor
"""

import argparse
import csv
import os
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # only needed for fetch-real

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "parking.db"
RAW_CSV_PATH = Path(__file__).parent / "parking_data_raw.csv"
DEV_FIXTURE_PATH = Path(__file__).parent / "sample_parking_data_DEV_ONLY.csv"

SEATTLE_DATASET_URL = "https://data.seattle.gov/resource/rke9-rsvs.json"
# Real dataset: "Paid Parking Occupancy (Last 30 Days)", Seattle Open Data
# https://data.seattle.gov/Transportation/Paid-Parking-Occupancy-Last-30-Days-/rke9-rsvs

WAIT_TIME_BREAKPOINTS = [
    (50, 0), (70, 3), (85, 8), (95, 15), (101, 25),
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS parking_data (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT NOT NULL,
    day_of_week         TEXT NOT NULL,
    hour                INTEGER NOT NULL,
    occupied_spots      INTEGER NOT NULL,
    total_spots         INTEGER NOT NULL,
    occupancy_percent   REAL NOT NULL,
    blockface_name      TEXT,
    source_element_key  TEXT,
    data_source         TEXT NOT NULL DEFAULT 'unknown'
);
"""
INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_day_hour ON parking_data (day_of_week, hour);"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(SCHEMA)
    conn.execute(INDEX_SQL)
    conn.commit()
    conn.close()
    print(f"[db] Database ready at {DB_PATH}")


def insert_rows(rows, data_source):
    if not rows:
        print("[db] No rows to insert.")
        return
    conn = get_connection()
    conn.executemany(
        """
        INSERT INTO parking_data
            (date, day_of_week, hour, occupied_spots, total_spots,
             occupancy_percent, blockface_name, source_element_key, data_source)
        VALUES
            (:date, :day_of_week, :hour, :occupied_spots, :total_spots,
             :occupancy_percent, :blockface_name, :source_element_key, :data_source)
        """,
        [{**r, "data_source": data_source} for r in rows],
    )
    conn.commit()
    conn.close()
    print(f"[db] Inserted {len(rows)} rows (source={data_source}).")


def clear_data_source(data_source):
    conn = get_connection()
    conn.execute("DELETE FROM parking_data WHERE data_source = ?", (data_source,))
    conn.commit()
    conn.close()


def fetch_all():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM parking_data").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def row_count():
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) AS n FROM parking_data").fetchone()["n"]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# REAL DATA FETCH (Seattle Open Data — genuinely public, no key needed for light use)
# ---------------------------------------------------------------------------
def fetch_seattle_parking_data(limit=5000, source_element_key=None):
    if requests is None:
        raise RuntimeError("Install 'requests' first: pip install requests --break-system-packages")

    headers = {}
    token = os.environ.get("SEATTLE_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token

    params = {"$limit": limit, "$order": "occupancydatetime DESC"}
    if source_element_key:
        params["sourceelementkey"] = source_element_key

    print(f"[fetch] Requesting up to {limit} rows from {SEATTLE_DATASET_URL} ...")
    resp = requests.get(SEATTLE_DATASET_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"[fetch] Received {len(data)} rows.")
    return data


def save_raw_csv(rows, path=RAW_CSV_PATH):
    if not rows:
        print("[fetch] No rows returned - nothing written.")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[fetch] Saved raw data to {path}")


# ---------------------------------------------------------------------------
# CLEANING (maps raw Seattle columns -> our agreed schema)
# ---------------------------------------------------------------------------
def parse_seattle_datetime(raw_value):
    for fmt in ["%m/%d/%Y %I:%M:%S %p", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(raw_value, fmt)
        except (ValueError, TypeError):
            continue
    return None


def clean_row(raw_row):
    def get(*keys):
        for k in keys:
            if k in raw_row and raw_row[k] not in (None, ""):
                return raw_row[k]
        return None

    dt = parse_seattle_datetime(get("OccupancyDateTime", "occupancydatetime"))
    if dt is None:
        return None

    try:
        occupied = int(float(get("PaidOccupancy", "paidoccupancy")))
        total = int(float(get("ParkingSpaceCount", "parkingspacecount")))
    except (TypeError, ValueError):
        return None

    if total <= 0 or occupied < 0 or occupied > total:
        return None

    return {
        "date": dt.strftime("%Y-%m-%d"),
        "day_of_week": dt.strftime("%A"),
        "hour": dt.hour,
        "occupied_spots": occupied,
        "total_spots": total,
        "occupancy_percent": round((occupied / total) * 100, 2),
        "blockface_name": get("BlockfaceName", "blockfacename"),
        "source_element_key": str(get("SourceElementKey", "sourceelementkey") or ""),
    }


def clean_file(path):
    with open(path, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    print(f"[clean] Read {len(raw_rows)} raw rows from {path}")
    cleaned = [clean_row(r) for r in raw_rows]
    cleaned = [r for r in cleaned if r is not None]
    dropped = len(raw_rows) - len(cleaned)
    print(f"[clean] Cleaned {len(cleaned)} rows, dropped {dropped} bad rows.")
    return cleaned


# ---------------------------------------------------------------------------
# SYNTHETIC DEV FIXTURE (clearly-labeled test data only — NOT real)
# ---------------------------------------------------------------------------
def generate_dev_fixture():
    random.seed(42)
    blockfaces = [
        ("1ST AVE BETWEEN PIKE ST AND PINE ST", "10123", 12),
        ("PIKE ST BETWEEN 1ST AVE AND 2ND AVE", "10456", 9),
        ("BROADWAY BETWEEN E PINE ST AND E OLIVE WAY", "20789", 15),
    ]
    rows = []
    now = datetime(2026, 6, 1)
    for d in range(14):
        day = now - timedelta(days=d)
        for hour in range(6, 22):
            for name, key, total in blockfaces:
                base = 30 + (35 if 10 <= hour <= 18 else 0)
                base += 15 if day.strftime("%A") in ("Saturday", "Sunday") else 0
                pct = max(0, min(100, base + random.randint(-10, 10)))
                occupied = max(0, min(total, round(total * pct / 100)))
                ts = day.replace(hour=hour, minute=random.randint(0, 59))
                rows.append({
                    "OccupancyDateTime": ts.strftime("%m/%d/%Y %I:%M:%S %p"),
                    "PaidOccupancy": occupied,
                    "ParkingSpaceCount": total,
                    "BlockfaceName": name,
                    "SourceElementKey": key,
                })

    fieldnames = ["OccupancyDateTime", "PaidOccupancy", "ParkingSpaceCount",
                  "BlockfaceName", "SourceElementKey"]
    with open(DEV_FIXTURE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[fixture] Wrote {len(rows)} SYNTHETIC rows to {DEV_FIXTURE_PATH} (test data only, not real).")


# ---------------------------------------------------------------------------
# BASELINE MODEL (grouped averages, no ML)
# ---------------------------------------------------------------------------
class BaselineModel:
    def __init__(self):
        rows = fetch_all()
        if not rows:
            raise ValueError("No data in DB yet. Run setup-dev or fetch-real + load-real first.")

        buckets = defaultdict(list)
        all_values = []
        for r in rows:
            buckets[(r["day_of_week"], r["hour"])].append(r["occupancy_percent"])
            all_values.append(r["occupancy_percent"])

        self.averages = {k: sum(v) / len(v) for k, v in buckets.items()}
        self.overall_average = sum(all_values) / len(all_values)
        self.sample_counts = {k: len(v) for k, v in buckets.items()}

    def predict(self, day_of_week, hour):
        key = (day_of_week, hour)
        if key in self.averages:
            return round(self.averages[key], 1), self.sample_counts[key], False
        return round(self.overall_average, 1), 0, True

    def predict_all_hours(self, day_of_week):
        return {h: self.predict(day_of_week, h)[0] for h in range(24)}


# ---------------------------------------------------------------------------
# PREDICT (matches agreed API response shape)
# ---------------------------------------------------------------------------
def estimate_wait_minutes(occupancy_percent):
    for threshold, wait in WAIT_TIME_BREAKPOINTS:
        if occupancy_percent < threshold:
            return wait
    return WAIT_TIME_BREAKPOINTS[-1][1]


def format_hour_12h(hour):
    period = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:00 {period}"


def find_best_time(model, day_of_week):
    hourly = model.predict_all_hours(day_of_week)
    best_hour = min(hourly, key=hourly.get)
    return format_hour_12h(best_hour), hourly[best_hour]


def predict_for(day_of_week, hour, model):
    occupancy_percent, sample_count, is_fallback = model.predict(day_of_week, hour)
    wait_minutes = estimate_wait_minutes(occupancy_percent)
    best_time, best_occupancy = find_best_time(model, day_of_week)
    return {
        "occupancy_percent": occupancy_percent,
        "wait_minutes": wait_minutes,
        "best_time": best_time,
        "_meta": {
            "sample_count": sample_count,
            "used_fallback_average": is_fallback,
            "best_time_predicted_occupancy_percent": best_occupancy,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_setup_dev():
    init_db()
    generate_dev_fixture()
    cleaned = clean_file(DEV_FIXTURE_PATH)
    clear_data_source("dev_fixture")
    insert_rows(cleaned, "dev_fixture")
    print(f"[main] Dev fixture loaded. Total rows: {row_count()}")


def cmd_fetch_real(limit=5000):
    rows = fetch_seattle_parking_data(limit=limit)
    save_raw_csv(rows)


def cmd_load_real():
    init_db()
    if not RAW_CSV_PATH.exists():
        print(f"[main] No raw data at {RAW_CSV_PATH}. Run fetch-real first.")
        return
    cleaned = clean_file(RAW_CSV_PATH)
    clear_data_source("seattle_real")
    insert_rows(cleaned, "seattle_real")
    print(f"[main] Real data loaded. Total rows: {row_count()}")


def cmd_predict():
    if row_count() == 0:
        print("[main] Database is empty. Run setup-dev or fetch-real + load-real first.")
        return
    model = BaselineModel()
    print("\nParking Occupancy Predictor (baseline, no ML/API yet). Type 'quit' to exit.\n")
    while True:
        day = input(f"Day of week {DAYS}: ").strip().capitalize()
        if day.lower() == "quit":
            break
        if day not in DAYS:
            print("  Please enter a valid day name.")
            continue
        hour_raw = input("Hour (0-23): ").strip()
        if hour_raw.lower() == "quit":
            break
        try:
            hour = int(hour_raw)
            assert 0 <= hour <= 23
        except (ValueError, AssertionError):
            print("  Please enter an hour between 0 and 23.")
            continue
        print(f"  -> {predict_for(day, hour, model)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["setup-dev", "fetch-real", "load-real", "predict"])
    args = parser.parse_args()

    if args.command == "setup-dev":
        cmd_setup_dev()
    elif args.command == "fetch-real":
        cmd_fetch_real()
    elif args.command == "load-real":
        cmd_load_real()
    elif args.command == "predict":
        cmd_predict()


if __name__ == "__main__":
    main()