"""SQL analysis for the Saudi Used Car Price Prediction project.

1. Builds a local SQLite database (saudi_used_cars.db) from the two CSV files.
2. Runs every query in queries.sql and prints the results.

Objects created:
  cars_raw            - the 8,035 raw listings, exactly as in the CSV
  cleaning_thresholds - the percentile cut-offs used by the Python pipeline
  cars_clean (view)   - raw listings after the same cleaning rules (5,370 rows)
  model_predictions   - test-set predictions from the tuned Random Forest

Usage:  python sql_analysis.py
"""
import re
import sqlite3
import pandas as pd

RAW_CSV = "saudi_used_cars_csv_.csv"      # semicolon-delimited
PRED_CSV = "model_predictions.csv"        # comma-delimited
DB_FILE = "saudi_used_cars.db"
QUERIES_FILE = "queries.sql"

# ---------- 1. Build the database ----------
raw = pd.read_csv(RAW_CSV, sep=";", encoding="utf-8-sig")
preds = pd.read_csv(PRED_CSV)

# Same cut-offs as used_car_price_prediction.py:
#   Price > 0, Price within the 1st-99th percentile, Mileage below the 99.5th percentile.
priced = raw[raw["Price"] > 0]
price_low, price_high = priced["Price"].quantile([0.01, 0.99])
price_trimmed = priced[(priced["Price"] >= price_low) & (priced["Price"] <= price_high)]
mileage_cap = price_trimmed["Mileage"].quantile(0.995)

conn = sqlite3.connect(DB_FILE)
raw.to_sql("cars_raw", conn, if_exists="replace", index=False)
preds.to_sql("model_predictions", conn, if_exists="replace", index=False)
pd.DataFrame(
    {"price_low": [price_low], "price_high": [price_high], "mileage_cap": [mileage_cap]}
).to_sql("cleaning_thresholds", conn, if_exists="replace", index=False)

conn.executescript("""
DROP VIEW IF EXISTS cars_clean;
CREATE VIEW cars_clean AS
SELECT r.*, 2026 - r.Year AS Car_Age
FROM cars_raw r, cleaning_thresholds t
WHERE r.Price > 0
  AND r.Price >= t.price_low AND r.Price <= t.price_high
  AND r.Mileage < t.mileage_cap;
""")
conn.commit()

print("DATABASE BUILT")
for name in ("cars_raw", "cars_clean", "model_predictions"):
    n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    print(f"  {name}: {n:,} rows")

# ---------- 2. Run the queries ----------
sql = open(QUERIES_FILE, encoding="utf-8").read()
for block in [b.strip() for b in sql.split(";") if "SELECT" in b or "WITH" in b]:
    title = re.search(r"-- (Q\d+\..*)", block)
    print("\n" + (title.group(1) if title else ""))
    print(pd.read_sql(block + ";", conn).to_string(index=False))
conn.close()
