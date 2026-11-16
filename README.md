# Saudi Used Car Price Prediction 🚗

## Overview
This project predicts used car prices in the Saudi Arabian market using machine learning. It analyzes the key factors that drive used car pricing and builds a regression model capable of estimating a fair market price for a given vehicle.

The project follows a complete, end-to-end data science workflow: data cleaning, exploratory data analysis (EDA), feature engineering, model training with cross-validation, hyperparameter tuning, final evaluation, and error analysis.

## Project Structure
```
├── used_car_price_prediction.ipynb   # Full notebook: EDA, modeling, evaluation, error analysis
├── used_car_price_prediction.py      # Same pipeline as a plain script (no notebook needed)
├── saudi_used_cars_csv_.csv          # Raw dataset
├── rf_model.joblib                   # Saved, trained model (ready for inference)
├── model_predictions.csv             # Test-set predictions (true vs predicted price, error bands)
├── queries.sql                       # 11 analytical SQL queries (market analysis + model performance)
├── sql_analysis.py                   # Builds the SQLite database (saudi_used_cars.db) and runs queries.sql
├── requirements.txt                  # Python dependencies
└── README.md
```

## Dataset
- **8,035** original listings
- **5,370** listings after cleaning
- Columns: `Make`, `Type`, `Year`, `Origin`, `Color`, `Options`, `Engine_Size`, `Fuel_Type`, `Gear_Type`, `Mileage`, `Region`, `Price`, `Negotiable`

### Data cleaning decisions
- **`Price == 0` listings (31.4% of raw data)**: these are listings marked `Negotiable = TRUE` with no listed asking price, not an actual price of zero. They are dropped rather than imputed, since a fabricated price would corrupt the training label.
- **Outliers**: top/bottom 1% of `Price` and top 0.5% of `Mileage` are trimmed to reduce the influence of rare luxury listings and likely data-entry errors.
- **Target transform**: `Price` is log-transformed before training, since raw prices are heavily right-skewed.

## Exploratory Data Analysis
The notebook includes:
- Price distribution (raw vs. cleaned vs. log-transformed)
- Median price trend by model year
- Median price by the 10 most-listed makes
- Mileage vs. price scatter plot
- Correlation matrix of numeric features (`Year`, `Engine_Size`, `Mileage`, `Price`)

## Modeling & Evaluation

| Model | Cross-Val R² (train) | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| Linear Regression | 0.530 | — | — | — |
| Random Forest (default params) | 0.709 | — | — | — |
| **Random Forest (tuned)** | **0.720** | **0.799** | **26,567** | **13,200** |

- **Cross-validation**: 3-fold CV on the training set is used to get a more reliable performance estimate than a single train/test split.
- **Hyperparameter tuning**: `RandomizedSearchCV` searches over tree count, max depth, split/leaf sizes, and feature sampling ratio.
- **Final model**: the tuned Random Forest, evaluated once on a held-out 20% test set.

### Error analysis
- Mean absolute percentage error on the test set is ~42%.
- The largest errors cluster in **rare, high-end makes** (Maserati, Porsche, Land Rover, Mercedes) and in the generic `Other` make/type category — these have few listings, so the model has little signal to learn their pricing pattern from.
- **Practical takeaway**: predictions are reliable as a general market estimate for common makes, but should be treated as a wide range — not an exact valuation — for luxury or rare vehicles.

## How to Run

**Notebook:**
```bash
pip install -r requirements.txt
jupyter notebook used_car_price_prediction.ipynb
```

**Script:**
```bash
pip install -r requirements.txt
python used_car_price_prediction.py
```

**Using the saved model directly:**
```python
import joblib
model = joblib.load("rf_model.joblib")
# model.predict(new_data)  # new_data must have the same columns as the training features
```

## SQL Analysis (SQLite)
The data is also loaded into a local **SQLite** database so the market and the model results can be queried with SQL. `sql_analysis.py` creates `saudi_used_cars.db` with:

| Object | Type | Description |
|---|---|---|
| `cars_raw` | table | The 8,035 raw listings, exactly as in the CSV |
| `cleaning_thresholds` | table | The percentile cut-offs used by the Python pipeline (1st/99th percentile of `Price`, 99.5th percentile of `Mileage`) |
| `cars_clean` | view | Raw listings after the same cleaning rules, plus `Car_Age`: **5,370 rows**, matching the Python pipeline |
| `model_predictions` | table | The 1,074 test-set predictions from the tuned Random Forest |

`queries.sql` contains 11 queries that use `GROUP BY`/`HAVING`, `CASE`, joins on sub-queries, CTEs, and window functions (`RANK`, `SUM() OVER`):

| # | Question | SQL techniques |
|---|---|---|
| Q1 | How many raw listings have no asking price? | Aggregates, conditional `SUM` |
| Q2 | Average price and mileage by make | `GROUP BY`, `HAVING` |
| Q3 | Average price by region | `GROUP BY` |
| Q4 | Depreciation by car age bucket | `CASE` |
| Q5 | Does the option level (Full / Semi Full / Standard) matter? | `GROUP BY` |
| Q6 | Each make's price premium over its region's average | Join on a sub-query |
| Q7 | Top 3 most expensive makes in each region | CTE, `RANK() OVER` |
| Q8 | Distribution of model accuracy bands | `CASE`, `SUM() OVER` |
| Q9 | Which makes have the largest model errors? | `GROUP BY`, `HAVING` |
| Q10 | Does the model over- or under-predict by price segment? | `CASE`, signed error |
| Q11 | Model error vs. the market's average price per make | Join on a sub-query |

### Key SQL findings
- **Missing prices**: 2,527 raw listings (31.4%) have `Price = 0`, and 2,526 of them are marked `Negotiable`, which confirms they are "no asking price" rows, not real prices of zero.
- **Depreciation**: average price falls from about 139K SAR for 3-5 year old cars, to 88K SAR for 6-10 years, to 54K SAR for 11+ years.
- **Options matter**: Full-option cars average about 101K SAR, versus 68K for Semi Full and 53K for Standard.
- **Make drives price**: Land Rover (about 186K SAR), Mercedes (154K), and Lexus (148K) are the most expensive makes among those with 50+ listings. Kia and Hyundai are the cheapest (about 44K to 45K).
- **Regions**: among the three biggest markets, Dammam has the highest average price (about 87K SAR), ahead of Riyadh (76K) and Jeddah (73K), largely because of its mix of Mercedes and Lexus listings.
- **Model accuracy**: about 75% of test predictions are within 25% of the true price (Excellent + Good), and about 9% are off by more than 50%.
- **Model bias**: the model tends to **over-predict cheap cars** (under 40K SAR: +4.6K SAR on average) and **under-predict expensive ones** (100K+ SAR: -25.6K SAR on average). Land Rover and Mercedes have the largest average errors (about 34K SAR each).

> **Notes**
> - `Negotiable` is loaded as `1`/`0` (pandas parses `TRUE`/`FALSE` as booleans), so queries filter with `Negotiable = 1`.
> - 8 test cars with extreme errors (above about 1,800%) have a blank `Prediction_Accuracy_Band` in `model_predictions.csv`. Q8 rebuilds the bands from `Pct_Error` with `CASE`, so its counts differ by a few cars from the pre-computed column.
> - The cleaned data has no cars younger than 3 years, so the "0-2 years" bucket in Q4 is empty.

**Run the SQL step:**
```bash
python sql_analysis.py   # builds saudi_used_cars.db, then runs and prints all queries in queries.sql
```
SQLite needs no installation, since `sqlite3` ships with Python. The script reads `saudi_used_cars_csv_.csv` and `model_predictions.csv` from the same folder. You can also open the generated `saudi_used_cars.db` in any SQLite client (such as DB Browser for SQLite) and run `queries.sql` directly.

## Power BI Dashboard
`model_predictions.csv` contains the test-set results — original car features, `True_Price`, `Predicted_Price`, `Abs_Error`, `Pct_Error`, and a pre-computed `Prediction_Accuracy_Band` (Excellent / Good / Fair / Poor) — so model performance can be explored visually without re-running Python.

To build a dashboard:
1. Import `saudi_used_cars_csv_.csv` (semicolon-delimited) for a market overview — price by make, by year, by region.
2. Import `model_predictions.csv` (comma-delimited) for model-performance visuals — True vs. Predicted scatter, error by make, and the accuracy-band distribution.
3. Optionally relate the two tables on `Make` for a combined view.

## Technologies Used
- Python, Pandas, NumPy
- SQL (SQLite, `sqlite3`)
- Matplotlib
- Scikit-learn (RandomForestRegressor, RandomizedSearchCV, cross-validation)
- Jupyter Notebook
- Joblib (model persistence)

## Skills Demonstrated
- Data Cleaning & handling of ambiguous/missing values
- SQL: aggregation, joins, CTEs, and window functions
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Regression Modeling (Linear Regression, Random Forest)
- Cross-Validation & Hyperparameter Tuning
- Model Evaluation & Error Analysis
- Model Persistence for deployment/inference

## Limitations & Future Work
- Add more listings for luxury/rare makes to reduce error in that segment.
- Try gradient boosting models (XGBoost/LightGBM) for a possible further accuracy gain.
- Add a simple web app (e.g. Streamlit) around `rf_model.joblib` for interactive price estimation.

## Author
**Rayan Omar**
Computer Science Student | Data Science Track
