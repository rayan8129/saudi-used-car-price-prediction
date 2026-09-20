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

## Power BI Dashboard
`model_predictions.csv` contains the test-set results — original car features, `True_Price`, `Predicted_Price`, `Abs_Error`, `Pct_Error`, and a pre-computed `Prediction_Accuracy_Band` (Excellent / Good / Fair / Poor) — so model performance can be explored visually without re-running Python.

To build a dashboard:
1. Import `saudi_used_cars_csv_.csv` (semicolon-delimited) for a market overview — price by make, by year, by region.
2. Import `model_predictions.csv` (comma-delimited) for model-performance visuals — True vs. Predicted scatter, error by make, and the accuracy-band distribution.
3. Optionally relate the two tables on `Make` for a combined view.

## Technologies Used
- Python, Pandas, NumPy
- Matplotlib
- Scikit-learn (RandomForestRegressor, RandomizedSearchCV, cross-validation)
- Jupyter Notebook
- Joblib (model persistence)

## Skills Demonstrated
- Data Cleaning & handling of ambiguous/missing values
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
