import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib, json, sys

RANDOM_STATE = 42

df_raw = pd.read_csv("/mnt/user-data/uploads/saudi_used_cars_csv_.csv", sep=";", encoding="utf-8-sig")
print("RAW SHAPE:", df_raw.shape); sys.stdout.flush()

plt.figure(figsize=(7,4))
plt.hist(df_raw['Price'], bins=60)
plt.title("Raw Price Distribution (includes Price=0 negotiable listings)")
plt.xlabel("Price (SAR)"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig("plots/eda_price_raw.png"); plt.close()

zero_price_pct = (df_raw['Price']==0).mean()*100
print(f"Zero-price rows: {zero_price_pct:.1f}%"); sys.stdout.flush()

df = df_raw[df_raw['Price']>0].copy()
low, high = df['Price'].quantile([0.01,0.99])
df = df[(df['Price']>=low)&(df['Price']<=high)]
df = df[df['Mileage'] < df['Mileage'].quantile(0.995)]
print("CLEAN SHAPE:", df.shape); sys.stdout.flush()

plt.figure(figsize=(7,4)); plt.hist(df['Price'], bins=60, color='seagreen')
plt.title("Cleaned Price Distribution"); plt.xlabel("Price (SAR)"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig("plots/eda_price_clean.png"); plt.close()

plt.figure(figsize=(7,4)); plt.hist(np.log1p(df['Price']), bins=60, color='darkorange')
plt.title("Log(Price+1) Distribution — closer to normal")
plt.xlabel("log(Price+1)"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig("plots/eda_price_log.png"); plt.close()

year_price = df.groupby('Year')['Price'].median().sort_index()
plt.figure(figsize=(8,4)); plt.plot(year_price.index, year_price.values, marker='o')
plt.title("Median Price by Model Year"); plt.xlabel("Year"); plt.ylabel("Median Price (SAR)")
plt.tight_layout(); plt.savefig("plots/eda_price_by_year.png"); plt.close()

top_makes = df['Make'].value_counts().head(10).index
make_price = df[df['Make'].isin(top_makes)].groupby('Make')['Price'].median().sort_values(ascending=False)
plt.figure(figsize=(8,4)); plt.bar(make_price.index, make_price.values, color='steelblue')
plt.title("Median Price — Top 10 Most Listed Makes"); plt.ylabel("Median Price (SAR)")
plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig("plots/eda_price_by_make.png"); plt.close()

plt.figure(figsize=(7,5)); plt.scatter(df['Mileage'], df['Price'], alpha=0.15, s=10)
plt.title("Mileage vs Price"); plt.xlabel("Mileage (km)"); plt.ylabel("Price (SAR)")
plt.tight_layout(); plt.savefig("plots/eda_mileage_vs_price.png"); plt.close()

num_df = df[['Year','Engine_Size','Mileage','Price']].corr()
print("\nCorrelation matrix:\n", num_df.round(2)); sys.stdout.flush()
plt.figure(figsize=(5,4)); im = plt.imshow(num_df, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(num_df.columns)), num_df.columns, rotation=45, ha='right')
plt.yticks(range(len(num_df.columns)), num_df.columns)
for i in range(len(num_df.columns)):
    for j in range(len(num_df.columns)):
        plt.text(j, i, f"{num_df.iloc[i,j]:.2f}", ha='center', va='center', color='black', fontsize=9)
plt.colorbar(im, fraction=0.046); plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout(); plt.savefig("plots/eda_correlation.png"); plt.close()
print("EDA plots saved."); sys.stdout.flush()

df['Car_Age'] = 2026 - df['Year']
X = df.drop(columns=['Price','Negotiable'])
y = np.log1p(df['Price'])
cat_cols = X.select_dtypes(include='object').columns.tolist()
print("Categorical cols:", cat_cols); sys.stdout.flush()

pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)], remainder='passthrough')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"Train size: {len(X_train)}  Test size: {len(X_test)}"); sys.stdout.flush()

kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

lin_pipe = Pipeline([('pre', pre), ('model', LinearRegression())])
lin_cv = cross_val_score(lin_pipe, X_train, y_train, cv=kf, scoring='r2')
print(f"LinearRegression 3-fold CV R2: {lin_cv.mean():.3f} +/- {lin_cv.std():.3f}"); sys.stdout.flush()

rf_base = Pipeline([('pre', pre), ('model', RandomForestRegressor(n_estimators=100, max_depth=16, random_state=RANDOM_STATE, n_jobs=1))])
rf_cv = cross_val_score(rf_base, X_train, y_train, cv=kf, scoring='r2', n_jobs=1)
print(f"RandomForest (default params) 3-fold CV R2: {rf_cv.mean():.3f} +/- {rf_cv.std():.3f}"); sys.stdout.flush()

param_dist = {
    'model__n_estimators': [80, 120, 150],
    'model__max_depth': [12, 16, 20],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt', 0.5],
}
search_pipe = Pipeline([('pre', pre), ('model', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))])
search = RandomizedSearchCV(search_pipe, param_distributions=param_dist, n_iter=4, cv=kf, scoring='r2', random_state=RANDOM_STATE, n_jobs=1, verbose=1)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print(f"Best CV R2: {search.best_score_:.3f}"); sys.stdout.flush()

best_model = search.best_estimator_
pred_log = best_model.predict(X_test)
pred = np.expm1(pred_log); true = np.expm1(y_test)
r2 = r2_score(true, pred); rmse = np.sqrt(mean_squared_error(true, pred)); mae = mean_absolute_error(true, pred)
print(f"FINAL TEST SET -> R2={r2:.3f} RMSE={rmse:,.0f} MAE={mae:,.0f}"); sys.stdout.flush()

ohe = best_model.named_steps['pre'].named_transformers_['cat']
feature_names = list(ohe.get_feature_names_out(cat_cols)) + [c for c in X.columns if c not in cat_cols]
importances = best_model.named_steps['model'].feature_importances_
top_idx = np.argsort(importances)[-15:]
plt.figure(figsize=(8,6)); plt.barh(np.array(feature_names)[top_idx], importances[top_idx], color='teal')
plt.xlabel("Importance"); plt.title("Top 15 Feature Importances — Tuned Random Forest")
plt.tight_layout(); plt.savefig("plots/feature_importance.png"); plt.close()

results_df = X_test.copy()
results_df['True_Price'] = true.values
results_df['Predicted_Price'] = pred
results_df['Abs_Error'] = np.abs(results_df['True_Price'] - results_df['Predicted_Price'])
results_df['Pct_Error'] = results_df['Abs_Error'] / results_df['True_Price'] * 100
print("Mean absolute % error:", round(results_df['Pct_Error'].mean(),1), "%")

worst = results_df.sort_values('Abs_Error', ascending=False).head(10)
print("Worst 10 predictions:\n", worst[['Make','Type','Year','Mileage','True_Price','Predicted_Price','Abs_Error']].to_string())

err_by_make = results_df.groupby('Make')['Abs_Error'].mean().sort_values(ascending=False).head(10)
print("Mean abs error by make (top 10 worst):\n", err_by_make.round(0))
sys.stdout.flush()

plt.figure(figsize=(7,5)); plt.scatter(results_df['True_Price'], results_df['Predicted_Price'], alpha=0.3, s=12)
mx = results_df['True_Price'].max(); plt.plot([0,mx],[0,mx],'r--', label='Perfect prediction')
plt.xlabel("True Price"); plt.ylabel("Predicted Price"); plt.title("Predicted vs True Price (Test Set)")
plt.legend(); plt.tight_layout(); plt.savefig("plots/pred_vs_true.png"); plt.close()

plt.figure(figsize=(7,4)); residuals = results_df['True_Price'] - results_df['Predicted_Price']
plt.hist(residuals, bins=50, color='indianred'); plt.axvline(0, color='black', linestyle='--')
plt.title("Residuals Distribution (True - Predicted)"); plt.xlabel("Residual (SAR)"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig("plots/residuals.png"); plt.close()

joblib.dump(best_model, "rf_model.joblib")
print("Model saved.")

summary = {
    "linreg_cv_r2_mean": float(lin_cv.mean()), "linreg_cv_r2_std": float(lin_cv.std()),
    "rf_default_cv_r2_mean": float(rf_cv.mean()), "rf_default_cv_r2_std": float(rf_cv.std()),
    "rf_tuned_cv_r2": float(search.best_score_), "best_params": search.best_params_,
    "test_r2": float(r2), "test_rmse": float(rmse), "test_mae": float(mae),
    "mean_pct_error": float(results_df['Pct_Error'].mean()),
    "rows_before_cleaning": int(len(df_raw)), "rows_after_cleaning": int(len(df)),
    "zero_price_pct": float(zero_price_pct),
}
with open("summary.json","w") as f: json.dump(summary, f, indent=2)
print("DONE")
