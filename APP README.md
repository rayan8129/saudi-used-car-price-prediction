# Saudi Car Value — Streamlit App

A real, working price-estimator app for the Saudi Used Car Price Prediction project.
Enter a car's details and get an instant market price estimate from the trained model.

## Files needed (all in the same folder)
- `app.py` — the Streamlit app
- `rf_model.joblib` — the trained Random Forest pipeline
- `app_reference_data.json` — dropdown options and market medians (generated from the training data)

## How to run

```bash
pip install streamlit scikit-learn pandas numpy joblib
streamlit run app.py
```

This opens the app in your browser automatically, usually at:
```
http://localhost:8501
```

## Notes
- This was built and validated in an environment without internet access, so Streamlit itself
  could not be installed or run there — the app was tested by running its prediction logic
  directly (model loading + inference) outside Streamlit to confirm it works end to end.
  Run it on your own machine (with internet) to see the actual app and take a real screenshot.
- The dark UI styling (amber accent, cards, progress bars) is done via custom CSS injected
  through `st.markdown(..., unsafe_allow_html=True)` to match the project's mockup design.
- The estimated price range uses the model's mean absolute error (±13,200 SAR) from the test set.
- "Market Average" is the median price for the selected make, computed from the cleaned
  training data (`saudi_used_cars_csv_.csv`).
