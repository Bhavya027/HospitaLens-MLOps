
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "hospital_weekly_features.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading prepared hospital data...")
df = pd.read_csv(DATA_PATH, parse_dates=["feature_date", "target_date"])
df = df.sort_values("feature_date").reset_index(drop=True)

feature_columns = [
    "current_occupancy_beds",
    "previous_occupancy_beds",
    "two_weeks_ago_occupancy_beds",
    "total_beds_7_day_avg",
    "current_occupancy_rate",
]

target_column = "target_occupancy_beds"

# Chronological split: older dates train the model; newer dates test it.
cutoff = df["feature_date"].quantile(0.80)
train_df = df[df["feature_date"] < cutoff].copy()
test_df = df[df["feature_date"] >= cutoff].copy()

X_train = train_df[feature_columns]
y_train = train_df[target_column]
X_test = test_df[feature_columns]
y_test = test_df[target_column]

# Keep test rows with valid targets.
valid_test = y_test.notna()
X_test = X_test.loc[valid_test]
y_test = y_test.loc[valid_test]

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Training period: {train_df['feature_date'].min()} to {train_df['feature_date'].max()}")
print(f"Testing period: {test_df['feature_date'].min()} to {test_df['feature_date'].max()}")

def evaluate(name, actual, predicted):
    return {
        "model": name,
        "MAE": round(float(mean_absolute_error(actual, predicted)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        "R2": round(float(r2_score(actual, predicted)), 4),
    }

# Baseline predicts next week's occupancy using current occupancy.
results = [
    evaluate(
        "Persistence Baseline",
        y_test,
        X_test["current_occupancy_beds"],
    )
]

# Imputation is fitted only on training data through the pipeline.
linear_model = make_pipeline(
    SimpleImputer(strategy="median"),
    LinearRegression(),
)
linear_model.fit(X_train, y_train)
results.append(
    evaluate("Linear Regression", y_test, linear_model.predict(X_test))
)

forest_model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
)
forest_model.fit(X_train, y_train)
results.append(
    evaluate("Random Forest", y_test, forest_model.predict(X_test))
)

results_df = pd.DataFrame(results)
print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_path = MODEL_DIR / "metrics.json"
results_path.write_text(json.dumps(results, indent=4), encoding="utf-8")

# Save an ML model only if it beats the baseline on MAE.
baseline_mae = results_df.loc[
    results_df["model"] == "Persistence Baseline", "MAE"
].iloc[0]

ml_results = results_df[
    results_df["model"].isin(["Linear Regression", "Random Forest"])
]
best_row = ml_results.sort_values("MAE").iloc[0]

if best_row["MAE"] < baseline_mae:
    best_model = (
        linear_model
        if best_row["model"] == "Linear Regression"
        else forest_model
    )
    import joblib
    joblib.dump(best_model, MODEL_DIR / "model.pkl")
    print(f"\nSaved best ML model: {best_row['model']}")
else:
    print("\nNeither ML model beat the baseline. No ML model was saved.")

print(f"Metrics saved to: {results_path}")
