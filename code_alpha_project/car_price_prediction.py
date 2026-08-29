"""
Car Price Prediction — End-to-End Regression Pipeline
=======================================================
Predicts a used car's selling price from features such as age, present
(ex-showroom) price, kilometers driven, fuel type, seller type,
transmission, and ownership history.

Dataset: car_data.csv (302 rows, CarDekho used-car listings)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
csv_candidates = ["car_data.csv", "car data.csv"]
csv_path = next((name for name in csv_candidates if os.path.exists(name)), csv_candidates[0])
df = pd.read_csv(csv_path)
print("Loaded file:", csv_path)
print("Shape:", df.shape)
print(df.dtypes)
missing_total = int(df.isnull().sum().sum())
print("Total missing values:", missing_total)

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
CURRENT_YEAR = 2026
df["Car_Age"] = CURRENT_YEAR - df["Year"]

# Brand goodwill proxy: extract the first word of Car_Name as "Brand"
# then encode it by its historical average selling price (target encoding).
df["Brand"] = df["Car_Name"].apply(lambda x: x.split(" ")[0].lower())
brand_avg_price = df.groupby("Brand")["Selling_Price"].mean()
df["Brand_Goodwill"] = df["Brand"].map(brand_avg_price)

# Depreciation-style ratio: how much value the car has lost vs new price
df["Price_Drop_Ratio"] = (df["Present_Price"] - df["Selling_Price"]) / df["Present_Price"]

# Drop columns that would leak the target or aren't usable directly
model_df = df.drop(columns=["Car_Name", "Year", "Brand", "Price_Drop_Ratio"])

# One-hot encode categoricals
model_df = pd.get_dummies(
    model_df, columns=["Fuel_Type", "Selling_type", "Transmission"], drop_first=True
)

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
X = model_df.drop(columns=["Selling_Price"])
y = model_df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Train models
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
}

results = {}
predictions = {}

for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    predictions[name] = preds
    results[name] = {
        "R2": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
    }

results_df = pd.DataFrame(results).T
print("\nModel comparison:\n", results_df)

# 5-fold cross-validation for the stronger model (Random Forest)
cv_scores = cross_val_score(
    RandomForestRegressor(n_estimators=300, random_state=42), X, y, cv=5, scoring="r2"
)
print("\nRandom Forest 5-fold CV R2:", cv_scores, "mean:", cv_scores.mean())

# ---------------------------------------------------------------------------
# 5. Feature importance (Random Forest)
# ---------------------------------------------------------------------------
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#4C72B0")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ---------------------------------------------------------------------------
# 6. Actual vs Predicted plot (best model)
# ---------------------------------------------------------------------------
best_name = results_df["R2"].idxmax()
best_preds = predictions[best_name]

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.7, edgecolor="k")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Selling Price (lakhs)")
plt.ylabel("Predicted Selling Price (lakhs)")
plt.title(f"Actual vs Predicted — {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

# ---------------------------------------------------------------------------
# 7. Correlation heatmap
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 6))
numeric_cols = df[["Selling_Price", "Present_Price", "Driven_kms", "Car_Age",
                    "Owner", "Brand_Goodwill"]]
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------------------
# 8. Save best model + scaler + feature list
# ---------------------------------------------------------------------------
joblib.dump(rf_model, "car_price_model_rf.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(list(X.columns), "feature_columns.joblib")

print("\nBest model:", best_name)
print("Saved: car_price_model_rf.joblib, scaler.joblib, feature_columns.joblib")
print("Saved plots: feature_importance.png, actual_vs_predicted.png, correlation_heatmap.png")
