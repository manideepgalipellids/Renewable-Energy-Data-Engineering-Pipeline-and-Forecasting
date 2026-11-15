import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Phase_3_Spark_Preprocessing",
    "output"
)

OUTPUT_PATH = os.path.join(BASE_DIR, "ML_Output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 70)
print("        RENEWABLE ENERGY MACHINE LEARNING")
print("=" * 70)

# =====================================================
# LOAD DATASETS
# =====================================================

print("\nLoading Processed Datasets...\n")

solar_df = pd.read_csv(
    os.path.join(DATA_PATH, "solar_processed.csv")
)

wind_df = pd.read_csv(
    os.path.join(DATA_PATH, "wind_processed.csv")
)

print("Solar Dataset Shape :", solar_df.shape)
print("Wind Dataset Shape  :", wind_df.shape)

# =====================================================
# SOLAR MACHINE LEARNING MODEL
# =====================================================

print("\n" + "=" * 70)
print("TRAINING SOLAR POWER PREDICTION MODEL")
print("=" * 70)

solar_features = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "hour",
    "day_of_week",
    "month",
    "year"
]

solar_target = "DC_POWER"

solar_df = solar_df.dropna(
    subset=solar_features + [solar_target]
)

X = solar_df[solar_features]
y = solar_df[solar_target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

solar_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

solar_model.fit(X_train, y_train)

solar_predictions = solar_model.predict(X_test)

# =====================================================
# SOLAR EVALUATION METRICS
# =====================================================

solar_mae = mean_absolute_error(
    y_test,
    solar_predictions
)

solar_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        solar_predictions
    )
)

solar_r2 = r2_score(
    y_test,
    solar_predictions
)

print("\nSolar Model Performance")
print("-" * 40)
print(f"MAE  : {solar_mae:.2f}")
print(f"RMSE : {solar_rmse:.2f}")
print(f"R²   : {solar_r2:.4f}")

# =====================================================
# SAVE SOLAR MODEL
# =====================================================

joblib.dump(
    solar_model,
    os.path.join(
        OUTPUT_PATH,
        "solar_model.pkl"
    )
)

print("\nSolar model saved successfully.")

# =====================================================
# SOLAR PREDICTION PLOT (PDF)
# =====================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    solar_predictions,
    color="royalblue",
    alpha=0.6,
    s=25,
    edgecolors="black",
    linewidth=0.3,
    label="Predicted Values"
)

minimum = min(
    y_test.min(),
    solar_predictions.min()
)

maximum = max(
    y_test.max(),
    solar_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linewidth=2,
    linestyle="--",
    label="Ideal Prediction"
)

plt.title(
    "Solar Power Prediction\nActual vs Predicted",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Actual DC Power",
    fontsize=13
)

plt.ylabel(
    "Predicted DC Power",
    fontsize=13
)

plt.grid(True, linestyle="--", alpha=0.4)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "solar_prediction_results.pdf"
    ),
    format="pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Solar prediction PDF generated successfully.")

# =====================================================
# WIND MACHINE LEARNING MODEL
# =====================================================

print("\n" + "=" * 70)
print("TRAINING WIND POWER PREDICTION MODEL")
print("=" * 70)

wind_features = [
    "Wind Speed (m/s)",
    "Wind Direction (°)",
    "Theoretical_Power_Curve (KWh)"
]

wind_target = "LV ActivePower (kW)"

wind_df = wind_df.dropna(
    subset=wind_features + [wind_target]
)

X = wind_df[wind_features]
y = wind_df[wind_target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

wind_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

wind_model.fit(X_train, y_train)

wind_predictions = wind_model.predict(X_test)

# =====================================================
# WIND EVALUATION METRICS
# =====================================================

wind_mae = mean_absolute_error(
    y_test,
    wind_predictions
)

wind_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        wind_predictions
    )
)

wind_r2 = r2_score(
    y_test,
    wind_predictions
)

print("\nWind Model Performance")
print("-" * 40)
print(f"MAE  : {wind_mae:.2f}")
print(f"RMSE : {wind_rmse:.2f}")
print(f"R²   : {wind_r2:.4f}")

# =====================================================
# SAVE WIND MODEL
# =====================================================

joblib.dump(
    wind_model,
    os.path.join(
        OUTPUT_PATH,
        "wind_model.pkl"
    )
)

print("\nWind model saved successfully.")

# =====================================================
# WIND PREDICTION PLOT (PDF)
# =====================================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    wind_predictions,
    color="forestgreen",
    alpha=0.6,
    s=25,
    edgecolors="black",
    linewidth=0.3,
    label="Predicted Values"
)

minimum = min(
    y_test.min(),
    wind_predictions.min()
)

maximum = max(
    y_test.max(),
    wind_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Ideal Prediction"
)

plt.title(
    "Wind Power Prediction\nActual vs Predicted",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Actual Wind Power (kW)",
    fontsize=13
)

plt.ylabel(
    "Predicted Wind Power (kW)",
    fontsize=13
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.4
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "wind_prediction_results.pdf"
    ),
    format="pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Wind prediction PDF generated successfully.")

# =====================================================
# MODEL PERFORMANCE COMPARISON (PDF)
# =====================================================

print("\n" + "=" * 70)
print("GENERATING MODEL COMPARISON CHART")
print("=" * 70)

metrics = ["MAE", "RMSE", "R² Score"]

solar_values = [
    solar_mae,
    solar_rmse,
    solar_r2
]

wind_values = [
    wind_mae,
    wind_rmse,
    wind_r2
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10,6))

bars1 = plt.bar(
    x - width/2,
    solar_values,
    width,
    label="Solar Model",
    color="orange"
)

bars2 = plt.bar(
    x + width/2,
    wind_values,
    width,
    label="Wind Model",
    color="seagreen"
)

plt.xticks(
    x,
    metrics,
    fontsize=12
)

plt.ylabel(
    "Metric Value",
    fontsize=13
)

plt.title(
    "Solar vs Wind Model Performance Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.legend()

# Show values on bars

for bar in bars1:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

for bar in bars2:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "model_metrics_comparison.pdf"
    ),
    format="pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Model comparison PDF saved.")

# =====================================================
# SOLAR FEATURE IMPORTANCE
# =====================================================

print("\nGenerating Solar Feature Importance...")

solar_importance = solar_model.feature_importances_

importance_df = pd.DataFrame({

    "Feature": solar_features,
    "Importance": solar_importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(8,6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="orange"
)

plt.xlabel(
    "Importance Score",
    fontsize=12
)

plt.ylabel(
    "Features",
    fontsize=12
)

plt.title(
    "Solar Model Feature Importance",
    fontsize=15,
    fontweight="bold"
)

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "solar_feature_importance.pdf"
    ),
    format="pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Solar Feature Importance PDF saved.")

# =====================================================
# WIND FEATURE IMPORTANCE
# =====================================================

print("\nGenerating Wind Feature Importance...")

wind_importance = wind_model.feature_importances_

importance_df = pd.DataFrame({

    "Feature": wind_features,
    "Importance": wind_importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(8,6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="forestgreen"
)

plt.xlabel(
    "Importance Score",
    fontsize=12
)

plt.ylabel(
    "Features",
    fontsize=12
)

plt.title(
    "Wind Model Feature Importance",
    fontsize=15,
    fontweight="bold"
)

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "wind_feature_importance.pdf"
    ),
    format="pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Wind Feature Importance PDF saved.")

# =====================================================
# SAVE MODEL METRICS
# =====================================================

print("\n" + "=" * 70)
print("SAVING MODEL EVALUATION METRICS")
print("=" * 70)

metrics_file = os.path.join(
    OUTPUT_PATH,
    "model_metrics.txt"
)

with open(metrics_file, "w") as f:

    f.write("=" * 60 + "\n")
    f.write("RENEWABLE ENERGY MACHINE LEARNING RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write("SOLAR MODEL\n")
    f.write("-" * 30 + "\n")
    f.write(f"MAE      : {solar_mae:.4f}\n")
    f.write(f"RMSE     : {solar_rmse:.4f}\n")
    f.write(f"R² Score : {solar_r2:.4f}\n\n")

    f.write("WIND MODEL\n")
    f.write("-" * 30 + "\n")
    f.write(f"MAE      : {wind_mae:.4f}\n")
    f.write(f"RMSE     : {wind_rmse:.4f}\n")
    f.write(f"R² Score : {wind_r2:.4f}\n")

print("Model metrics saved successfully.")

# =====================================================
# SAVE METRICS AS CSV
# =====================================================

metrics_df = pd.DataFrame({

    "Model": [
        "Solar",
        "Wind"
    ],

    "MAE": [
        solar_mae,
        wind_mae
    ],

    "RMSE": [
        solar_rmse,
        wind_rmse
    ],

    "R2 Score": [
        solar_r2,
        wind_r2
    ]

})

metrics_df.to_csv(

    os.path.join(
        OUTPUT_PATH,
        "model_metrics.csv"
    ),

    index=False

)

print("CSV summary generated.")

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n")
print("=" * 70)
print(" MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY ")
print("=" * 70)

print("\nGenerated Files\n")

print("Models")
print("------")
print("✓ solar_model.pkl")
print("✓ wind_model.pkl")

print("\nPrediction Results")
print("------------------")
print("✓ solar_prediction_results.pdf")
print("✓ wind_prediction_results.pdf")

print("\nPerformance Comparison")
print("----------------------")
print("✓ model_metrics_comparison.pdf")

print("\nFeature Importance")
print("------------------")
print("✓ solar_feature_importance.pdf")
print("✓ wind_feature_importance.pdf")

print("\nEvaluation Reports")
print("------------------")
print("✓ model_metrics.txt")
print("✓ model_metrics.csv")

print("\nOutput Directory")
print("----------------")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print(" All machine learning outputs generated successfully.")
print("=" * 70)