import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# RENEWABLE ENERGY EDA VISUALIZATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Phase_3_Spark_Preprocessing",
    "output"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "EDA_Output"
)

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 70)
print("RENEWABLE ENERGY EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nLoading Processed Datasets...\n")

solar_df = pd.read_csv(
    os.path.join(DATA_PATH, "solar_processed.csv")
)

wind_df = pd.read_csv(
    os.path.join(DATA_PATH, "wind_processed.csv")
)

grid_df = pd.read_csv(
    os.path.join(DATA_PATH, "household_processed.csv")
)

daily_df = pd.read_csv(
    os.path.join(DATA_PATH, "daily_solar_generation.csv")
)

print(f"Solar Dataset      : {solar_df.shape}")
print(f"Wind Dataset       : {wind_df.shape}")
print(f"Household Dataset  : {grid_df.shape}")
print(f"Daily Dataset      : {daily_df.shape}")

# =====================================================
# SOLAR DATASET VISUALIZATIONS
# =====================================================

print("\nGenerating Solar Dataset Visualizations...")

numeric_columns = solar_df.select_dtypes(
    include="number"
).columns

# =====================================================
# Solar DC Power Distribution
# =====================================================

if "DC_POWER" in solar_df.columns:

    plt.figure(figsize=(9,6))

    plt.hist(
        solar_df["DC_POWER"],
        bins=30,
        edgecolor="black"
    )

    plt.title(
        "Solar DC Power Distribution",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "DC Power",
        fontsize=12
    )

    plt.ylabel(
        "Frequency",
        fontsize=12
    )

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "solar_dc_power_distribution.pdf"
        ),
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# Solar Irradiation vs DC Power
# =====================================================

if (
    "IRRADIATION" in solar_df.columns and
    "DC_POWER" in solar_df.columns
):

    plt.figure(figsize=(9,6))

    plt.scatter(
        solar_df["IRRADIATION"],
        solar_df["DC_POWER"],
        s=10,
        alpha=0.5
    )

    plt.title(
        "Solar Irradiation vs DC Power",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Irradiation",
        fontsize=12
    )

    plt.ylabel(
        "DC Power",
        fontsize=12
    )

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "solar_irradiation_vs_power.pdf"
        ),
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# Solar Correlation Heatmap
# =====================================================

plt.figure(figsize=(10,8))

sns.heatmap(
    solar_df[numeric_columns].corr(),
    cmap="coolwarm",
    annot=False
)

plt.title(
    "Solar Correlation Heatmap",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "solar_heatmap.pdf"
    ),
    format="pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# Daily Solar Generation
# =====================================================

print("Generating Daily Solar Generation Plot...")

if daily_df.shape[1] >= 2:

    plt.figure(figsize=(12,5))

    plt.plot(
        daily_df.iloc[:,0],
        daily_df.iloc[:,1],
        linewidth=2
    )

    plt.xticks(rotation=45)

    plt.title(
        "Daily Solar Generation",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Date",
        fontsize=12
    )

    plt.ylabel(
        "Generation",
        fontsize=12
    )

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "daily_solar_generation.pdf"
        ),
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# WIND DATASET VISUALIZATIONS
# =====================================================

print("\nGenerating Wind Dataset Visualizations...")

# =====================================================
# Wind Speed vs Power Output
# =====================================================

if (
    "Wind Speed (m/s)" in wind_df.columns and
    "LV ActivePower (kW)" in wind_df.columns
):

    plt.figure(figsize=(9,6))

    plt.scatter(
        wind_df["Wind Speed (m/s)"],
        wind_df["LV ActivePower (kW)"],
        s=10,
        alpha=0.5
    )

    plt.title(
        "Wind Speed vs Power Output",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Wind Speed (m/s)",
        fontsize=12
    )

    plt.ylabel(
        "Power Output (kW)",
        fontsize=12
    )

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "wind_speed_vs_power.pdf"
        ),
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# Wind Correlation Heatmap
# =====================================================

wind_numeric = wind_df.select_dtypes(
    include="number"
)

plt.figure(figsize=(10,8))

sns.heatmap(
    wind_numeric.corr(),
    cmap="viridis",
    annot=False
)

plt.title(
    "Wind Correlation Heatmap",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "wind_heatmap.pdf"
    ),
    format="pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# HOUSEHOLD DATASET VISUALIZATIONS
# =====================================================

print("\nGenerating Household Dataset Visualizations...")

# =====================================================
# Global Active Power Trend
# =====================================================

if "global_active_power" in grid_df.columns:

    plt.figure(figsize=(12,5))

    plt.plot(
        grid_df["global_active_power"].head(1000),
        linewidth=2
    )

    plt.title(
        "Global Active Power Trend",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Samples",
        fontsize=12
    )

    plt.ylabel(
        "Global Active Power",
        fontsize=12
    )

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "global_active_power_trend.pdf"
        ),
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# Voltage Distribution
# =====================================================

if "voltage" in grid_df.columns:

    plt.figure(figsize=(9,6))

    plt.hist(
        grid_df["voltage"],
        bins=40,
        edgecolor="black"
    )

    plt.title(
        "Voltage Distribution",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Voltage",
        fontsize=12
    )

    plt.ylabel(
        "Frequency",
        fontsize=12
    )

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "voltage_distribution.pdf"
        ),
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# Household Correlation Heatmap
# =====================================================

grid_numeric = grid_df.select_dtypes(
    include="number"
)

plt.figure(figsize=(10,8))

sns.heatmap(
    grid_numeric.corr(),
    cmap="rocket",
    annot=False
)

plt.title(
    "Household Power Correlation Heatmap",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "household_heatmap.pdf"
    ),
    format="pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# DATASET SUMMARY
# =====================================================

print("\nSaving Dataset Summary...")

summary_file = os.path.join(
    OUTPUT_PATH,
    "dataset_summary.txt"
)

with open(summary_file, "w") as f:

    f.write("=" * 70 + "\n")
    f.write("RENEWABLE ENERGY DATASET SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("SOLAR DATASET\n")
    f.write("-" * 40 + "\n")
    f.write(str(solar_df.describe()))
    f.write("\n\n")

    f.write("WIND DATASET\n")
    f.write("-" * 40 + "\n")
    f.write(str(wind_df.describe()))
    f.write("\n\n")

    f.write("HOUSEHOLD DATASET\n")
    f.write("-" * 40 + "\n")
    f.write(str(grid_df.describe()))

# =====================================================
# COMPLETION MESSAGE
# =====================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Files:")

print("\nSolar Dataset")
print("----------------------------")
print("✓ solar_dc_power_distribution.pdf")
print("✓ solar_irradiation_vs_power.pdf")
print("✓ solar_heatmap.pdf")
print("✓ daily_solar_generation.pdf")

print("\nWind Dataset")
print("----------------------------")
print("✓ wind_speed_vs_power.pdf")
print("✓ wind_heatmap.pdf")

print("\nHousehold Dataset")
print("----------------------------")
print("✓ global_active_power_trend.pdf")
print("✓ voltage_distribution.pdf")
print("✓ household_heatmap.pdf")

print("\nSummary")
print("----------------------------")
print("✓ dataset_summary.txt")

print("\nOutput Folder:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("All EDA visualizations have been saved as high-quality PDF files.")
print("=" * 70)