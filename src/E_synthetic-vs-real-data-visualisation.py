# Helps visualise the data to serve as a reality check ensuring the data makes sense. 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # assumes this script is in src/
DATA_DIR = os.path.join(BASE_DIR, "data")

REAL_FILE = os.path.join(DATA_DIR, "Altitude_dataset.xlsx")
SYNTHETIC_FILE = os.path.join(DATA_DIR, "synthetic_rockets_pivoted.xlsx")

# Loading datasets - no need to set index since format is different
real_df = pd.read_excel(REAL_FILE)
synthetic_df = pd.read_excel(SYNTHETIC_FILE)

# Since we have multiple rows per rocket, we need to aggregate or sample
# Let's take one row per rocket (e.g., at 100km altitude) for stage parameter comparison
real_rockets_100km = real_df[real_df['Altitude_km'] == 100].set_index('Stage_Rockets')
synthetic_rockets_100km = synthetic_df[synthetic_df['Altitude_km'] == 100].set_index('Stage_Rockets')

# Key comparison parameters - mapped to new column names
key_params = [
    "1st_Stage_Average_Isp_s",
    "1st_Stage_Delta_v_m_per_s", 
    "1st_Stage_Start_Mass_kg",
    "1st_Stage_Final_Mass_kg",
    "2nd_Stage_Delta_v_m_per_s",
    "Payload_kg"  # Using payload at 100km as comparison point
]

# Get real data for these parameters
real_selected = real_rockets_100km[key_params].T
real_selected.index.name = "Parameter"

# Randomly select 100 synthetic rockets to avoid overcrowding the graphs
synthetic_sampled_cols = np.random.choice(synthetic_rockets_100km.index.unique(), size=min(100, len(synthetic_rockets_100km.index.unique())), replace=False)
synthetic_selected = synthetic_rockets_100km.loc[synthetic_sampled_cols, key_params].T
synthetic_selected.index.name = "Parameter"

# Side by side comparison
for param in key_params:
    plt.figure(figsize=(10,6))
    
    # Real data
    real_values = real_selected.loc[param].dropna()
    if len(real_values) > 0:
        plt.plot(range(len(real_values)), real_values.values, marker='o', label="Real", alpha=0.7)
    
    # Synthetic data  
    synthetic_values = synthetic_selected.loc[param].dropna()
    if len(synthetic_values) > 0:
        plt.plot(range(len(synthetic_values)), synthetic_values.values, marker='x', linestyle='', alpha=0.7, label="Synthetic (sampled)")
    
    plt.xticks(rotation=45)
    plt.title(f"{param}")
    plt.ylabel(param)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Scatter matrix to evaluate relationships
real_for_scatter = real_rockets_100km[key_params].dropna()
synthetic_for_scatter = synthetic_rockets_100km.loc[synthetic_sampled_cols, key_params].dropna()

real_for_scatter["Dataset"] = "Real"
synthetic_for_scatter["Dataset"] = "Synthetic"

combined = pd.concat([real_for_scatter, synthetic_for_scatter])

# Plot scatter matrix
fig = plt.figure(figsize=(12, 12))
pd.plotting.scatter_matrix(combined.drop(columns="Dataset"), diagonal='kde', alpha=0.7, figsize=(12,12))
plt.suptitle("Scatter Matrix of Key Rocket Parameters (100km altitude)", y=0.95)
plt.tight_layout()
plt.show()

# Additional comparison: Payload vs Altitude relationship for a few sample rockets
plt.figure(figsize=(12, 8))

# Sample a few real rockets
real_rocket_samples = real_df['Stage_Rockets'].unique()[:5]
for rocket in real_rocket_samples:
    rocket_data = real_df[real_df['Stage_Rockets'] == rocket]
    if len(rocket_data) == 8:  # Ensure we have all altitude points
        plt.plot(rocket_data['Altitude_km'], rocket_data['Payload_kg'], 
                marker='o', linewidth=2, label=f"Real: {rocket}")

# Sample a few synthetic rockets
synthetic_rocket_samples = synthetic_df['Stage_Rockets'].unique()[:5]
for rocket in synthetic_rocket_samples:
    rocket_data = synthetic_df[synthetic_df['Stage_Rockets'] == rocket]
    if len(rocket_data) == 8:
        plt.plot(rocket_data['Altitude_km'], rocket_data['Payload_kg'], 
                marker='x', linestyle='--', linewidth=2, label=f"Synthetic: {rocket}")

plt.xlabel('Altitude (km)')
plt.ylabel('Payload (kg)')
plt.title('Payload vs Altitude Comparison (Sample Rockets)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Statistical comparison
print("Statistical Comparison of Key Parameters (at 100km altitude):")
print("="*60)
for param in key_params:
    real_vals = real_rockets_100km[param].dropna()
    synthetic_vals = synthetic_rockets_100km[param].dropna()
    
    if len(real_vals) > 0 and len(synthetic_vals) > 0:
        print(f"\n{param}:")
        print(f"  Real:      mean={real_vals.mean():.2f}, std={real_vals.std():.2f}, n={len(real_vals)}")
        print(f"  Synthetic: mean={synthetic_vals.mean():.2f}, std={synthetic_vals.std():.2f}, n={len(synthetic_vals)}")
        print(f"  Difference: {((synthetic_vals.mean() - real_vals.mean()) / real_vals.mean() * 100):.1f}%")

# Distribution comparison using histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, param in enumerate(key_params[:6]):  # Plot first 6 parameters
    real_vals = real_rockets_100km[param].dropna()
    synthetic_vals = synthetic_rockets_100km[param].dropna()
    
    if len(real_vals) > 0 and len(synthetic_vals) > 0:
        axes[i].hist(real_vals, alpha=0.7, label='Real', bins=20, density=True)
        axes[i].hist(synthetic_vals, alpha=0.7, label='Synthetic', bins=20, density=True)
        axes[i].set_title(param)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Distribution Comparison: Real vs Synthetic Data', y=1.02)
plt.show()