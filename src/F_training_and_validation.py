# This is the main code that trains the model and then applies those params to real data and plots for visualisation purposes.
# It first removes the outliers (data that is too far off from mean) to ensure that errors dont skyrocket.
# The inputs for the models have been selected using results from the (feature_importance_analysis.py) file.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import xgboost as xgb
from tabulate import tabulate
import os
import json
from datetime import datetime
import joblib

# Loading paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create results directory to save all output PNGs and CSVs
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create subdirectories for better organization
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")

for directory in [PLOTS_DIR, TABLES_DIR, MODELS_DIR, VALIDATION_DIR]:
    os.makedirs(directory, exist_ok=True)

SYNTHETIC_FILE = os.path.join(DATA_DIR, "synthetic_rockets_pivoted.xlsx")
REAL_DATA_FILE = os.path.join(DATA_DIR, "Altitude_dataset.xlsx")

# Read the data - NEW: No transposing needed
df = pd.read_excel(SYNTHETIC_FILE)

# Set random seed for reproducibility
np.random.seed(42)

# Display the structure of the data
print("Original Data Shape:", df.shape)
print("\nFirst 10 rows of the data:")
print(df.head(10))

# NEW: Data is already in correct format (rockets as rows), no transposing needed
# We have multiple rows per rocket (8 altitudes), so we need to handle this properly
print("\nData Structure - No transposing needed:")
print("Columns:", df.columns.tolist())

# NEW: Separate features and targets - now we predict payload at different altitudes
# Target is Payload_kg, and we have Altitude_km as a feature
target_column = 'Payload_kg'
feature_columns = [col for col in df.columns if col not in ['Stage_Rockets', 'Payload_kg', 'Slope', 'Intercept']]

# For modeling, we'll use the features + altitude to predict payload
X = df[feature_columns].copy()
y = df[[target_column]].copy()  # Single target now

print(f"\nFeature columns: {len(feature_columns)}")
print(f"Target column: {target_column}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# NEW: Identify rockets with and without transfer stage using Transfer Stage columns
transfer_stage_columns = [col for col in X.columns if 'Transfer_Stage' in col]
has_transfer = X[transfer_stage_columns].notna().any(axis=1)

print(f"\nRockets with transfer stage: {has_transfer.sum()}")
print(f"Rockets without transfer stage: {(~has_transfer).sum()}")

# NEW: Split data into with and without transfer stage
X_with_transfer = X[has_transfer]
y_with_transfer = y[has_transfer]
X_without_transfer = X[~has_transfer]
y_without_transfer = y[~has_transfer]

print(f"\nWith transfer stage - X: {X_with_transfer.shape}, y: {y_with_transfer.shape}")
print(f"Without transfer stage - X: {X_without_transfer.shape}, y: {y_without_transfer.shape}")

# Function to preprocess the data - UPDATED for new column names
def preprocess_data(X):
    X_processed = X.copy()
    
    # Fill NaN values with 0 (for missing stage parameters)
    X_processed = X_processed.fillna(0)
    
    # Ensure all columns are numeric
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
    
    X_processed = X_processed.fillna(0)
    return X_processed

# Preprocess both datasets
X_with_transfer_processed = preprocess_data(X_with_transfer)
X_without_transfer_processed = preprocess_data(X_without_transfer)

print("\nProcessed data with transfer stage shape:", X_with_transfer_processed.shape)
print("Processed data without transfer stage shape:", X_without_transfer_processed.shape)

# Function to detect and remove outliers using IQR method - UPDATED for single target
def remove_outliers_iqr(X, y, threshold=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method
    """
    # Combine X and y to ensure we remove the same rows from both
    combined = pd.concat([X, y], axis=1)
    
    # Calculate IQR for the target variable (now single column)
    Q1 = y.iloc[:, 0].quantile(0.25)
    Q3 = y.iloc[:, 0].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Mark outliers
    outliers_mask = (y.iloc[:, 0] < lower_bound) | (y.iloc[:, 0] > upper_bound)
    
    print(f"Number of outliers detected: {outliers_mask.sum()}")
    print(f"Percentage of data removed: {outliers_mask.sum() / len(combined) * 100:.2f}%")
    
    # Remove outliers
    clean_combined = combined[~outliers_mask]
    X_clean = clean_combined[X.columns]
    y_clean = clean_combined[y.columns]
    
    return X_clean, y_clean

# Remove outliers from both datasets
print("\nRemoving outliers from rockets with transfer stage:")
X_with_transfer_clean, y_with_transfer_clean = remove_outliers_iqr(X_with_transfer_processed, y_with_transfer)

print("\nRemoving outliers from rockets without transfer stage:")
X_without_transfer_clean, y_without_transfer_clean = remove_outliers_iqr(X_without_transfer_processed, y_without_transfer)

# UPDATED: Define feature sets for each model - now for payload prediction
feature_sets = {
    "Random Forest": {
        "With Transfer": "all",
        "Without Transfer":"all"
    },
    "XGBoost": {
        "With Transfer": "all",
        "Without Transfer":"all"
    },
    "LightGBM": {
        "With Transfer": "all",
        "Without Transfer": "all"
    }
}

feature_sets_path = os.path.join(DATA_DIR, "feature_sets.json")
with open(feature_sets_path, 'w') as f:
    json.dump(feature_sets, f, indent=4)
print(f"Saved feature sets to {feature_sets_path}")

# Function to save plots with timestamp
def save_plot(fig, name, folder=PLOTS_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")

# Function to save tables as CSV
def save_table(df, name, folder=TABLES_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.csv"
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved table: {filepath}")

# Function to save models
def save_model(model, name, folder=MODELS_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create normalized filename (lowercase, underscores)
    normalized_name = name.lower().replace(' ', '_')
    filename = f"{normalized_name}_{timestamp}.pkl"
    filepath = os.path.join(folder, filename)
    
    # Save the actual model using joblib
    joblib.dump(model, filepath)
    print(f"Saved model: {filepath}")
    
    # Also save a version without timestamp for the app to use
    latest_filename = f"{normalized_name}_latest.pkl"
    latest_filepath = os.path.join(folder, latest_filename)
    joblib.dump(model, latest_filepath)
    print(f"Saved latest model: {latest_filepath}")

# UPDATED: Function to train and evaluate models - now for single target (payload)
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model, feature_set_name, feature_set):
    # Select features if not using all
    if feature_set != "all":
        # Check if all requested features are available
        available_features = [f for f in feature_set if f in X_train.columns]
        missing_features = [f for f in feature_set if f not in X_train.columns]
        
        if missing_features:
            print(f"Warning: Missing features for {model_name} ({feature_set_name}): {missing_features}")
        
        X_train_selected = X_train[available_features]
        X_test_selected = X_test[available_features]
        print(f"Using {len(available_features)} features for {model_name} ({feature_set_name})")
    else:
        X_train_selected = X_train
        X_test_selected = X_test
        print(f"Using all features for {model_name} ({feature_set_name})")
    
    # Train the model - NEW: Single target, no MultiOutputRegressor needed
    model.fit(X_train_selected, y_train.values.ravel())
    
    # Make predictions on training and test data
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)
    
    # Calculate metrics for training data - NEW: Single target
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Calculate metrics for test data - NEW: Single target
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Calculate relative errors (RMSE as percentage of mean)
    test_rmse_percentage = test_rmse / y_test.iloc[:, 0].mean() * 100
    
    # Create results DataFrames - NEW: Simplified for single target
    train_results = pd.DataFrame({
        'Metric': ['R2', 'RMSE', 'MAE'],
        'Value': [train_r2, train_rmse, train_mae]
    })
    
    test_results = pd.DataFrame({
        'Metric': ['R2', 'RMSE', 'MAE', 'RMSE % of Mean'],
        'Value': [test_r2, test_rmse, test_mae, test_rmse_percentage]
    })
    
    # Create plots for training data - NEW: Single plot for payload
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_name} - Predicted vs Actual Payload ({feature_set_name})', fontsize=14)
    
    # Training data plot
    ax1.scatter(y_train, y_train_pred, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=2)
    ax1.set_xlabel('Actual Payload (kg)')
    ax1.set_ylabel('Predicted Payload (kg)')
    ax1.set_title('Training Data')
    ax1.grid(True, alpha=0.3)
    
    # Add metrics to training plot
    ax1.text(0.05, 0.95, f'R²: {train_r2:.3f}\nRMSE: {train_rmse:.1f}\nMAE: {train_mae:.1f}', 
             transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Test data plot
    ax2.scatter(y_test, y_test_pred, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax2.set_xlabel('Actual Payload (kg)')
    ax2.set_ylabel('Predicted Payload (kg)')
    ax2.set_title('Test Data')
    ax2.grid(True, alpha=0.3)
    
    # Add metrics to test plot
    ax2.text(0.05, 0.95, f'R²: {test_r2:.3f}\nRMSE: {test_rmse:.1f}\nMAE: {test_mae:.1f}\nRMSE %: {test_rmse_percentage:.1f}%', 
             transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_plot(fig, f'{model_name}_{feature_set_name}_payload_prediction')
    plt.show()
    
    # Save results to CSV
    save_table(train_results, f'{model_name}_{feature_set_name}_train_results')
    save_table(test_results, f'{model_name}_{feature_set_name}_test_results')
    
    return train_results, test_results, y_test_pred, model

# UPDATED: Initialize models - single target regression
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42)
}

# Function to run complete training and evaluation pipeline
def run_pipeline(X, y, dataset_name):
    print(f"\n{'='*50}")
    print(f"Training on {dataset_name}")
    print(f"{'='*50}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    all_train_results = {}
    all_test_results = {}
    all_predictions = {}
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Get the appropriate feature set
        feature_set = feature_sets[model_name][dataset_name]
        
        train_results, test_results, predictions, trained_model = train_and_evaluate(
            X_train, X_test, y_train, y_test, model_name, model, dataset_name, feature_set
        )
        
        all_train_results[model_name] = train_results
        all_test_results[model_name] = test_results
        all_predictions[model_name] = predictions
        trained_models[model_name] = trained_model
        
        # Save model information
        save_model(trained_model, f'{model_name}_{dataset_name}')
        
        # Print results in tabular format
        print(f"\n{model_name} Training Results for {dataset_name}:")
        print(tabulate(train_results, headers='keys', tablefmt='grid', floatfmt=".4f"))
        
        print(f"\n{model_name} Test Results for {dataset_name}:")
        print(tabulate(test_results, headers='keys', tablefmt='grid', floatfmt=".4f"))
    
    return all_train_results, all_test_results, all_predictions, trained_models, X_test, y_test

# Run pipeline for rockets with transfer stage
if len(X_with_transfer_clean) > 0:
    train_results_with_transfer, test_results_with_transfer, predictions_with_transfer, models_with_transfer, X_test_with, y_test_with = run_pipeline(
        X_with_transfer_clean, y_with_transfer_clean, "With Transfer"
    )

# Run pipeline for rockets without transfer stage
if len(X_without_transfer_clean) > 0:
    train_results_without, test_results_without, predictions_without, models_without, X_test_without, y_test_without = run_pipeline(
        X_without_transfer_clean, y_without_transfer_clean, "Without Transfer"
    )

# Create a summary table of all test results
summary_data = []

if len(X_with_transfer_clean) > 0:
    for model_name, results in test_results_with_transfer.items():
        # Extract values from the test results DataFrame
        r2 = results[results['Metric'] == 'R2']['Value'].values[0]
        rmse = results[results['Metric'] == 'RMSE']['Value'].values[0]
        mae = results[results['Metric'] == 'MAE']['Value'].values[0]
        rmse_pct = results[results['Metric'] == 'RMSE % of Mean']['Value'].values[0]
        
        summary_data.append([
            f"{model_name} (With Transfer)",
            r2,
            rmse,
            mae,
            rmse_pct
        ])

if len(X_without_transfer_clean) > 0:
    for model_name, results in test_results_without.items():
        # Extract values from the test results DataFrame
        r2 = results[results['Metric'] == 'R2']['Value'].values[0]
        rmse = results[results['Metric'] == 'RMSE']['Value'].values[0]
        mae = results[results['Metric'] == 'MAE']['Value'].values[0]
        rmse_pct = results[results['Metric'] == 'RMSE % of Mean']['Value'].values[0]
        
        summary_data.append([
            f"{model_name} (Without Transfer)",
            r2,
            rmse,
            mae,
            rmse_pct
        ])

# Create summary DataFrame
summary_df = pd.DataFrame(summary_data, columns=['Model', 'R2', 'RMSE', 'MAE', 'RMSE % of Mean'])
save_table(summary_df, 'model_comparison_summary')

print("\n" + "="*80)
print("SUMMARY OF ALL MODELS (Test Results)")
print("="*80)
print(tabulate(summary_df, headers='keys', tablefmt='grid', floatfmt=".4f"))

# Print final model parameters
print("\n" + "="*80)
print("MODEL PARAMETERS (Random Forest as example)")
print("="*80)

if len(X_with_transfer_clean) > 0:
    print("\nRandom Forest parameters for rockets with transfer stage:")
    print(models_with_transfer['Random Forest'].get_params())
    
if len(X_without_transfer_clean) > 0:
    print("\nRandom Forest parameters for rockets without transfer stage:")
    print(models_without['Random Forest'].get_params())

# UPDATED: VALIDATION ON REAL ROCKET DATA ---------------------------------------------------------------------------------------------------------

def load_and_preprocess_real_data(file_path):
    """
    Load and preprocess the real rocket data from Excel file - UPDATED for new format
    """
    print(f"\n{'='*80}")
    print("LOADING AND PREPROCESSING REAL ROCKET DATA")
    print(f"{'='*80}")
    
    # Read Excel file - NEW: No transposing needed
    df_real = pd.read_excel(file_path)
    print(f"Real data shape: {df_real.shape}")
    print(f"\nReal data columns: {list(df_real.columns)}")
    
    # NEW: Data is already in correct format, no transposing needed
    # We have multiple rows per rocket (8 altitudes)
    print(f"Number of unique rockets: {df_real['Stage_Rockets'].nunique()}")
    
    # Separate payload (actual) data from features - NEW: Single target
    X_real = df_real.drop(columns=['Stage_Rockets', 'Payload_kg', 'Slope', 'Intercept'], errors='ignore')
    y_real_actual = df_real[['Payload_kg']].copy()
    
    # Handle calculation errors - replace NaN with 0
    calc_error_cols = [col for col in X_real.columns if 'Calculation_Error' in col]
    for col in calc_error_cols:
        X_real[col] = X_real[col].fillna(0)
    
    # NEW: Identify rockets with and without transfer stage
    transfer_stage_cols = [col for col in X_real.columns if 'Transfer_Stage' in col]
    if transfer_stage_cols:
        has_transfer_real = X_real[transfer_stage_cols].notna().any(axis=1)
    else:
        has_transfer_real = pd.Series([False] * len(X_real))
    
    print(f"\nReal rockets with transfer stage: {has_transfer_real.sum()}")
    print(f"Real rockets without transfer stage: {(~has_transfer_real).sum()}")
    
    # Keep rocket names for validation
    rocket_names = df_real['Stage_Rockets']
    
    return X_real, y_real_actual, has_transfer_real, rocket_names

def validate_on_real_data(X_real, y_real_actual, has_transfer_real, trained_models, rocket_names):
    """
    Validate trained models on real rocket data - UPDATED for new format
    """
    print(f"\n{'='*80}")
    print("VALIDATION ON REAL ROCKET DATA")
    print(f"{'='*80}")
    
    # Preprocess real data
    X_real_processed = preprocess_data(X_real)
    
    # DEBUG: Print information about transfer stage detection
    print(f"Total data points: {len(has_transfer_real)}")
    print(f"Data points with transfer stage: {has_transfer_real.sum()}")
    print(f"Data points without transfer stage: {(~has_transfer_real).sum()}")
    
    # Split real data based on transfer stage
    X_real_with_transfer = X_real_processed[has_transfer_real]
    X_real_without_transfer = X_real_processed[~has_transfer_real]
    y_real_with_transfer = y_real_actual[has_transfer_real]
    y_real_without_transfer = y_real_actual[~has_transfer_real]
    
    rocket_names_with = rocket_names[has_transfer_real]
    rocket_names_without = rocket_names[~has_transfer_real]
    
    # DEBUG: Print rocket distribution
    print(f"Unique rockets with transfer stage: {rocket_names_with.nunique()}")
    print(f"Unique rockets without transfer stage: {rocket_names_without.nunique()}")
    
    validation_results = {}
    all_predictions_real = {}
    
    # Check if models for with-transfer rockets exist and were trained
    models_with_transfer_available = 'models_with_transfer' in globals() and len(X_with_transfer_clean) > 0
    models_without_available = 'models_without' in globals() and len(X_without_transfer_clean) > 0
    
    print(f"Models with transfer available: {models_with_transfer_available}")
    print(f"Models without transfer available: {models_without_available}")
    
    # Validate models for rockets with transfer stage
    if len(X_real_with_transfer) > 0 and models_with_transfer_available:
        print(f"\nValidating models on {len(X_real_with_transfer)} data points WITH transfer stage...")
        validation_results['With Transfer'] = validate_model_group(
            X_real_with_transfer, y_real_with_transfer, rocket_names_with,
            models_with_transfer, "With Transfer", X_with_transfer_clean.columns
        )
        all_predictions_real['With Transfer'] = validation_results['With Transfer']['predictions']
    else:
        print(f"\nSkipping validation for data with transfer stage.")
        print(f"Reason: {len(X_real_with_transfer)} data points with transfer stage, models available: {models_with_transfer_available}")
    
    # Validate models for rockets without transfer stage  
    if len(X_real_without_transfer) > 0 and models_without_available:
        print(f"\nValidating models on {len(X_real_without_transfer)} data points WITHOUT transfer stage...")
        validation_results['Without Transfer'] = validate_model_group(
            X_real_without_transfer, y_real_without_transfer, rocket_names_without,
            models_without, "Without Transfer", X_without_transfer_clean.columns
        )
        all_predictions_real['Without Transfer'] = validation_results['Without Transfer']['predictions']
    else:
        print(f"\nSkipping validation for data without transfer stage.")
        print(f"Reason: {len(X_real_without_transfer)} data points without transfer stage, models available: {models_without_available}")
    
    return validation_results, all_predictions_real

def align_features(X_real, X_train_columns):
    """
    Ensure the real data has the same features as the training data
    """
    X_aligned = pd.DataFrame(index=X_real.index)
    
    # Copy available features
    for col in X_train_columns:
        if col in X_real.columns:
            X_aligned[col] = X_real[col]
        else:
            # Fill missing features with 0
            X_aligned[col] = 0
            print(f"Warning: Feature {col} not found in real data, filling with 0")
    
    # Remove extra features not in training data
    extra_cols = [col for col in X_real.columns if col not in X_train_columns]
    if extra_cols:
        print(f"Warning: Removing extra features from real data: {extra_cols}")
    
    return X_aligned

# UPDATED: Validate model group for single target
def validate_model_group(X_real, y_real, rocket_names, trained_models, group_name, X_train_columns):
    """
    Validate a group of models on real data with aligned features - UPDATED for single target
    """
    predictions_dict = {}
    metrics_dict = {}
    
    # Align features with training data
    X_real_aligned = align_features(X_real, X_train_columns)
    
    for model_name, trained_model in trained_models.items():
        print(f"\n  Validating {model_name} for {group_name}...")
        
        # Get feature set used during training
        feature_set = feature_sets[model_name][group_name]
        
        # Select appropriate features
        if feature_set != "all":
            available_features = [f for f in feature_set if f in X_real_aligned.columns]
            missing_features = [f for f in feature_set if f not in X_real_aligned.columns]
            
            if missing_features:
                print(f"    Warning: Missing features: {missing_features}")
            
            X_real_selected = X_real_aligned[available_features]
        else:
            X_real_selected = X_real_aligned
        
        # Make predictions
        try:
            predictions = trained_model.predict(X_real_selected)
            predictions_dict[model_name] = predictions
            
            # Calculate metrics for available data (ignore NaN values)
            model_metrics = {}
            
            # Get actual and predicted values, removing NaN entries
            actual = y_real.iloc[:, 0].values
            pred = predictions
            
            # Create mask for non-NaN values
            valid_mask = ~np.isnan(actual) & ~np.isnan(pred)
            
            if valid_mask.sum() > 0:  # If we have valid data points
                actual_valid = actual[valid_mask]
                pred_valid = pred[valid_mask]
                
                # Calculate metrics
                r2 = r2_score(actual_valid, pred_valid) if len(actual_valid) > 1 else np.nan
                rmse = np.sqrt(mean_squared_error(actual_valid, pred_valid))
                mae = mean_absolute_error(actual_valid, pred_valid)
                mape = np.mean(np.abs((actual_valid - pred_valid) / actual_valid)) * 100 if np.all(actual_valid != 0) else np.nan
                
                model_metrics = {
                    'R2': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'n_samples': len(actual_valid)
                }
            else:
                model_metrics = {
                    'R2': np.nan,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'MAPE': np.nan,
                    'n_samples': 0
                }
            
            metrics_dict[model_name] = model_metrics
            
            print(f"    Validation metrics - R²: {r2:.3f}, RMSE: {rmse:.1f}, MAE: {mae:.1f}")
            
        except Exception as e:
            print(f"    Error predicting with {model_name}: {e}")
            predictions_dict[model_name] = None
            metrics_dict[model_name] = None
    
    return {
        'predictions': predictions_dict,
        'metrics': metrics_dict,
        'rocket_names': rocket_names,
        'actual_values': y_real
    }

# UPDATED: Create validation summary for single target
def create_validation_summary_tables(validation_results):
    """
    Create comprehensive summary tables for validation results - UPDATED for single target
    """
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    all_summary_data = []
    
    for group_name, group_results in validation_results.items():
        if group_results and 'metrics' in group_results:
            print(f"\n{group_name.upper()} DATA:")
            print("-" * 50)
            
            for model_name, model_metrics in group_results['metrics'].items():
                if model_metrics:
                    if model_metrics['n_samples'] > 0:
                        summary_row = [
                            f"{model_name} ({group_name})",
                            model_metrics['R2'],
                            model_metrics['RMSE'],
                            model_metrics['MAE'],
                            model_metrics['MAPE'],
                            model_metrics['n_samples']
                        ]
                        
                        all_summary_data.append(summary_row)
                        
                        print(f"\n{model_name} Results:")
                        print(f"  R²: {model_metrics['R2']:.3f}")
                        print(f"  RMSE: {model_metrics['RMSE']:.1f}")
                        print(f"  MAE: {model_metrics['MAE']:.1f}")
                        print(f"  MAPE: {model_metrics['MAPE']:.1f}%")
                        print(f"  Samples: {model_metrics['n_samples']}")
                    
                    # Save validation results
                    summary_df = pd.DataFrame([summary_row], 
                                            columns=['Model', 'R²', 'RMSE', 'MAE', 'MAPE%', 'N'])
                    save_table(summary_df, f'{model_name}_{group_name}_validation', VALIDATION_DIR)
    
    # Create overall summary table
    if all_summary_data:
        overall_summary_df = pd.DataFrame(all_summary_data, 
                                        columns=['Model', 'R²', 'RMSE', 'MAE', 'MAPE%', 'N'])
        print(f"\n{'='*80}")
        print("OVERALL VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(tabulate(overall_summary_df, headers='keys', tablefmt='grid', floatfmt=".4f"))
        
        # Save overall validation summary
        save_table(overall_summary_df, 'overall_validation_summary', VALIDATION_DIR)
        
        return overall_summary_df
    
    return None

# UPDATED: Create validation comparison plots for single target
def create_validation_comparison_plots(validation_results):
    """
    Create comparison plots for validation results - UPDATED for single target
    """
    print(f"\n{'='*80}")
    print("CREATING VALIDATION COMPARISON PLOTS")
    print(f"{'='*80}")

    for group_name, group_results in validation_results.items():
        if not group_results or 'predictions' not in group_results:
            print(f"Skipping group {group_name} — no predictions found.")
            continue

        rocket_names = group_results['rocket_names']
        actual_values = group_results['actual_values']
        predictions = group_results['predictions']

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get valid data
        actual = actual_values.iloc[:, 0].values.astype(float)
        valid_mask = ~np.isnan(actual)

        if valid_mask.sum() > 0:
            actual_valid = actual[valid_mask]
            rocket_names_valid = rocket_names[valid_mask].reset_index(drop=True)

            # Plot predictions from each model
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for j, (model_name, pred) in enumerate(predictions.items()):
                if pred is None:
                    continue
                pred = np.asarray(pred)
                if pred.shape[0] == len(actual):
                    pred_valid = pred[valid_mask]
                    ax.scatter(actual_valid, pred_valid, alpha=0.7, label=model_name, color=colors[j % len(colors)])

            # Perfect prediction line
            min_val, max_val = actual_valid.min(), actual_valid.max()
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
            ax.set_xlabel('Actual Payload (kg)')
            ax.set_ylabel('Predicted Payload (kg)')
            ax.set_title(f'Validation Results - {group_name} Data: Predicted vs Actual Payload')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Annotations for rocket names (if small dataset)
            if len(actual_valid) <= 20:
                for idx, rocket_label in enumerate(rocket_names_valid):
                    x = actual_valid[idx]
                    y = list(predictions.values())[0][valid_mask][idx] if predictions else x
                    ax.annotate(rocket_label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

        plt.tight_layout()
        save_plot(fig, f'{group_name}_validation_comparison', VALIDATION_DIR)
        plt.show()

# UPDATED: Create detailed rocket comparison table for single target
def create_detailed_rocket_comparison_table(validation_results):
    """
    Create detailed table comparing predicted vs actual values - UPDATED for single target
    """
    print(f"\n{'='*80}")
    print("DETAILED ROCKET-BY-ROCKET COMPARISON")
    print(f"{'='*80}")
    
    for group_name, group_results in validation_results.items():
        if not group_results or 'predictions' not in group_results:
            continue
            
        print(f"\n{group_name.upper()} DATA:")
        print("-" * 70)
        
        rocket_names = group_results['rocket_names']
        actual_values = group_results['actual_values']
        predictions = group_results['predictions']
        
        # Create comparison table
        comparison_data = []
        
        for i, rocket_name in enumerate(rocket_names):
            actual_val = actual_values.iloc[i, 0]
            
            if not np.isnan(actual_val):
                row_data = [rocket_name, f"{actual_val:.1f}"]
                
                # Add predictions from each model
                for model_name, pred in predictions.items():
                    if pred is not None:
                        pred_val = pred[i]
                        error = ((pred_val - actual_val) / actual_val * 100) if actual_val != 0 else np.nan
                        row_data.extend([f"{pred_val:.1f}", f"{error:+.1f}%"])
                    else:
                        row_data.extend(["N/A", "N/A"])
                
                comparison_data.append(row_data)
        
        if comparison_data:
            # Create column headers
            headers = ['Rocket', 'Actual Payload (kg)']
            for model_name in predictions.keys():
                headers.extend([f'{model_name} Pred', f'{model_name} Error%'])
            
            comparison_df = pd.DataFrame(comparison_data, columns=headers)
            print(tabulate(comparison_df, headers='keys', tablefmt='grid', floatfmt=".1f"))
            
            # Save detailed comparison
            save_table(comparison_df, f'{group_name}_detailed_comparison', VALIDATION_DIR)

# Execute validation pipeline
if __name__ == "__main__":
    try:
        # Load real rocket data - UPDATED for new format
        X_real, y_real_actual, has_transfer_real, rocket_names = load_and_preprocess_real_data(REAL_DATA_FILE)
        
        # Validate models on real data
        validation_results, all_predictions_real = validate_on_real_data(
            X_real, y_real_actual, has_transfer_real, 
            {'models_with_transfer': models_with_transfer if 'models_with_transfer' in locals() else {},
             'models_without': models_without if 'models_without' in locals() else {}},
            rocket_names
        )
        
        # Create summary tables
        overall_summary = create_validation_summary_tables(validation_results)
        
        # Create comparison plots
        create_validation_comparison_plots(validation_results)
        
        # Create detailed comparison tables
        create_detailed_rocket_comparison_table(validation_results)
        
        print(f"\n{'='*80}")
        print("VALIDATION ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        # Final insights
        if overall_summary is not None and not overall_summary.empty:
            print(f"\nKEY INSIGHTS:")
            print("-" * 40)
            
            # Find best performing models
            best_r2_idx = overall_summary['R²'].idxmax()
            best_rmse_idx = overall_summary['RMSE'].idxmin()
            
            best_r2 = overall_summary.loc[best_r2_idx]
            best_rmse = overall_summary.loc[best_rmse_idx]
            
            print(f"Best R²: {best_r2['Model']} (R² = {best_r2['R²']:.3f})")
            print(f"Best RMSE: {best_rmse['Model']} (RMSE = {best_rmse['RMSE']:.1f})")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {REAL_DATA_FILE}")
        print("Please ensure the Altitude_dataset.xlsx file is in the data folder.")
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()

print("\nAnalysis completed successfully!")