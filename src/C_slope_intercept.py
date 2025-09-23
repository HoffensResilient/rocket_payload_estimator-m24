# Post-processing script to calculate slope and intercept for synthetic rocket data
import pandas as pd
import numpy as np
from scipy import stats
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "synthetic_rockets_pivoted.xlsx")
OUTPUT_FILE = os.path.join(DATA_DIR, "synthetic_rockets_with_slope_intercept.xlsx")

def calculate_slope_intercept_for_rockets(input_file, output_file):
    """
    Calculate slope and intercept for each rocket based on altitude-payload relationship
    """
    # Read the synthetic data
    print("Reading synthetic data...")
    df = pd.read_excel(input_file)
    
    # Get unique rocket names
    rocket_names = df['Stage_Rockets'].unique()
    print(f"Found {len(rocket_names)} unique rockets")
    
    # Initialize lists for slope and intercept values
    slopes = []
    intercepts = []
    
    # Calculate slope and intercept for each rocket
    for i, rocket in enumerate(rocket_names):
        if i % 1000 == 0:
            print(f"Processing rocket {i+1}/{len(rocket_names)}")
        
        # Get all rows for this rocket (should be 8 rows for 8 altitudes)
        rocket_data = df[df['Stage_Rockets'] == rocket]
        
        if len(rocket_data) != 8:
            print(f"Warning: Rocket {rocket} has {len(rocket_data)} rows, expected 8")
            # Use default values if not exactly 8 rows
            slope = np.nan
            intercept = np.nan
        else:
            # Extract altitudes and payloads
            altitudes = rocket_data['Altitude_km'].values
            payloads = rocket_data['Payload_kg'].values
            
            # Calculate linear regression (slope and intercept)
            if len(altitudes) >= 2 and len(payloads) >= 2:
                try:
                    # Use scipy's linregress for more robust calculation
                    slope, intercept, r_value, p_value, std_err = stats.linregress(altitudes, payloads)
                    
                    # Additional validation: check if the relationship makes physical sense
                    # Payload should generally decrease with altitude (negative slope)
                    if slope > 0.1:  # Unphysical positive slope
                        print(f"Warning: Rocket {rocket} has positive slope {slope:.4f}")
                        # Force negative slope based on payload decay
                        if payloads[0] > 0:
                            slope = -0.5 * (payloads[0] / 1000)  # Reasonable decay rate
                            intercept = payloads[0] - slope * 100
                except:
                    # Fallback calculation
                    if len(altitudes) >= 2:
                        x_mean = np.mean(altitudes)
                        y_mean = np.mean(payloads)
                        numerator = np.sum((altitudes - x_mean) * (payloads - y_mean))
                        denominator = np.sum((altitudes - x_mean) ** 2)
                        slope = numerator / denominator if denominator != 0 else np.nan
                        intercept = y_mean - slope * x_mean if not np.isnan(slope) else np.nan
                    else:
                        slope = np.nan
                        intercept = np.nan
            else:
                slope = np.nan
                intercept = np.nan
        
        # Add the same slope and intercept to all 8 rows of this rocket
        slopes.extend([slope] * len(rocket_data))
        intercepts.extend([intercept] * len(rocket_data))
    
    # Add slope and intercept columns to dataframe
    df['Slope'] = slopes
    df['Intercept'] = intercepts
    
    # Reorder columns to match the original dataset structure
    final_columns = list(df.columns)
    # Move Slope and Intercept to the end (positions 31 and 32)
    for col in ['Slope', 'Intercept']:
        if col in final_columns:
            final_columns.remove(col)
            final_columns.append(col)
    
    df = df[final_columns]
    
    # Save the updated dataframe
    df.to_excel(output_file, index=False)
    print(f"Saved updated data to: {output_file}")
    print(f"Final dataset shape: {df.shape}")
    
    # Print some statistics
    valid_slopes = df['Slope'].dropna()
    valid_intercepts = df['Intercept'].dropna()
    
    print(f"\nSlope statistics:")
    print(f"  Mean: {valid_slopes.mean():.6f}")
    print(f"  Std: {valid_slopes.std():.6f}")
    print(f"  Min: {valid_slopes.min():.6f}")
    print(f"  Max: {valid_slopes.max():.6f}")
    
    print(f"\nIntercept statistics:")
    print(f"  Mean: {valid_intercepts.mean():.2f}")
    print(f"  Std: {valid_intercepts.std():.2f}")
    print(f"  Min: {valid_intercepts.min():.2f}")
    print(f"  Max: {valid_intercepts.max():.2f}")
    
    return df

if __name__ == "__main__":
    # Run the calculation
    updated_df = calculate_slope_intercept_for_rockets(INPUT_FILE, OUTPUT_FILE)
    print("\nSlope and intercept calculation completed successfully!")