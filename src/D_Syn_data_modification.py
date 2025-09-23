# Real_data_modification.py
import pandas as pd
import numpy as np
import os

def process_altitude_data():
    # Define file paths using os.path
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, '..', 'data')
    input_file = os.path.join(data_dir, 'synthetic_rockets_with_slope_intercept.xlsx')
    output_file = os.path.join(data_dir, 'Extended_syn_dataset.xlsx')
    parquet_file = os.path.join(data_dir, 'Altitude_dataset.parquet')
    output_parquet = os.path.join(data_dir, 'Extended_syn_dataset.parquet')
    
    # Read the original Excel file
    df = pd.read_excel(input_file, header=[0])
    
    # Save as parquet
    df.to_parquet(parquet_file)
    
    # Read from parquet
    df = pd.read_parquet(parquet_file)
    
    # Find the correct column names
    slope_col = None
    intercept_col = None
    alt_col = None
    payload_col = None
    
    for col in df.columns:
        if 'Slope' in str(col):
            slope_col = col
        elif 'Intercept' in str(col):
            intercept_col = col
        elif 'Altitude' in str(col):
            alt_col = col
        elif 'Payload' in str(col):
            payload_col = col
    
    print(f"Found columns:")
    print(f"Slope: {slope_col}")
    print(f"Intercept: {intercept_col}")
    print(f"Altitude: {alt_col}")
    print(f"Payload: {payload_col}")
    
    # Group by all columns except altitude, payload, slope and intercept
    rocket_col = df.columns[0]  # First column should be the rocket name
    
    extended_dfs = []
    
    for rocket_name, group in df.groupby(rocket_col):
        if len(group) == 0:
            continue
            
        # Get slope and intercept from first row
        slope = group[slope_col].iloc[0]
        intercept = group[intercept_col].iloc[0]
        
        # Generate 100 altitude points
        altitudes = np.linspace(100, 800, 100)
        payloads = slope * altitudes + intercept
        
        # Create new rows
        new_rows = []
        for alt, payload in zip(altitudes, payloads):
            new_row = group.iloc[0].copy()
            new_row[alt_col] = alt
            new_row[payload_col] = payload
            new_rows.append(new_row)
        
        extended_dfs.extend(new_rows)
    
    # Create extended DataFrame
    extended_df = pd.DataFrame(extended_dfs)
    
    # Remove slope and intercept columns
    extended_df = extended_df.drop(columns=[slope_col, intercept_col])
    
    # Save results - first save as parquet
    extended_df.to_parquet(output_parquet)
    
    # For Excel, we need to handle MultiIndex columns differently
    # Option 1: Flatten the column names
    extended_df_excel = extended_df.copy()
    extended_df_excel.columns = [' '.join(col).strip() for col in extended_df_excel.columns]
    
    # Option 2: Save with index=True to work around the MultiIndex issue
    extended_df_excel.to_excel(output_file, index=False)
    
    print(f"\nProcessing complete. Output files created:")
    print(f" - {output_parquet}")
    print(f" - {output_file}")
    print(f"Original data shape: {df.shape}")
    print(f"Extended data shape: {extended_df.shape}")

if __name__ == '__main__':
    process_altitude_data()