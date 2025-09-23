# This script helps generate sensible synthetic data which is used to train the model.

import os
import math
import numpy as np
import pandas as pd
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")

# Globalising all paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_PATH = os.path.join(DATA_DIR, "Altitude_dataset.xlsx")
OUTPUT_XLSX = os.path.join(DATA_DIR, "synthetic_rockets_pivoted.xlsx")
REPORT_JSON = os.path.join(DATA_DIR, "synthetic_generation_report.json")

RANDOM_SEED = 42
N_SYNTH = 6000
G0 = 9.80665
MIN_LIFTOFF_TW = 1.2

np.random.seed(RANDOM_SEED)

def safe_array(x):
    a = np.array(x, dtype=float)
    a = a[np.isfinite(a)]
    return a

def sample_empirical(vals, size=1, default=None, bounds=None):
    vals = safe_array(vals)
    if vals.size >= 5:
        choices = np.random.choice(vals, size=size, replace=True)
        jitter = np.random.normal(0, max(vals.std() * 0.05, 1e-6), size=size)
        out = choices + jitter
        if bounds is not None:
            out = np.clip(out, bounds[0], bounds[1])
        return out if size > 1 else float(out[0])
    elif vals.size > 0:
        choices = np.random.choice(vals, size=size, replace=True)
        jitter = np.random.normal(0, max(vals.std() * 0.05, 1e-6), size=size)
        out = choices + jitter
        if bounds is not None:
            out = np.clip(out, bounds[0], bounds[1])
        return out if size > 1 else float(out[0])
    else:
        if size == 1:
            if default is None:
                raise ValueError("No empirical values and no default provided")
            return float(default)
        else:
            if default is None:
                raise ValueError("No empirical values and no default provided")
            return np.full(size, float(default))

def fit_beta_from_samples(samples):
    s = safe_array(samples)
    s = s[(s > 0) & (s < 1)]
    if s.size < 4:
        return None
    mean = s.mean()
    var = s.var(ddof=1)
    if var <= 0 or mean <= 0 or mean >= 1:
        return None
    tmp = mean * (1 - mean) / var - 1
    a = max(0.5, mean * tmp)
    b = max(0.5, (1 - mean) * tmp)
    return float(a), float(b)

def estimate_isp_from_technology_level(stage_mass, stage_role, empirical_isp_samples=None):
    """Estimate Isp based on stage mass and technology level with empirical grounding"""
    
    # First, try to use empirical data if available and relevant
    if empirical_isp_samples is not None and len(empirical_isp_samples) > 0:
        # Weight the empirical sample based on stage mass similarity
        empirical_isp = float(sample_empirical(empirical_isp_samples, default=300.0))
    else:
        empirical_isp = 300.0
    
    # Base Isp by mass category (correlated with propellant type and engine complexity)
    if stage_mass > 500000:  # Super heavy: advanced methalox or kerolox
        base_isp = np.random.normal(335, 8)   # Raptor-like engines
        tech_variation = np.random.normal(0, 4)
    elif stage_mass > 100000:  # Heavy: modern kerolox/methalox
        base_isp = np.random.normal(320, 10)  # Merlin-like engines
        tech_variation = np.random.normal(0, 6)
    elif stage_mass > 50000:   # Medium: variety of propellants
        base_isp = np.random.normal(310, 12)  # RD-180-like engines
        tech_variation = np.random.normal(0, 8)
    elif stage_mass > 10000:   # Small-medium: simpler systems
        base_isp = np.random.normal(300, 15)  # Smaller liquid engines
        tech_variation = np.random.normal(0, 10)
    else:   # Small: often solids or simple liquids
        base_isp = np.random.normal(285, 20)  # Small launch vehicles
        tech_variation = np.random.normal(0, 12)
    
    # Adjust for stage role (upper stages have higher Isp due to vacuum optimization)
    role_adjustment = {
        '1st_stage': -15,    # Sea-level optimized, lower Isp
        '2nd_stage': +10,    # Partially vacuum optimized
        'transfer_stage': +25  # Fully vacuum optimized
    }
    
    role_bonus = role_adjustment.get(stage_role, 0)
    
    # Blend empirical data with physics-based model (70% empirical weight if available)
    if empirical_isp_samples is not None and len(empirical_isp_samples) > 5:
        # More weight to empirical for stages similar to real data
        blend_ratio = 0.7 if len(empirical_isp_samples) > 10 else 0.5
        final_isp = (blend_ratio * empirical_isp + 
                    (1 - blend_ratio) * (base_isp + tech_variation + role_bonus))
    else:
        final_isp = base_isp + tech_variation + role_bonus
    
    # Physical limits for chemical rockets
    return np.clip(final_isp, 250, 460)

# Read the new dataset format
real_df = pd.read_excel(INPUT_PATH)

# Since data is already in long format, we can work directly with it
# Get unique rockets (each rocket appears 8 times for different altitudes)
unique_rockets = real_df['Stage_Rockets'].unique()

# Extract empirical distributions from the new column structure
# Build dict: stage -> {parameter: values}
dists = {}

# Stage parameter mappings for the new column names
stage_params_mapping = {
    '1st_Stage': ['Average_Isp_s', 'Delta_v_m_per_s', 'Dry_Mass_kg', 'Engine_Run_Time_s', 
                  'Final_Mass_kg', 'Max_Acceleration_m_per_s_squared', 'Propellant_Mass_kg',
                  'Start_Mass_kg', 'Total_Impulse_s', 'Total_Thrust_N'],
    '2nd_Stage': ['Calculation_Error_m_per_s', 'Delta_v_m_per_s', 'Dry_Mass_kg', 'Engine_Run_Time_s',
                  'Final_Mass_kg', 'Max_Acceleration_m_per_s_squared', 'Propellant_Mass_kg',
                  'Start_Mass_kg', 'Total_Impulse_s', 'Total_Thrust_N'],
    'Transfer_Stage': ['Calculation_Error_m_per_s', 'Delta_v_m_per_s', 'Dry_Mass_kg', 'Final_Mass_kg',
                       'Max_Acceleration_m_per_s_squared', 'Propellant_Mass_kg', 'Start_Mass_kg',
                       'Total_Impulse_s']
}

# Extract empirical values for each stage parameter
for stage_prefix, parameters in stage_params_mapping.items():
    stage_data = {}
    for param in parameters:
        col_name = f"{stage_prefix}_{param}"
        if col_name in real_df.columns:
            values = safe_array(real_df[col_name].values)
            # Use simple parameter name as key
            if 'Isp' in param:
                stage_data['isp'] = values.tolist()
            elif 'Delta_v' in param:
                stage_data['dv'] = values.tolist()
            elif 'Dry_Mass' in param:
                stage_data['dry_mass'] = values.tolist()
            elif 'Start_Mass' in param:
                stage_data['start_mass'] = values.tolist()
            elif 'Propellant_Mass' in param:
                stage_data['propellant'] = values.tolist()
            elif 'Thrust' in param:
                stage_data['thrust'] = values.tolist()
            elif 'Impulse' in param:
                stage_data['impulse'] = values.tolist()
            elif 'Engine_Run_Time' in param:
                stage_data['burn_time'] = values.tolist()
    
    # Calculate structural fractions if we have dry and start mass
    if 'dry_mass' in stage_data and 'start_mass' in stage_data:
        dry = np.array(stage_data['dry_mass'], dtype=float)
        start = np.array(stage_data['start_mass'], dtype=float)
        s_vals = []
        if dry.size == start.size and dry.size > 0:
            for a, b in zip(dry, start):
                if np.isfinite(a) and np.isfinite(b) and b > 0:
                    r = a / b
                    if 0 < r < 0.5:
                        s_vals.append(r)
        stage_data['s_vals'] = s_vals
    
    dists[stage_prefix.replace('_', ' ')] = stage_data

# Extract payload samples at 100km altitude (baseline)
payload_100km_samples = []
for rocket in unique_rockets:
    rocket_data = real_df[real_df['Stage_Rockets'] == rocket]
    if len(rocket_data) > 0:
        payload_100km = rocket_data[rocket_data['Altitude_km'] == 100]['Payload_kg'].values
        if len(payload_100km) > 0 and np.isfinite(payload_100km[0]) and payload_100km[0] > 0:
            payload_100km_samples.append(payload_100km[0])

# Calculate baseline total delta-v from empirical data
per_rocket_sum_dv = []
for rocket in unique_rockets:
    rocket_data = real_df[real_df['Stage_Rockets'] == rocket].iloc[0]
    dv_sum = 0.0
    found_dv = False
    for stage in ['1st_Stage', '2nd_Stage', 'Transfer_Stage']:
        dv_col = f"{stage}_Delta_v_m_per_s"
        if dv_col in rocket_data and pd.notna(rocket_data[dv_col]):
            dv_sum += rocket_data[dv_col]
            found_dv = True
    if found_dv:
        per_rocket_sum_dv.append(dv_sum)

if len(per_rocket_sum_dv) >= 3:
    baseline_total_dv = float(np.median(per_rocket_sum_dv))
else:
    baseline_total_dv = 9400.0

# Define the generator for one rocket with updated Isp model
def generate_one_rocket(dists, payload_100km_samples, baseline_total_dv):
    """
    Returns a dict with flattened keys for the new column structure
    """
    payload_100km = float(sample_empirical(payload_100km_samples, default=500.0))
    
    # Determine stages to generate
    canonical = []
    for label in ['Transfer Stage', '2nd Stage', '1st Stage']:
        if label in dists:
            canonical.append(label)
    
    n_possible = len(canonical)
    if n_possible == 0:
        canonical = ['1st Stage', '2nd Stage']
        n_possible = 2
    
    if n_possible >= 3:
        n_stages = 3 if np.random.rand() > 0.6 else 2
    else:
        n_stages = 2
    
    chosen_stages = canonical[:n_stages][::-1]
    upper_mass = 0.0
    stage_results = []
    
    for st_label in chosen_stages:
        sd = dists.get(st_label, {})
        
        # Get empirical ISP samples for this stage type
        empirical_isp_samples = sd.get('isp', [])
        
        # First estimate stage mass roughly from payload to determine Isp
        # We'll do an initial estimate, then refine after Isp is set
        estimated_stage_mass = payload_100km * (10 if st_label == '1st Stage' else 3)
        stage_role = '1st_stage' if '1st' in st_label else '2nd_stage' if '2nd' in st_label else 'transfer_stage'
        
        # Use the new Isp model that depends on mass and role
        isp = estimate_isp_from_technology_level(estimated_stage_mass, stage_role, empirical_isp_samples)
        
        # Sample other parameters
        dv = float(sample_empirical(sd.get('dv', []), default=2000.0, bounds=(100.0, 12000.0)))
        
        # Structural fraction
        s_vals = sd.get('s_vals', [])
        if len(s_vals) >= 3:
            beta = fit_beta_from_samples(s_vals)
            if beta:
                a, b = beta
                s = float(stats.beta(a, b).rvs())
                s = float(np.clip(s, 0.02, 0.25))
            else:
                s = float(np.clip(np.random.choice(s_vals) * (1 + np.random.normal(0, 0.05)), 0.02, 0.25))
        elif len(s_vals) >= 1:
            s = float(np.clip(np.random.choice(s_vals) * (1 + np.random.normal(0, 0.05)), 0.02, 0.25))
        else:
            s = float(np.random.beta(2.0, 20.0))
        
        # Mass ratio and mass calculations
        R = math.exp(dv / (isp * G0))
        attempts = 0
        while R * s >= 0.98 and attempts < 20:
            if np.random.rand() < 0.5:
                isp = min(460.0, isp * 1.05)
            else:
                s = max(0.02, s * 0.9)
            R = math.exp(dv / (isp * G0))
            attempts += 1
        if R * s >= 0.98:
            dv = max(100.0, dv * 0.7)
            R = math.exp(dv / (isp * G0))
        
        denom = 1.0 - R * s
        if denom <= 0:
            return None
        
        m0 = R * (upper_mass + payload_100km) / denom
        mf = m0 / R
        dry_mass = s * m0
        prop_mass = m0 - dry_mass - upper_mass - payload_100km
        
        if prop_mass <= 0 or m0 <= 0 or dry_mass <= 0:
            return None
        
        # Now that we have actual mass, we can refine Isp
        refined_isp = estimate_isp_from_technology_level(m0, stage_role, empirical_isp_samples)
        
        # Adjust delta-v based on refined Isp if significant difference
        if abs(refined_isp - isp) > 10:
            isp = refined_isp
            R = math.exp(dv / (isp * G0))
            # Recalculate masses with new R
            m0 = R * (upper_mass + payload_100km) / (1 - R * s)
            mf = m0 / R
            dry_mass = s * m0
            prop_mass = m0 - dry_mass - upper_mass - payload_100km
        
        # Thrust & impulse
        thrust = float(sample_empirical(sd.get('thrust', []), default=max(1e5, m0 * G0 * MIN_LIFTOFF_TW)))
        impulse = float(sample_empirical(sd.get('impulse', []), default=thrust * 120.0))
        burn_time = float(impulse / thrust) if thrust > 0 else float(np.random.uniform(10, 300))
        
        # Calculate acceleration
        max_acc = thrust / max(m0, 1e-6)
        
        stage_info = {
            'stage_label': st_label,
            'start_mass_kg': m0,
            'final_mass_kg': mf,
            'dry_mass_kg': dry_mass,
            'propellant_mass_kg': prop_mass,
            'delta_v_m_s': dv,
            'isp_s': isp,
            'mass_ratio': R,
            'structural_fraction': s,
            'total_thrust_n': thrust,
            'total_impulse': impulse,
            'engine_run_time_s': burn_time,
            'max_acceleration_m_s2': max_acc
        }
        stage_results.append(stage_info)
        upper_mass = m0

    # Liftoff TW adjustment
    bottom = stage_results[-1]
    total_initial_mass = bottom['start_mass_kg']
    total_thrust = bottom['total_thrust_n']
    liftoff_tw = total_thrust / (total_initial_mass * G0)
    
    if liftoff_tw < MIN_LIFTOFF_TW:
        scale = (MIN_LIFTOFF_TW * total_initial_mass * G0) / (total_thrust + 1e-9)
        bottom['total_thrust_n'] *= scale
        bottom['total_impulse'] *= scale
        bottom['engine_run_time_s'] = bottom['total_impulse'] / bottom['total_thrust_n'] if bottom['total_thrust_n'] > 0 else bottom['engine_run_time_s']
        liftoff_tw = MIN_LIFTOFF_TW
        stage_results[-1] = bottom

    sum_dv = sum(s['delta_v_m_s'] for s in stage_results)
    
    # Adjust payload by delta-v ratio
    dv_ratio = min(1.0, sum_dv / max(1.0, baseline_total_dv))
    payload_100km_adj = float(max(0.0, payload_100km * dv_ratio))
    
    # Flatten output for new column structure
    flat = {}
    
    # Map stage results to new column names
    for idx, s in enumerate(reversed(stage_results), start=1):
        stage_prefix = f"{'1st' if idx == 1 else '2nd' if idx == 2 else 'Transfer'}_Stage"
        
        flat[f'{stage_prefix}_Average_Isp_s'] = s['isp_s']
        flat[f'{stage_prefix}_Delta_v_m_per_s'] = s['delta_v_m_s']
        flat[f'{stage_prefix}_Dry_Mass_kg'] = s['dry_mass_kg']
        if idx <= 2:  # Only 1st and 2nd stages have engine run time in the dataset
            flat[f'{stage_prefix}_Engine_Run_Time_s'] = s['engine_run_time_s']
        flat[f'{stage_prefix}_Final_Mass_kg'] = s['final_mass_kg']
        flat[f'{stage_prefix}_Max_Acceleration_m_per_s_squared'] = s['max_acceleration_m_s2']
        flat[f'{stage_prefix}_Propellant_Mass_kg'] = s['propellant_mass_kg']
        flat[f'{stage_prefix}_Start_Mass_kg'] = s['start_mass_kg']
        flat[f'{stage_prefix}_Total_Impulse_s'] = s['total_impulse']
        if idx <= 2:  # Only 1st and 2nd stages have thrust in the dataset
            flat[f'{stage_prefix}_Total_Thrust_N'] = s['total_thrust_n']
        
        # Add calculation error for 2nd and transfer stages
        if idx >= 2:
            calc_error = s['delta_v_m_s'] - (s['isp_s'] * G0 * math.log(s['start_mass_kg'] / s['final_mass_kg']))
            flat[f'{stage_prefix}_Calculation_Error_m_per_s'] = calc_error
    
    flat['payload_100km_kg'] = payload_100km_adj
    flat['liftoff_TW'] = liftoff_tw
    flat['total_initial_mass_kg'] = total_initial_mass
    flat['sum_delta_v_m_s'] = sum_dv
    flat['n_stages'] = len(stage_results)
    
    return flat

# Generate synthetic rockets
synthetic_flat_list = []
failed = 0
print("Generating synthetic rockets... (this may take a moment)")
for i in range(N_SYNTH):
    out = generate_one_rocket(dists, payload_100km_samples, baseline_total_dv)
    if out is None:
        failed += 1
        continue
    synthetic_flat_list.append(out)

print(f"Finished generation: successful={len(synthetic_flat_list)}, failed_attempts={failed}")

# Create output DataFrame in the new format (8 rows per rocket for different altitudes)
altitudes = [100, 200, 300, 400, 500, 600, 700, 800]
output_rows = []

for i, rocket_data in enumerate(synthetic_flat_list):
    rocket_name = f"Rocket {i+1:03d}"
    
    # Generate payload for each altitude using a simple decay model
    # Payload decreases with altitude - we'll use a exponential decay model
    payload_100km = rocket_data['payload_100km_kg']
    
    # Random decay rate based on rocket size (larger rockets have less relative decay)
    base_decay = np.random.normal(0.001, 0.0003)
    size_factor = 1.0 / (1.0 + rocket_data['total_initial_mass_kg'] / 100000)  # Larger rockets decay less
    decay_rate = base_decay * size_factor
    
    for alt in altitudes:
        # Exponential decay: payload = payload_100km * exp(-decay_rate * (alt - 100))
        altitude_penalty = math.exp(-decay_rate * (alt - 100))
        payload = payload_100km * altitude_penalty
        payload = max(0.0, payload)  # Ensure non-negative payload
        
        # Create a row for this altitude
        row_data = {'Stage_Rockets': rocket_name, 'Altitude_km': alt, 'Payload_kg': payload}
        
        # Copy all the rocket parameters (same for all altitudes)
        for key, value in rocket_data.items():
            if key not in ['payload_100km_kg']:  # We use the computed payload per altitude
                row_data[key] = value
        
        output_rows.append(row_data)

# Create DataFrame
out_df = pd.DataFrame(output_rows)

# Define expected columns based on the dataset structure
expected_columns = [
    'Stage_Rockets', 
    '1st_Stage_Average_Isp_s', '1st_Stage_Delta_v_m_per_s', '1st_Stage_Dry_Mass_kg', 
    '1st_Stage_Engine_Run_Time_s', '1st_Stage_Final_Mass_kg', '1st_Stage_Max_Acceleration_m_per_s_squared', 
    '1st_Stage_Propellant_Mass_kg', '1st_Stage_Start_Mass_kg', '1st_Stage_Total_Impulse_s', 
    '1st_Stage_Total_Thrust_N', '2nd_Stage_Calculation_Error_m_per_s', '2nd_Stage_Delta_v_m_per_s', 
    '2nd_Stage_Dry_Mass_kg', '2nd_Stage_Engine_Run_Time_s', '2nd_Stage_Final_Mass_kg', 
    '2nd_Stage_Max_Acceleration_m_per_s_squared', '2nd_Stage_Propellant_Mass_kg', '2nd_Stage_Start_Mass_kg', 
    '2nd_Stage_Total_Impulse_s', '2nd_Stage_Total_Thrust_N', 'Transfer_Stage_Calculation_Error_m_per_s', 
    'Transfer_Stage_Delta_v_m_per_s', 'Transfer_Stage_Dry_Mass_kg', 'Transfer_Stage_Final_Mass_kg', 
    'Transfer_Stage_Max_Acceleration_m_per_s_squared', 'Transfer_Stage_Propellant_Mass_kg', 
    'Transfer_Stage_Start_Mass_kg', 'Transfer_Stage_Total_Impulse_s', 'Altitude_km', 'Payload_kg'
]

# Add any missing columns with NaN
for col in expected_columns:
    if col not in out_df.columns:
        if col == 'Stage_Rockets':
            # This should already be set, but just in case
            continue
        else:
            out_df[col] = np.nan

# Reorder columns to match expected structure
out_df = out_df[expected_columns]

# Save output
out_df.to_excel(OUTPUT_XLSX, index=False)

# Write report
report = {
    'n_requested': N_SYNTH,
    'n_generated': len(synthetic_flat_list),
    'n_failed_attempts': failed,
    'n_total_rows': len(out_df),
    'baseline_total_dv_used_m_s': baseline_total_dv,
    'rockets_per_altitude': 8,
    'altitude_range': '100-800km in 100km increments',
    'isp_model': 'mass_and_technology_based'
}
with open(REPORT_JSON, 'w') as f:
    json.dump(report, f, indent=2)

print("Saved synthetic dataset to:", OUTPUT_XLSX)
print(f"Output shape: {out_df.shape} ({len(synthetic_flat_list)} rockets × 8 altitudes + header)")
print("Report written to:", REPORT_JSON)
print("Generation summary:", report)