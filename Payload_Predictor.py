import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.metrics import r2_score, mean_squared_error

def align_features(input_df, model):
    """
    Aligns input_df columns to match the training features of the model.
    Adds missing features (set to 0.0) and removes unexpected extras.
    """
    if hasattr(model.estimators_[0], "feature_names_in_"):
        expected = model.estimators_[0].feature_names_in_
        input_df = input_df.reindex(columns=expected, fill_value=0.0)
    return input_df

# Set page config
st.set_page_config(
    page_title="Rocket Payload Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Define all possible features for LightGBM
first_stage_features = [
    "1st Stage_Average Isp (s)",
    "1st Stage_Delta-v (m/s)",
    "1st Stage_Dry Mass (kg)",
    "1st Stage_Engine Run Time (s)",
    "1st Stage_Final Mass (kg)",
    "1st Stage_Max Acceleration (m/s²)",
    "1st Stage_Propellant Mass (kg)",
    "1st Stage_Start Mass (kg)",
    "1st Stage_Total Impulse (s)",
    "1st Stage_Total Thrust (N)"
]

second_stage_features = [
    "2nd Stage_Calculation Error (m/s)",
    "2nd Stage_Delta-v (m/s)",
    "2nd Stage_Dry Mass (kg)",
    "2nd Stage_Engine Run Time (s)",
    "2nd Stage_Final Mass (kg)",
    "2nd Stage_Max Acceleration (m/s²)",
    "2nd Stage_Propellant Mass (kg)",
    "2nd Stage_Start Mass (kg)",
    "2nd Stage_Total Impulse (s)",
    "2nd Stage_Total Thrust (N)"
]

transfer_stage_features = [
    "Transfer Stage_Calculation Error (m/s)",
    "Transfer Stage_Delta-v (m/s)",
    "Transfer Stage_Dry Mass (kg)",
    "Transfer Stage_Final Mass (kg)",
    "Transfer Stage_Max Acceleration (m/s²)",
    "Transfer Stage_Propellant Mass (kg)",
    "Transfer Stage_Start Mass (kg)",
    "Transfer Stage_Total Impulse (s)"
]

# Load models and feature sets
@st.cache_resource
def load_models():
    models = {}
    model_types = ['Random Forest', 'XGBoost', 'LightGBM']
    transfer_types = ['With Transfer', 'Without Transfer']
    
    for model_type in model_types:
        models[model_type] = {}
        for transfer_type in transfer_types:
            try:
                # Try the expected filename pattern first
                model_path = os.path.join(MODELS_DIR, f"{model_type.lower().replace(' ', '_')}_{transfer_type.lower().replace(' ', '_')}_latest.pkl")
                
                # If not found, try to find any matching file
                if not os.path.exists(model_path):
                    # Look for any file that contains the model type and transfer type
                    matching_files = [f for f in os.listdir(MODELS_DIR) if all(term in f.lower() for term in [model_type.lower().replace(' ', '_'), transfer_type.lower().replace(' ', '_')]) and f.endswith('.pkl')]
                    
                    if matching_files:
                        # Use the most recent file (by modification time)
                        matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True)
                        model_path = os.path.join(MODELS_DIR, matching_files[0])
                        st.info(f"Using {matching_files[0]} for {model_type} ({transfer_type})")
                
                if os.path.exists(model_path):
                    models[model_type][transfer_type] = joblib.load(model_path)
                else:
                    st.error(f"Could not find model file for {model_type} ({transfer_type})")
                    
            except Exception as e:
                st.error(f"Could not load {model_type} model for {transfer_type}: {str(e)}")
    
    return models

@st.cache_data
def load_feature_sets():
    try:
        with open(os.path.join(DATA_DIR, 'feature_sets.json'), 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load feature sets: {str(e)}")
        return {}

# Load models and feature sets
models = load_models()
feature_sets = load_feature_sets()

# App title and description
st.title("🚀 Rocket Payload Predictior")
st.markdown("""
This app predicts rocket payload capacity for 5 different orbits using 3 different machine learning frameworks.
Select a model and input the required parameters to get predictions.
            Note: The 'With Transfer Stage' button represents the third stage of a rocket.
            
""")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_type = st.sidebar.selectbox(
    "Select Model",
    options=list(models.keys()),
    help="Choose which machine learning model to use for prediction"
)

has_transfer = st.sidebar.radio(
    "Does the rocket have a transfer stage?",
    options=["With Transfer", "Without Transfer"],
    help="Select whether your rocket design includes a transfer stage"
)

# Get the appropriate feature set
if model_type in feature_sets and has_transfer in feature_sets[model_type]:
    required_features = feature_sets[model_type][has_transfer]
else:
    st.error("Feature set not found for the selected model and transfer type")
    st.stop()

# Display input fields based on selected model and transfer stage
st.header("Input Parameters")
input_data = {}

if required_features == "all":
    st.info("This model uses all available features. Please provide values for all parameters.")
    
    # Determine which features to show based on transfer stage selection
    if has_transfer == "With Transfer":
        all_features = first_stage_features + second_stage_features + transfer_stage_features
    else:
        all_features = first_stage_features + second_stage_features
    
    # Create input fields for all features
    for feature in all_features:
        if '_' in feature:
            stage, param = feature.split('_', 1)
            # Create a clean label for the input field
            label = param.replace('_', ' ').title()
            input_data[feature] = st.number_input(
                label=f"{stage}: {label}",
                value=0.0,
                step=0.1,
                help=f"Enter value for {stage} {label}",
                key=feature  # Add this line - use the full feature name as unique key
            )
else:
    # Group features by stage for better organization
    stages = {}
    for feature in required_features:
        if '_' in feature:
            stage, param = feature.split('_', 1)
            if stage not in stages:
                stages[stage] = []
            stages[stage].append((param, feature))
    
    # Create input fields organized by stage
    for stage, params in stages.items():
        st.subheader(f"{stage} Stage Parameters")
        cols = st.columns(2)
        col_idx = 0
    
        for param, feature in params:
            with cols[col_idx]:
                input_data[feature] = st.number_input(
                    label=param.replace('_', ' ').title(),
                    value=0.0,
                    step=0.1,
                    help=f"Enter value for {param.replace('_', ' ')}",
                    key=feature  # Add this line - use the full feature name as unique key
                )
            col_idx = (col_idx + 1) % 2

# Prediction button
if st.button("Predict Payload", type="primary"):
    if required_features == "all":
        # For LightGBM, use all features based on transfer stage selection
        if has_transfer == "With Transfer":
            all_features = first_stage_features + second_stage_features + transfer_stage_features
        else:
            all_features = first_stage_features + second_stage_features
        
        # Prepare input data for prediction
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present
        for feature in all_features:
            if feature not in input_df.columns:
                input_df[feature] = 0.0
        
        # Select only the required features in the correct order
        input_df = input_df[all_features]
    else:
        # Prepare input data for prediction
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for feature in required_features:
            if feature not in input_df.columns:
                input_df[feature] = 0.0
        
        # Select only the required features in the correct order
        input_df = input_df[required_features]
    
    # Make prediction
    try:
        model = models[model_type][has_transfer]
        input_df = align_features(input_df, model)
        prediction = model.predict(input_df)
        
        # Display results
        st.header("Prediction Results")
        
        orbits = ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']
        results_df = pd.DataFrame({
            'Orbit': orbits,
            'Predicted Payload (kg)': prediction[0]
        })
        
        # Display prediction table
        st.dataframe(results_df, use_container_width=True)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(orbits, prediction[0], color=sns.color_palette("viridis", len(orbits)))
        ax.set_ylabel('Payload (kg)')
        ax.set_title('Predicted Payload by Orbit')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add some information about the models
st.sidebar.header("About")
st.sidebar.info("""
This app uses machine learning models trained on rocket performance data to predict payload capacity for various orbits.

**Models available:**
- Random Forest
- XGBoost
- LightGBM

Select a model and input the required parameters to get predictions.
""")