import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ---------------------------------------------------------------------------
# Streamlit app: updated with CSV batch upload, demo values, and human-readable
# labels for each input box (e.g. "1st Stage: Average Isp (s)").
# ---------------------------------------------------------------------------

def align_features(input_df, model):
    """
    Align input_df columns to match the training features expected by the model.
    """
    try:
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            input_df = input_df.reindex(columns=expected, fill_value=0.0)
        elif hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_names_in_"):
            expected = list(model.estimators_[0].feature_names_in_)
            input_df = input_df.reindex(columns=expected, fill_value=0.0)
    except Exception:
        pass
    return input_df

# Page config
st.set_page_config(
    page_title="Rocket Payload Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Feature list (exact internal column names expected by the training pipeline)
FEATURE_COLUMNS = [
    '1st_Stage_Average_Isp_s', '1st_Stage_Delta_v_m_per_s', '1st_Stage_Dry_Mass_kg',
    '1st_Stage_Engine_Run_Time_s', '1st_Stage_Final_Mass_kg',
    '1st_Stage_Max_Acceleration_m_per_s_squared', '1st_Stage_Propellant_Mass_kg',
    '1st_Stage_Start_Mass_kg', '1st_Stage_Total_Impulse_s', '1st_Stage_Total_Thrust_N',
    '2nd_Stage_Calculation_Error_m_per_s', '2nd_Stage_Delta_v_m_per_s', '2nd_Stage_Dry_Mass_kg',
    '2nd_Stage_Engine_Run_Time_s', '2nd_Stage_Final_Mass_kg',
    '2nd_Stage_Max_Acceleration_m_per_s_squared', '2nd_Stage_Propellant_Mass_kg',
    '2nd_Stage_Start_Mass_kg', '2nd_Stage_Total_Impulse_s', '2nd_Stage_Total_Thrust_N',
    'Transfer_Stage_Calculation_Error_m_per_s', 'Transfer_Stage_Delta_v_m_per_s',
    'Transfer_Stage_Dry_Mass_kg', 'Transfer_Stage_Final_Mass_kg',
    'Transfer_Stage_Max_Acceleration_m_per_s_squared', 'Transfer_Stage_Propellant_Mass_kg',
    'Transfer_Stage_Start_Mass_kg', 'Transfer_Stage_Total_Impulse_s', 'Altitude_km'
]

# Groupings for UI
FIRST_STAGE = FEATURE_COLUMNS[0:10]
SECOND_STAGE = FEATURE_COLUMNS[10:20]
TRANSFER_STAGE = FEATURE_COLUMNS[20:28]  # excludes Altitude
ALTITUDE = FEATURE_COLUMNS[-1]

MODEL_TYPES = ['Random Forest', 'XGBoost', 'LightGBM']
TRANSFER_TYPES = ['With Transfer', 'Without Transfer']

# Human-readable label formatter
def human_label(col_name: str) -> str:
    # Split into stage and param (first underscore)
    if '_' in col_name:
        stage_key, param = col_name.split('_', 1)
    else:
        stage_key, param = col_name, ''

    # Friendly stage
    stage_map = {
        '1st': '1st Stage',
        '1st_Stage': '1st Stage',
        '2nd': '2nd Stage',
        '2nd_Stage': '2nd Stage',
        'Transfer_Stage': 'Transfer Stage',
        'Transfer': 'Transfer Stage'
    }
    stage = stage_map.get(stage_key, stage_key.replace('_', ' '))

    # Clean parameter part
    p = param.replace('_', ' ')
    p = p.replace('m per s squared', 'm/s²')
    p = p.replace('m per s', 'm/s')
    p = p.replace('m per s', 'm/s')
    p = p.replace(' s', ' (s)') if p.endswith(' s') else p
    # handle common suffix tokens
    p = p.replace(' s', '(s)')
    p = p.replace('kg', 'kg')
    p = p.replace('Calculation Error', 'Calculation Error (m/s)') if 'Calculation Error' in p and '(m/s)' not in p else p
    p = p.replace('Delta v', 'Delta-v (m/s)') if 'Delta v' in p and '(m/s)' not in p else p
    p = p.replace('Average Isp s', 'Average Isp (s)') if 'Average Isp' in p and '(s)' not in p else p

    # Make final label
    label = f"{stage}: {p.strip()}" if p.strip() else stage
    # Small tidy-ups
    label = label.replace('  ', ' ')
    label = label.replace('(s))', '(s)')
    label = label.replace('(m/s))', '(m/s)')
    return label

# Provide an explicit mapping table for display and CSV guidance
HUMAN_LABELS = {col: human_label(col) for col in FEATURE_COLUMNS}

# Cached model loader
@st.cache_resource
def load_models():
    models = {m: {} for m in MODEL_TYPES}
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)

    for model_type in MODEL_TYPES:
        for transfer_type in TRANSFER_TYPES:
            normalized_model = model_type.lower().replace(' ', '_')
            normalized_transfer = transfer_type.lower().replace(' ', '_')
            expected_filename = f"{normalized_model}_{normalized_transfer}_latest.pkl"
            expected_path = os.path.join(MODELS_DIR, expected_filename)
            model_obj = None
            chosen_path = None
            try:
                if os.path.exists(expected_path):
                    model_obj = joblib.load(expected_path)
                    chosen_path = expected_path
                else:
                    # try to find matching file
                    if os.path.exists(MODELS_DIR):
                        candidates = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
                        matches = [f for f in candidates if normalized_model in f and normalized_transfer in f]
                        if matches:
                            matches.sort(key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True)
                            chosen = matches[0]
                            chosen_path = os.path.join(MODELS_DIR, chosen)
                            model_obj = joblib.load(chosen_path)
                            st.info(f"Using {chosen} for {model_type} ({transfer_type})")
            except Exception as e:
                st.error(f"Failed to load {model_type} ({transfer_type}): {e}")

            models[model_type][transfer_type] = {'model': model_obj, 'path': chosen_path}
            if model_obj is None:
                st.warning(f"Model not available: {model_type} ({transfer_type})")
    return models

@st.cache_data
def load_feature_sets():
    path = os.path.join(DATA_DIR, 'feature_sets.json')
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Could not load feature_sets.json: {e}")
    return {}

models_info = load_models()
feature_sets = load_feature_sets()

# --- Header
st.title("🚀 Rocket Payload Predictor")
st.markdown("Provide stage parameters in the boxes below, choose a model and whether the rocket includes a transfer stage, then click Predict Payload.")

# --- Sidebar with CSV upload guidance
st.sidebar.header('Model Selection')
model_choice = st.sidebar.selectbox('Select Model', options=MODEL_TYPES)
has_transfer = st.sidebar.radio('Does the rocket have a transfer stage?', options=TRANSFER_TYPES)

st.sidebar.markdown('---')
st.sidebar.write('If a pre-trained model is not found you can upload a .pkl file here:')
uploaded_model = st.sidebar.file_uploader('Upload model (.pkl)', type=['pkl', 'joblib'])

st.sidebar.markdown('---')
st.sidebar.header('Batch Predictions (CSV)')
st.sidebar.write('Upload a CSV with one row per rocket to run batch predictions. The CSV must contain the exact internal column names (underscored) shown below as headers. Missing columns will be filled with zeros automatically, but it is best to include all columns to avoid unintended results.')

# Show mapping table small
label_df = pd.DataFrame({'column_name': FEATURE_COLUMNS, 'label': [HUMAN_LABELS[c] for c in FEATURE_COLUMNS]})
st.sidebar.dataframe(label_df, height=400)

# Download a CSV template with demo values
if st.sidebar.button('Download CSV template'):
    demo_row = {c: (0.0 if 'Calculation_Error' in c else 1.0 if 'Average_Isp' in c else 100.0) for c in FEATURE_COLUMNS}
    template_df = pd.DataFrame([demo_row])
    csv = template_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button('Download template (CSV)', csv, file_name='payload_input_template.csv')

# CSV uploader in main area for batch predictions
st.header('Batch Predictions (optional)')
csv_file = st.file_uploader('Upload CSV for multiple rockets (optional)', type=['csv'])

# --- Input area (single-row inputs)
st.header('Single Rocket Inputs')
st.caption('Click "Fill demo values" to populate fields automatically for a quick test.')

# Demo values dictionary - choose realistic-ish defaults for quick demo
DEMO_VALUES = {
    '1st_Stage_Average_Isp_s': 300.0,
    '1st_Stage_Delta_v_m_per_s': 2500.0,
    '1st_Stage_Dry_Mass_kg': 1500.0,
    '1st_Stage_Engine_Run_Time_s': 150.0,
    '1st_Stage_Final_Mass_kg': 400.0,
    '1st_Stage_Max_Acceleration_m_per_s_squared': 40.0,
    '1st_Stage_Propellant_Mass_kg': 5000.0,
    '1st_Stage_Start_Mass_kg': 6500.0,
    '1st_Stage_Total_Impulse_s': 12000.0,
    '1st_Stage_Total_Thrust_N': 800000.0,
    '2nd_Stage_Calculation_Error_m_per_s': 0.0,
    '2nd_Stage_Delta_v_m_per_s': 6000.0,
    '2nd_Stage_Dry_Mass_kg': 400.0,
    '2nd_Stage_Engine_Run_Time_s': 200.0,
    '2nd_Stage_Final_Mass_kg': 150.0,
    '2nd_Stage_Max_Acceleration_m_per_s_squared': 25.0,
    '2nd_Stage_Propellant_Mass_kg': 2000.0,
    '2nd_Stage_Start_Mass_kg': 2500.0,
    '2nd_Stage_Total_Impulse_s': 6000.0,
    '2nd_Stage_Total_Thrust_N': 200000.0,
    'Transfer_Stage_Calculation_Error_m_per_s': 0.0,
    'Transfer_Stage_Delta_v_m_per_s': 800.0,
    'Transfer_Stage_Dry_Mass_kg': 50.0,
    'Transfer_Stage_Final_Mass_kg': 20.0,
    'Transfer_Stage_Max_Acceleration_m_per_s_squared': 10.0,
    'Transfer_Stage_Propellant_Mass_kg': 300.0,
    'Transfer_Stage_Start_Mass_kg': 370.0,
    'Transfer_Stage_Total_Impulse_s': 1200.0,
    'Altitude_km': 500.0
}

# Buttons for demo / reset
demo_col, reset_col = st.columns([1,1])
with demo_col:
    if st.button('Fill demo values'):
        for k, v in DEMO_VALUES.items():
            st.session_state[k] = float(v)
        st.success('Demo values populated. You can modify any field before predicting.')

with reset_col:
    if st.button('Reset inputs'):
        for k in FEATURE_COLUMNS:
            st.session_state[k] = 0.0
        st.info('Inputs reset to 0.0. (You may need to scroll to see fields update.)')

# Render inputs — use keys = exact feature column names so demo/reset populate correctly
input_data = {}
col1, col2 = st.columns(2)
with col1:
    st.subheader('1st Stage')
    for feat in FIRST_STAGE:
        label = HUMAN_LABELS[feat]
        input_data[feat] = st.number_input(label=label, value=st.session_state.get(feat, 0.0), step=0.1, key=feat, help=f'Internal name: {feat}')

with col2:
    st.subheader('2nd Stage')
    for feat in SECOND_STAGE:
        label = HUMAN_LABELS[feat]
        input_data[feat] = st.number_input(label=label, value=st.session_state.get(feat, 0.0), step=0.1, key=feat+'_2', help=f'Internal name: {feat}')
        # Note: use a slightly different key for side-by-side inputs to avoid duplicate-key errors
        # We'll still map the value below using st.session_state[feat+'_2']

# Transfer stage
st.subheader('Transfer Stage & Mission')
if has_transfer == 'With Transfer':
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        for feat in TRANSFER_STAGE[:4]:
            label = HUMAN_LABELS[feat]
            input_data[feat] = st.number_input(label=label, value=st.session_state.get(feat, 0.0), step=0.1, key=feat+'_t1', help=f'Internal name: {feat}')
    with tcol2:
        for feat in TRANSFER_STAGE[4:]:
            label = HUMAN_LABELS[feat]
            input_data[feat] = st.number_input(label=label, value=st.session_state.get(feat, 0.0), step=0.1, key=feat+'_t2', help=f'Internal name: {feat}')
    # Altitude
    input_data[ALTITUDE] = st.number_input(label=HUMAN_LABELS[ALTITUDE], value=st.session_state.get(ALTITUDE, 0.0), step=0.1, key=ALTITUDE)
else:
    # Without transfer: show altitude and keep transfer fields zero (but still mapped)
    input_data[ALTITUDE] = st.number_input(label=HUMAN_LABELS[ALTITUDE], value=st.session_state.get(ALTITUDE, 0.0), step=0.1, key=ALTITUDE)
    for feat in TRANSFER_STAGE:
        # create disabled inputs visually to show they are not used
        _ = st.number_input(label=f"{HUMAN_LABELS[feat]} (hidden)", value=0.0, step=0.1, key=feat+'_hidden', disabled=True, help='Transfer stage not selected')
        input_data[feat] = 0.0

# Collect actual numeric values from session state where alternate keys were used
# (because of duplicate-key side-by-side handling)
for feat in FIRST_STAGE:
    # keys used exactly as feat
    input_data[feat] = float(st.session_state.get(feat, input_data.get(feat, 0.0)))
for idx, feat in enumerate(SECOND_STAGE):
    # second-stage inputs used keys feat+'_2'
    input_data[feat] = float(st.session_state.get(feat+'_2', input_data.get(feat, 0.0)))
if has_transfer == 'With Transfer':
    for i, feat in enumerate(TRANSFER_STAGE[:4]):
        input_data[feat] = float(st.session_state.get(feat+'_t1', input_data.get(feat, 0.0)))
    for i, feat in enumerate(TRANSFER_STAGE[4:]):
        input_data[feat] = float(st.session_state.get(feat+'_t2', input_data.get(feat, 0.0)))

# Build ordered list matching FEATURE_COLUMNS
ordered_values = [float(input_data.get(col, 0.0)) for col in FEATURE_COLUMNS]

# --- Batch CSV handling
batch_df = None
if csv_file is not None:
    try:
        batch_df = pd.read_csv(csv_file)
        st.success(f'Loaded CSV with {len(batch_df)} rows')

        # Check and fill missing columns
        missing_cols = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
        if missing_cols:
            st.warning(f'Missing columns detected in CSV. These will be filled with zeros: {missing_cols}')
            for c in missing_cols:
                batch_df[c] = 0.0

        # Reorder to training order
        batch_df = batch_df.reindex(columns=FEATURE_COLUMNS)

        st.dataframe(batch_df.head(10))

    except Exception as e:
        st.error(f'Failed to read CSV: {e}')

# --- Prediction buttons
colA, colB = st.columns([1,1])
with colA:
    if st.button('Predict Payload (Single)'):
        # Single prediction validation
        if all([v == 0.0 for v in ordered_values]):
            st.error('Please fill all values')
        else:
            X = pd.DataFrame([ordered_values], columns=FEATURE_COLUMNS)

            # load model: uploaded overrides auto-loaded
            model_obj = None
            model_path_info = None
            if uploaded_model is not None:
                try:
                    model_obj = joblib.load(uploaded_model)
                    model_path_info = 'Uploaded model'
                except Exception as e:
                    st.error(f'Uploaded model could not be loaded: {e}')

            if model_obj is None:
                scraped = models_info.get(model_choice, {}).get(has_transfer, {})
                model_obj = scraped.get('model')
                model_path_info = scraped.get('path')

            if model_obj is None:
                st.error('No suitable model found. Upload a .pkl in the sidebar or place a model in results/models/.')
            else:
                try:
                    X_aligned = align_features(X, model_obj)
                    pred = model_obj.predict(X_aligned)
                    pred_arr = np.asarray(pred)

                    if pred_arr.size == 1:
                        pred_val = float(pred_arr.reshape(-1)[0])
                        st.header('Prediction Result')
                        st.success(f'Predicted payload: {pred_val:.3f} kg')

                        out_df = pd.DataFrame([ordered_values], columns=FEATURE_COLUMNS)
                        out_df['Predicted_Payload_kg'] = pred_val
                        csv = out_df.to_csv(index=False).encode('utf-8')
                        st.download_button('Download prediction (CSV)', csv, file_name='payload_prediction.csv')
                    else:
                        st.header('Prediction Result (multiple outputs)')
                        orbits = ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']
                        if pred_arr.reshape(-1).size == len(orbits):
                            results_df = pd.DataFrame({'Orbit': orbits, 'Predicted Payload (kg)': pred_arr.reshape(-1)})
                        else:
                            results_df = pd.DataFrame({'Prediction_Index': list(range(pred_arr.reshape(-1).size)), 'Value': pred_arr.reshape(-1)})
                        st.dataframe(results_df, use_container_width=True)
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button('Download predictions (CSV)', csv, file_name='payload_predictions_single.csv')

                    if model_path_info:
                        st.info(f'Using model file: {model_path_info}')

                except Exception as e:
                    st.error(f'Prediction failed: {e}')

with colB:
    if st.button('Predict Payload (Batch from CSV)'):
        if batch_df is None:
            st.error('Please upload a CSV with input rows first')
        else:
            # load model
            model_obj = None
            model_path_info = None
            if uploaded_model is not None:
                try:
                    model_obj = joblib.load(uploaded_model)
                    model_path_info = 'Uploaded model'
                except Exception as e:
                    st.error(f'Uploaded model could not be loaded: {e}')

            if model_obj is None:
                scraped = models_info.get(model_choice, {}).get(has_transfer, {})
                model_obj = scraped.get('model')
                model_path_info = scraped.get('path')

            if model_obj is None:
                st.error('No suitable model found. Upload a .pkl in the sidebar or place a model in results/models/.')
            else:
                try:
                    batch_aligned = align_features(batch_df.copy(), model_obj)
                    preds = model_obj.predict(batch_aligned)
                    preds_arr = np.asarray(preds)

                    # attach predictions to DataFrame
                    if preds_arr.ndim == 1:
                        batch_df['Predicted_Payload_kg'] = preds_arr
                    else:
                        # if multiple columns predicted, flatten appropriate
                        # try to add columns P0,P1,...
                        for i in range(preds_arr.shape[1]):
                            batch_df[f'Pred_{i}'] = preds_arr[:, i]

                    st.success('Batch prediction complete')
                    st.dataframe(batch_df.head(10))
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button('Download batch predictions (CSV)', csv, file_name='payload_batch_predictions.csv')

                    if model_path_info:
                        st.info(f'Using model file: {model_path_info}')

                except Exception as e:
                    st.error(f'Batch prediction failed: {e}')

# Footer note / CSV structure guidance
st.markdown('---')
st.header('CSV Structure & Notes')
st.write('The CSV used for batch predictions should have one row per rocket and column headers that match the internal feature names exactly (underscored). Example header names:')
example_cols = pd.DataFrame({'internal_name': FEATURE_COLUMNS, 'label': [HUMAN_LABELS[c] for c in FEATURE_COLUMNS]})
st.table(example_cols)
st.write('Missing columns will be filled with zeros automatically, but this may produce misleading results if crucial features are absent. Altitude (Altitude_km) should be in kilometers. Calculation error columns should be numeric; if not available set them to 0.')

# End of script
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ---------------------------------------------------------------------------
# Streamlit app: updated with CSV batch upload, demo values, and human-readable
# labels for each input box (e.g. "1st Stage: Average Isp (s)").
# ---------------------------------------------------------------------------

def align_features(input_df, model):
    """
    Align input_df columns to match the training features expected by the model.
    """
    try:
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            input_df = input_df.reindex(columns=expected, fill_value=0.0)
        elif hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_names_in_"):
            expected = list(model.estimators_[0].feature_names_in_)
            input_df = input_df.reindex(columns=expected, fill_value=0.0)
    except Exception:
        pass
    return input_df

# Page config
st.set_page_config(
    page_title="Rocket Payload Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Feature list (exact internal column names expected by the training pipeline)
FEATURE_COLUMNS = [
    '1st_Stage_Average_Isp_s', '1st_Stage_Delta_v_m_per_s', '1st_Stage_Dry_Mass_kg',
    '1st_Stage_Engine_Run_Time_s', '1st_Stage_Final_Mass_kg',
    '1st_Stage_Max_Acceleration_m_per_s_squared', '1st_Stage_Propellant_Mass_kg',
    '1st_Stage_Start_Mass_kg', '1st_Stage_Total_Impulse_s', '1st_Stage_Total_Thrust_N',
    '2nd_Stage_Calculation_Error_m_per_s', '2nd_Stage_Delta_v_m_per_s', '2nd_Stage_Dry_Mass_kg',
    '2nd_Stage_Engine_Run_Time_s', '2nd_Stage_Final_Mass_kg',
    '2nd_Stage_Max_Acceleration_m_per_s_squared', '2nd_Stage_Propellant_Mass_kg',
    '2nd_Stage_Start_Mass_kg', '2nd_Stage_Total_Impulse_s', '2nd_Stage_Total_Thrust_N',
    'Transfer_Stage_Calculation_Error_m_per_s', 'Transfer_Stage_Delta_v_m_per_s',
    'Transfer_Stage_Dry_Mass_kg', 'Transfer_Stage_Final_Mass_kg',
    'Transfer_Stage_Max_Acceleration_m_per_s_squared', 'Transfer_Stage_Propellant_Mass_kg',
    'Transfer_Stage_Start_Mass_kg', 'Transfer_Stage_Total_Impulse_s', 'Altitude_km'
]

# Groupings for UI
FIRST_STAGE = FEATURE_COLUMNS[0:10]
SECOND_STAGE = FEATURE_COLUMNS[10:20]
TRANSFER_STAGE = FEATURE_COLUMNS[20:28]  # excludes Altitude
ALTITUDE = FEATURE_COLUMNS[-1]

MODEL_TYPES = ['Random Forest', 'XGBoost', 'LightGBM']
TRANSFER_TYPES = ['With Transfer', 'Without Transfer']

# Human-readable label formatter
def human_label(col_name: str) -> str:
    # Split into stage and param (first underscore)
    if '_' in col_name:
        stage_key, param = col_name.split('_', 1)
    else:
        stage_key, param = col_name, ''

    # Friendly stage
    stage_map = {
        '1st': '1st Stage',
        '1st_Stage': '1st Stage',
        '2nd': '2nd Stage',
        '2nd_Stage': '2nd Stage',
        'Transfer_Stage': 'Transfer Stage',
        'Transfer': 'Transfer Stage'
    }
    stage = stage_map.get(stage_key, stage_key.replace('_', ' '))

    # Clean parameter part
    p = param.replace('_', ' ')
    p = p.replace('m per s squared', 'm/s²')
    p = p.replace('m per s', 'm/s')
    p = p.replace('m per s', 'm/s')
    p = p.replace(' s', ' (s)') if p.endswith(' s') else p
    # handle common suffix tokens
    p = p.replace(' s', '(s)')
    p = p.replace('kg', 'kg')
    p = p.replace('Calculation Error', 'Calculation Error (m/s)') if 'Calculation Error' in p and '(m/s)' not in p else p
    p = p.replace('Delta v', 'Delta-v (m/s)') if 'Delta v' in p and '(m/s)' not in p else p
    p = p.replace('Average Isp s', 'Average Isp (s)') if 'Average Isp' in p and '(s)' not in p else p

    # Make final label
    label = f"{stage}: {p.strip()}" if p.strip() else stage
    # Small tidy-ups
    label = label.replace('  ', ' ')
    label = label.replace('(s))', '(s)')
    label = label.replace('(m/s))', '(m/s)')
    return label

# Provide an explicit mapping table for display and CSV guidance
HUMAN_LABELS = {col: human_label(col) for col in FEATURE_COLUMNS}

# Cached model loader
@st.cache_resource
def load_models():
    models = {m: {} for m in MODEL_TYPES}
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)

    for model_type in MODEL_TYPES:
        for transfer_type in TRANSFER_TYPES:
            normalized_model = model_type.lower().replace(' ', '_')
            normalized_transfer = transfer_type.lower().replace(' ', '_')
            expected_filename = f"{normalized_model}_{normalized_transfer}_latest.pkl"
            expected_path = os.path.join(MODELS_DIR, expected_filename)
            model_obj = None
            chosen_path = None
            try:
                if os.path.exists(expected_path):
                    model_obj = joblib.load(expected_path)
                    chosen_path = expected_path
                else:
                    # try to find matching file
                    if os.path.exists(MODELS_DIR):
                        candidates = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
                        matches = [f for f in candidates if normalized_model in f and normalized_transfer in f]
                        if matches:
                            matches.sort(key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True)
                            chosen = matches[0]
                            chosen_path = os.path.join(MODELS_DIR, chosen)
                            model_obj = joblib.load(chosen_path)
                            st.info(f"Using {chosen} for {model_type} ({transfer_type})")
            except Exception as e:
                st.error(f"Failed to load {model_type} ({transfer_type}): {e}")

            models[model_type][transfer_type] = {'model': model_obj, 'path': chosen_path}
            if model_obj is None:
                st.warning(f"Model not available: {model_type} ({transfer_type})")
    return models

@st.cache_data
def load_feature_sets():
    path = os.path.join(DATA_DIR, 'feature_sets.json')
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Could not load feature_sets.json: {e}")
    return {}

models_info = load_models()
feature_sets = load_feature_sets()

# --- Header
st.title("🚀 Rocket Payload Predictor")
st.markdown("Provide stage parameters in the boxes below, choose a model and whether the rocket includes a transfer stage, then click Predict Payload.")

# --- Sidebar with CSV upload guidance
st.sidebar.header('Model Selection')
model_choice = st.sidebar.selectbox('Select Model', options=MODEL_TYPES)
has_transfer = st.sidebar.radio('Does the rocket have a transfer stage?', options=TRANSFER_TYPES)

st.sidebar.markdown('---')
st.sidebar.write('If a pre-trained model is not found you can upload a .pkl file here:')
uploaded_model = st.sidebar.file_uploader('Upload model (.pkl)', type=['pkl', 'joblib'])

st.sidebar.markdown('---')
st.sidebar.header('Batch Predictions (CSV)')
st.sidebar.write('Upload a CSV with one row per rocket to run batch predictions. The CSV must contain the exact internal column names (underscored) shown below as headers. Missing columns will be filled with zeros automatically, but it is best to include all columns to avoid unintended results.')

# Show mapping table small
label_df = pd.DataFrame({'column_name': FEATURE_COLUMNS, 'label': [HUMAN_LABELS[c] for c in FEATURE_COLUMNS]})
st.sidebar.dataframe(label_df, height=400)

# Download a CSV template with demo values
if st.sidebar.button('Download CSV template'):
    demo_row = {c: (0.0 if 'Calculation_Error' in c else 1.0 if 'Average_Isp' in c else 100.0) for c in FEATURE_COLUMNS}
    template_df = pd.DataFrame([demo_row])
    csv = template_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button('Download template (CSV)', csv, file_name='payload_input_template.csv')

# CSV uploader in main area for batch predictions
st.header('Batch Predictions (optional)')
csv_file = st.file_uploader('Upload CSV for multiple rockets (optional)', type=['csv'])

# --- Input area (single-row inputs)
st.header('Single Rocket Inputs')
st.caption('Click "Fill demo values" to populate fields automatically for a quick test.')

# Demo values dictionary - choose realistic-ish defaults for quick demo
DEMO_VALUES = {
    '1st_Stage_Average_Isp_s': 300.0,
    '1st_Stage_Delta_v_m_per_s': 2500.0,
    '1st_Stage_Dry_Mass_kg': 1500.0,
    '1st_Stage_Engine_Run_Time_s': 150.0,
    '1st_Stage_Final_Mass_kg': 400.0,
    '1st_Stage_Max_Acceleration_m_per_s_squared': 40.0,
    '1st_Stage_Propellant_Mass_kg': 5000.0,
    '1st_Stage_Start_Mass_kg': 6500.0,
    '1st_Stage_Total_Impulse_s': 12000.0,
    '1st_Stage_Total_Thrust_N': 800000.0,
    '2nd_Stage_Calculation_Error_m_per_s': 0.0,
    '2nd_Stage_Delta_v_m_per_s': 6000.0,
    '2nd_Stage_Dry_Mass_kg': 400.0,
    '2nd_Stage_Engine_Run_Time_s': 200.0,
    '2nd_Stage_Final_Mass_kg': 150.0,
    '2nd_Stage_Max_Acceleration_m_per_s_squared': 25.0,
    '2nd_Stage_Propellant_Mass_kg': 2000.0,
    '2nd_Stage_Start_Mass_kg': 2500.0,
    '2nd_Stage_Total_Impulse_s': 6000.0,
    '2nd_Stage_Total_Thrust_N': 200000.0,
    'Transfer_Stage_Calculation_Error_m_per_s': 0.0,
    'Transfer_Stage_Delta_v_m_per_s': 800.0,
    'Transfer_Stage_Dry_Mass_kg': 50.0,
    'Transfer_Stage_Final_Mass_kg': 20.0,
    'Transfer_Stage_Max_Acceleration_m_per_s_squared': 10.0,
    'Transfer_Stage_Propellant_Mass_kg': 300.0,
    'Transfer_Stage_Start_Mass_kg': 370.0,
    'Transfer_Stage_Total_Impulse_s': 1200.0,
    'Altitude_km': 500.0
}

# Buttons for demo / reset
demo_col, reset_col = st.columns([1,1])
with demo_col:
    if st.button('Fill demo values'):
        for k, v in DEMO_VALUES.items():
            st.session_state[k] = float(v)
        st.success('Demo values populated. You can modify any field before predicting.')

with reset_col:
    if st.button('Reset inputs'):
        for k in FEATURE_COLUMNS:
            st.session_state[k] = 0.0
        st.info('Inputs reset to 0.0. (You may need to scroll to see fields update.)')

# Render inputs — use keys = exact feature column names so demo/reset populate correctly
input_data = {}
col1, col2 = st.columns(2)
with col1:
    st.subheader('1st Stage')
    for feat in FIRST_STAGE:
        label = HUMAN_LABELS[feat]
        input_data[feat] = st.number_input(label=label, value=st.session_state.get(feat, 0.0), step=0.1, key=feat, help=f'Internal name: {feat}')

with col2:
    st.subheader('2nd Stage')
    for feat in SECOND_STAGE:
        label = HUMAN_LABELS[feat]
        input_data[feat] = st.number_input(label=label, value=st.session_state.get(feat, 0.0), step=0.1, key=feat+'_2', help=f'Internal name: {feat}')
        # Note: use a slightly different key for side-by-side inputs to avoid duplicate-key errors
        # We'll still map the value below using st.session_state[feat+'_2']

# Transfer stage
st.subheader('Transfer Stage & Mission')
if has_transfer == 'With Transfer':
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        for feat in TRANSFER_STAGE[:4]:
            label = HUMAN_LABELS[feat]
            input_data[feat] = st.number_input(label=label, value=st.session_state.get(feat, 0.0), step=0.1, key=feat+'_t1', help=f'Internal name: {feat}')
    with tcol2:
        for feat in TRANSFER_STAGE[4:]:
            label = HUMAN_LABELS[feat]
            input_data[feat] = st.number_input(label=label, value=st.session_state.get(feat, 0.0), step=0.1, key=feat+'_t2', help=f'Internal name: {feat}')
    # Altitude
    input_data[ALTITUDE] = st.number_input(label=HUMAN_LABELS[ALTITUDE], value=st.session_state.get(ALTITUDE, 0.0), step=0.1, key=ALTITUDE)
else:
    # Without transfer: show altitude and keep transfer fields zero (but still mapped)
    input_data[ALTITUDE] = st.number_input(label=HUMAN_LABELS[ALTITUDE], value=st.session_state.get(ALTITUDE, 0.0), step=0.1, key=ALTITUDE)
    for feat in TRANSFER_STAGE:
        # create disabled inputs visually to show they are not used
        _ = st.number_input(label=f"{HUMAN_LABELS[feat]} (hidden)", value=0.0, step=0.1, key=feat+'_hidden', disabled=True, help='Transfer stage not selected')
        input_data[feat] = 0.0

# Collect actual numeric values from session state where alternate keys were used
# (because of duplicate-key side-by-side handling)
for feat in FIRST_STAGE:
    # keys used exactly as feat
    input_data[feat] = float(st.session_state.get(feat, input_data.get(feat, 0.0)))
for idx, feat in enumerate(SECOND_STAGE):
    # second-stage inputs used keys feat+'_2'
    input_data[feat] = float(st.session_state.get(feat+'_2', input_data.get(feat, 0.0)))
if has_transfer == 'With Transfer':
    for i, feat in enumerate(TRANSFER_STAGE[:4]):
        input_data[feat] = float(st.session_state.get(feat+'_t1', input_data.get(feat, 0.0)))
    for i, feat in enumerate(TRANSFER_STAGE[4:]):
        input_data[feat] = float(st.session_state.get(feat+'_t2', input_data.get(feat, 0.0)))

# Build ordered list matching FEATURE_COLUMNS
ordered_values = [float(input_data.get(col, 0.0)) for col in FEATURE_COLUMNS]

# --- Batch CSV handling
batch_df = None
if csv_file is not None:
    try:
        batch_df = pd.read_csv(csv_file)
        st.success(f'Loaded CSV with {len(batch_df)} rows')

        # Check and fill missing columns
        missing_cols = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
        if missing_cols:
            st.warning(f'Missing columns detected in CSV. These will be filled with zeros: {missing_cols}')
            for c in missing_cols:
                batch_df[c] = 0.0

        # Reorder to training order
        batch_df = batch_df.reindex(columns=FEATURE_COLUMNS)

        st.dataframe(batch_df.head(10))

    except Exception as e:
        st.error(f'Failed to read CSV: {e}')

# --- Prediction buttons
colA, colB = st.columns([1,1])
with colA:
    if st.button('Predict Payload (Single)'):
        # Single prediction validation
        if all([v == 0.0 for v in ordered_values]):
            st.error('Please fill all values')
        else:
            X = pd.DataFrame([ordered_values], columns=FEATURE_COLUMNS)

            # load model: uploaded overrides auto-loaded
            model_obj = None
            model_path_info = None
            if uploaded_model is not None:
                try:
                    model_obj = joblib.load(uploaded_model)
                    model_path_info = 'Uploaded model'
                except Exception as e:
                    st.error(f'Uploaded model could not be loaded: {e}')

            if model_obj is None:
                scraped = models_info.get(model_choice, {}).get(has_transfer, {})
                model_obj = scraped.get('model')
                model_path_info = scraped.get('path')

            if model_obj is None:
                st.error('No suitable model found. Upload a .pkl in the sidebar or place a model in results/models/.')
            else:
                try:
                    X_aligned = align_features(X, model_obj)
                    pred = model_obj.predict(X_aligned)
                    pred_arr = np.asarray(pred)

                    if pred_arr.size == 1:
                        pred_val = float(pred_arr.reshape(-1)[0])
                        st.header('Prediction Result')
                        st.success(f'Predicted payload: {pred_val:.3f} kg')

                        out_df = pd.DataFrame([ordered_values], columns=FEATURE_COLUMNS)
                        out_df['Predicted_Payload_kg'] = pred_val
                        csv = out_df.to_csv(index=False).encode('utf-8')
                        st.download_button('Download prediction (CSV)', csv, file_name='payload_prediction.csv')
                    else:
                        st.header('Prediction Result (multiple outputs)')
                        orbits = ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']
                        if pred_arr.reshape(-1).size == len(orbits):
                            results_df = pd.DataFrame({'Orbit': orbits, 'Predicted Payload (kg)': pred_arr.reshape(-1)})
                        else:
                            results_df = pd.DataFrame({'Prediction_Index': list(range(pred_arr.reshape(-1).size)), 'Value': pred_arr.reshape(-1)})
                        st.dataframe(results_df, use_container_width=True)
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button('Download predictions (CSV)', csv, file_name='payload_predictions_single.csv')

                    if model_path_info:
                        st.info(f'Using model file: {model_path_info}')

                except Exception as e:
                    st.error(f'Prediction failed: {e}')

with colB:
    if st.button('Predict Payload (Batch from CSV)'):
        if batch_df is None:
            st.error('Please upload a CSV with input rows first')
        else:
            # load model
            model_obj = None
            model_path_info = None
            if uploaded_model is not None:
                try:
                    model_obj = joblib.load(uploaded_model)
                    model_path_info = 'Uploaded model'
                except Exception as e:
                    st.error(f'Uploaded model could not be loaded: {e}')

            if model_obj is None:
                scraped = models_info.get(model_choice, {}).get(has_transfer, {})
                model_obj = scraped.get('model')
                model_path_info = scraped.get('path')

            if model_obj is None:
                st.error('No suitable model found. Upload a .pkl in the sidebar or place a model in results/models/.')
            else:
                try:
                    batch_aligned = align_features(batch_df.copy(), model_obj)
                    preds = model_obj.predict(batch_aligned)
                    preds_arr = np.asarray(preds)

                    # attach predictions to DataFrame
                    if preds_arr.ndim == 1:
                        batch_df['Predicted_Payload_kg'] = preds_arr
                    else:
                        # if multiple columns predicted, flatten appropriate
                        # try to add columns P0,P1,...
                        for i in range(preds_arr.shape[1]):
                            batch_df[f'Pred_{i}'] = preds_arr[:, i]

                    st.success('Batch prediction complete')
                    st.dataframe(batch_df.head(10))
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button('Download batch predictions (CSV)', csv, file_name='payload_batch_predictions.csv')

                    if model_path_info:
                        st.info(f'Using model file: {model_path_info}')

                except Exception as e:
                    st.error(f'Batch prediction failed: {e}')

# Footer note / CSV structure guidance
st.markdown('---')
st.header('CSV Structure & Notes')
st.write('The CSV used for batch predictions should have one row per rocket and column headers that match the internal feature names exactly (underscored). Example header names:')
example_cols = pd.DataFrame({'internal_name': FEATURE_COLUMNS, 'label': [HUMAN_LABELS[c] for c in FEATURE_COLUMNS]})
st.table(example_cols)
st.write('Missing columns will be filled with zeros automatically, but this may produce misleading results if crucial features are absent. Altitude (Altitude_km) should be in kilometers. Calculation error columns should be numeric; if not available set them to 0.')

# End of script
