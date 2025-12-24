import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Atrial Fibrillation Recurrence Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title
st.title("❤️ Atrial Fibrillation Recurrence Prediction System")
st.markdown("Predict patient atrial fibrillation recurrence using TabPFN model")

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'probability' not in st.session_state:
    st.session_state.probability = None

# Feature definitions
con_feature_line = [
    'Age (years)',
    'LA(mm)',   
    "HBDH(U/L)",
    "LDH(U/L)",
    'SBP(mmHg)',
    'DBP(mmHg)',
    'BMI(kg/m2)',
    'LV(mm)',
    'RV(mm)',
    'RA(mm)',
    'Emv(m/s)',
    'IVS(mm)',
    'EDD(mm)',
    'EDV(ml)',
    'LAAeV(m/s)',
    'HDL-c(mmol/L)',
    "FBG(mmol/L)",
    "eGFR(ml/min/1.73m2)"
]

object_feature_line = [
    "Early Recurrence",
    'NPAF',
    'HT',
    "PAH(>=Mild)",
    "No Abnormalities on Echocardiogram",
    "MR",
    "TR",
    "LAA-SC Grade"
]

# Categorical feature options with detailed explanations
categorical_options = {
    "Early Recurrence": ["No", "Yes"],
    'NPAF': ["No", "Yes"],
    'HT': ["No", "Yes"],
    "PAH(>=Mild)": ["No", "Yes"],
    "No Abnormalities on Echocardiogram": ["No", "Yes"],
    "MR": ["0", "1", "2", "3", "4"],  # Direct input values
    "TR": ["0", "1", "2", "3", "4"],  # Direct input values
    "LAA-SC Grade": ["0", "1", "2", "3"]  # Direct input values
}

# Function to load model
@st.cache_resource
def load_model():
    """Load pre-trained model and preprocessors"""
    try:
        # Load scaler
        scaler = joblib.load('minmax_scaler.pkl')
        
        # Load OneHot encoder
        OHE = joblib.load('onehot_encoder.pkl')
        
        # Load feature information
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        # Load TabPFN model
        model_data = joblib.load('tabpfn.joblib')
        clf = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # Ensure feature_info contains feature_columns
        if 'feature_columns' not in feature_info:
            feature_info['feature_columns'] = feature_columns
            
        return scaler, OHE, feature_info, clf, feature_columns
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None, None, None, None

# Function to preprocess input data
def preprocess_input_data(df, scaler, OHE, feature_info):
    """Preprocess input data"""
    # Extract feature information
    con_feature_line = feature_info['con_feature_line']
    object_feature_line = feature_info['object_feature_line']
    onehot_columns = feature_info['onehot_columns']
    feature_columns = feature_info['feature_columns']
    
    # Ensure data only contains required feature columns
    needed_features = con_feature_line + object_feature_line
    df = df[needed_features].copy()
    
    # Separate numerical and categorical features
    X_num = df[con_feature_line].copy()
    X_cat = df[object_feature_line].copy()
    
    # Normalize numerical features
    if len(con_feature_line) > 0:
        X_num_scaled = scaler.transform(X_num)
        X_num_scaled_df = pd.DataFrame(X_num_scaled, columns=con_feature_line)
    else:
        X_num_scaled_df = pd.DataFrame()
    
    # OneHot encode categorical features
    if len(object_feature_line) > 0:
        X_cat_encoded = OHE.transform(X_cat)
        X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=onehot_columns)
    else:
        X_cat_encoded_df = pd.DataFrame()
    
    # Merge processed features
    X_processed = pd.concat([X_num_scaled_df, X_cat_encoded_df], axis=1)
    
    # Force ordering according to training feature order
    if feature_columns is not None:
        # Add missing feature columns (fill with 0)
        missing_cols = set(feature_columns) - set(X_processed.columns)
        for col in missing_cols:
            X_processed[col] = 0
        
        # Remove extra feature columns
        extra_cols = set(X_processed.columns) - set(feature_columns)
        for col in extra_cols:
            if col in X_processed.columns:
                X_processed = X_processed.drop(col, axis=1)
            
        # Reorder according to training order
        X_processed = X_processed.reindex(columns=feature_columns)
    
    return X_processed

# Function to make predictions
def make_prediction(input_data):
    """Make predictions"""
    scaler, OHE, feature_info, clf, feature_columns = load_model()
    
    if scaler is None:
        return None, None
    
    # Preprocess input data
    X_processed = preprocess_input_data(input_data, scaler, OHE, feature_info)
    
    # Ensure feature order consistency
    X_processed = X_processed.reindex(columns=feature_columns, fill_value=0)
    X_processed.columns = feature_columns
    
    # Convert to numpy array for prediction
    X_array = X_processed.values
    
    try:
        # Make predictions
        predictions = clf.predict(X_array)
        probabilities = clf.predict_proba(X_array)
        return predictions, probabilities
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

# Sidebar - Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select Function",
    ["Single Patient Prediction", "Batch Prediction", "Model Information", "Instructions"]
)

# Main application logic
if app_mode == "Single Patient Prediction":
    st.header("Single Patient Prediction")
    st.markdown("Please fill in the following patient information for prediction")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Clinical Information", 
        "📊 Echocardiography", 
        "🔬 Laboratory Tests", 
        "🏥 Special Examinations"
    ])
    
    # Dictionary to store all input data
    input_data = {}
    
    # Tab 1: Clinical Information
    with tab1:
        st.subheader("Clinical Information")
        col1, col2 = st.columns(2)
        
        with col1:
            input_data['Age (years)'] = st.number_input(
                'Age (years)', 
                min_value=0, 
                max_value=120, 
                value=60,
                help="Patient age in years"
            )
            
            input_data['NPAF'] = st.selectbox(
                'NPAF',
                categorical_options['NPAF'],
                help="Non-Paroxysmal Atrial Fibrillation"
            )
            
            input_data['SBP(mmHg)'] = st.number_input(
                'SBP(mmHg)', 
                min_value=60, 
                max_value=250, 
                value=120,
                help="Systolic Blood Pressure"
            )
            
            input_data['DBP(mmHg)'] = st.number_input(
                'DBP(mmHg)', 
                min_value=40, 
                max_value=150, 
                value=80,
                help="Diastolic Blood Pressure"
            )
        
        with col2:
            input_data['HT'] = st.selectbox(
                'HT',
                categorical_options['HT'],
                help="Hypertension"
            )
            
            input_data['Early Recurrence'] = st.selectbox(
                "Early Recurrence",
                categorical_options['Early Recurrence'],
                help="Early recurrence of arrhythmia"
            )
            
            input_data['BMI(kg/m2)'] = st.number_input(
                'BMI(kg/m2)', 
                min_value=10.0, 
                max_value=50.0, 
                value=24.0, 
                step=0.1,
                help="Body Mass Index"
            )
    
    # Tab 2: Echocardiography
    with tab2:
        st.subheader("Echocardiography Measurements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_data['LA(mm)'] = st.number_input(
                'LA(mm)', 
                value=40.0, 
                step=0.1,
                help="Left Atrium diameter"
            )
            
            input_data['LV(mm)'] = st.number_input(
                'LV(mm)', 
                value=50.0, 
                step=0.1,
                help="Left Ventricle diameter"
            )
            
            input_data['RV(mm)'] = st.number_input(
                'RV(mm)', 
                value=30.0, 
                step=0.1,
                help="Right Ventricle diameter"
            )
            
            input_data['RA(mm)'] = st.number_input(
                'RA(mm)', 
                value=40.0, 
                step=0.1,
                help="Right Atrium diameter"
            )
            
            input_data['Emv(m/s)'] = st.number_input(
                'Emv(m/s)', 
                value=0.1, 
                step=0.01,
                help="Early mitral annular velocity"
            )
        
        with col2:
            input_data['IVS(mm)'] = st.number_input(
                'IVS(mm)', 
                value=10.0, 
                step=0.1,
                help="Interventricular Septum thickness"
            )
            
            input_data['EDD(mm)'] = st.number_input(
                'EDD(mm)', 
                value=50.0, 
                step=0.1,
                help="End Diastolic Diameter"
            )
            
            input_data['EDV(ml)'] = st.number_input(
                'EDV(ml)', 
                value=120.0, 
                step=1.0,
                help="End Diastolic Volume"
            )
            
            input_data['LAAeV(m/s)'] = st.number_input(
                'LAAeV(m/s)', 
                value=0.5, 
                step=0.01,
                help="Left Atrial Appendage emptying velocity"
            )
        
        st.divider()
        st.subheader("Echocardiography Findings")
        
        col_find1, col_find2 = st.columns(2)
        
        with col_find1:
            input_data['PAH(>=Mild)'] = st.selectbox(
                "PAH(>=Mild)",
                categorical_options['PAH(>=Mild)'],
                help="Pulmonary Arterial Hypertension (≥Mild)"
            )
            
            input_data['No Abnormalities on Echocardiogram'] = st.selectbox(
                "No Abnormalities on Echocardiogram",
                categorical_options['No Abnormalities on Echocardiogram'],
                help="No abnormalities detected on echocardiogram"
            )
    
    # Tab 3: Laboratory Tests
    with tab3:
        st.subheader("Laboratory Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_data["HBDH(U/L)"] = st.number_input(
                "HBDH(U/L)", 
                value=150.0, 
                step=1.0,
                help="α-Hydroxybutyrate Dehydrogenase"
            )
            
            input_data["LDH(U/L)"] = st.number_input(
                "LDH(U/L)", 
                value=200.0, 
                step=1.0,
                help="Lactate Dehydrogenase"
            )
            
            input_data['HDL-c(mmol/L)'] = st.number_input(
                'HDL-c(mmol/L)', 
                min_value=0.0, 
                max_value=5.0, 
                value=1.5, 
                step=0.1,
                help="High-Density Lipoprotein Cholesterol"
            )
        
        with col2:
            input_data["FBG(mmol/L)"] = st.number_input(
                "FBG(mmol/L)", 
                min_value=0.0, 
                max_value=30.0, 
                value=5.5, 
                step=0.1,
                help="Fasting Blood Glucose"
            )
            
            input_data["eGFR(ml/min/1.73m2)"] = st.number_input(
                "eGFR(ml/min/1.73m2)", 
                min_value=0.0, 
                max_value=200.0, 
                value=90.0, 
                step=0.1,
                help="Estimated Glomerular Filtration Rate"
            )
    
    # Tab 4: Special Examinations
    with tab4:
        st.subheader("Special Examinations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # MR Grade input with explanation
            st.markdown("**Mitral Regurgitation (MR)**")
            mr_explanation = """
            **MR Grade Values:**
            - **4**: > Moderate regurgitation
            - **3**: Moderate regurgitation
            - **2**:  Mild -Moderate  regurgitation
            - **1**:  Mild regurgitation
            - **0**: No regurgitation 
            """
            st.caption(mr_explanation)
            input_data['MR'] = st.selectbox(
                "Select MR Grade",
                categorical_options['MR'],
                key="mr_grade"
            )
        
        with col2:
            # TR Grade input with explanation
            st.markdown("**Tricuspid Regurgitation (TR)**")
            tr_explanation = """
            **TR Grade Values:**
            - **4**: > Moderate regurgitation
            - **3**: Moderate regurgitation
            - **2**:  Mild -Moderate  regurgitation
            - **1**:  Mild regurgitation
            - **0**: No regurgitation 
            """
            st.caption(tr_explanation)
            input_data['TR'] = st.selectbox(
                "Select TR Grade",
                categorical_options['TR'],
                key="tr_grade"
            )
        
        with col3:
            # LAA-SC Grade input with explanation
            st.markdown("**LAA Spontaneous Contrast Grade**")
            laa_explanation = """
            **LAA-SC Grade Values:**
            - **3**: Moderate or Greater Spontaneous Contrast
            - **2**: Mild to moderate Spontaneous Contrast
            - **1**: Mild Spontaneous Contrast
            - **0**: No Spontaneous Contrast
            """
            st.caption(laa_explanation)
            input_data['LAA-SC Grade'] = st.selectbox(
                "Select LAA-SC Grade",
                categorical_options['LAA-SC Grade'],
                key="laa_grade"
            )
    
    # Prediction button
    if st.button("Start Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            # Create DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Make prediction
            predictions, probabilities = make_prediction(df_input)
            
            if predictions is not None:
                st.session_state.prediction_made = True
                st.session_state.prediction_result = predictions[0]
                st.session_state.probability = probabilities[0]
    
    # Display prediction results
    if st.session_state.prediction_made:
        st.divider()
        st.subheader("📊 Prediction Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                label="**Predicted Class**",
                value="High Risk" if st.session_state.prediction_result == 1 else "Low Risk",
                delta=None
            )
        
        with result_col2:
            risk_prob = st.session_state.probability[1] if st.session_state.probability is not None else 0
            st.metric(
                label="**Risk Probability**",
                value=f"{risk_prob:.1%}",
                delta=None
            )
        
        with result_col3:
            # Risk level
            if risk_prob >= 0.7:
                risk_level = "🔴 High Risk"
            elif risk_prob >= 0.4:
                risk_level = "🟡 Moderate Risk"
            else:
                risk_level = "🟢 Low Risk"
            st.metric(label="**Risk Level**", value=risk_level)
        
        # Risk interpretation
        st.info("💡 **Interpretation**: " + 
               ("This patient has high cardiac risk. Further evaluation and regular follow-up are recommended." 
                if st.session_state.prediction_result == 1 
                else "This patient currently has low cardiac risk. Maintaining a healthy lifestyle is recommended."))
        
        # Show probability details
        if st.session_state.probability is not None:
            with st.expander("View Detailed Probabilities"):
                prob_df = pd.DataFrame({
                    'Class': ['Low Risk', 'High Risk'],
                    'Probability': [f"{st.session_state.probability[0]:.2%}", 
                                   f"{st.session_state.probability[1]:.2%}"]
                })
                st.dataframe(prob_df, use_container_width=True)
        
        # Download results
        result_df = pd.DataFrame([{
            **input_data,
            'Predicted Class': 'High Risk' if st.session_state.prediction_result == 1 else 'Low Risk',
            'High Risk Probability': f"{st.session_state.probability[1]:.2%}" if st.session_state.probability is not None else "N/A",
            'Low Risk Probability': f"{st.session_state.probability[0]:.2%}" if st.session_state.probability is not None else "N/A"
        }])
        
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Prediction Results (CSV)",
            data=csv,
            file_name="cardiac_risk_prediction_result.csv",
            mime="text/csv",
            use_container_width=True
        )

elif app_mode == "Batch Prediction":
    st.header("Batch Prediction")
    st.markdown("Upload Excel or CSV file containing patient data for batch prediction")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['csv', 'xlsx', 'xls'],
        help="Please ensure the file contains all required feature columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Successfully read file! Total {len(df)} records")
            
            # Display data preview
            with st.expander("View Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Check for required features
            required_features = con_feature_line + object_feature_line
            missing_features = [f for f in required_features if f not in df.columns]
            
            if missing_features:
                st.warning(f"Following required features are missing: {missing_features}")
                st.info("Please ensure the file contains all required feature columns")
            else:
                if st.button("Start Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner("Processing batch prediction..."):
                        # Process data in batches
                        batch_size = 100
                        all_predictions = []
                        all_probabilities = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(0, len(df), batch_size):
                            batch_df = df.iloc[i:i+batch_size]
                            predictions, probabilities = make_prediction(batch_df)
                            
                            if predictions is not None:
                                all_predictions.extend(predictions)
                                if probabilities is not None:
                                    all_probabilities.extend(probabilities)
                            
                            # Update progress
                            progress = min((i + batch_size) / len(df), 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: {min(i + batch_size, len(df))}/{len(df)} records")
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if all_predictions:
                            # Add prediction results to original data
                            result_df = df.copy()
                            result_df['Predicted Class'] = ['High Risk' if p == 1 else 'Low Risk' for p in all_predictions]
                            
                            if all_probabilities:
                                result_df['High Risk Probability'] = [f"{p[1]:.2%}" for p in all_probabilities]
                                result_df['Low Risk Probability'] = [f"{p[0]:.2%}" for p in all_probabilities]
                            
                            # Show result statistics
                            st.subheader("Prediction Results Statistics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                high_risk_count = sum(all_predictions)
                                st.metric("High Risk Count", high_risk_count)
                            
                            with col2:
                                low_risk_count = len(all_predictions) - high_risk_count
                                st.metric("Low Risk Count", low_risk_count)
                            
                            with col3:
                                high_risk_rate = high_risk_count / len(all_predictions)
                                st.metric("High Risk Rate", f"{high_risk_rate:.1%}")
                            
                            # Show result table
                            st.subheader("Prediction Results Details")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Download button
                            output_format = st.radio("Select Download Format", ["Excel", "CSV"], horizontal=True)
                            
                            if output_format == "Excel":
                                # Convert to Excel
                                output = result_df.to_excel(index=False)
                                st.download_button(
                                    label="Download Full Results (Excel)",
                                    data=output,
                                    file_name="batch_prediction_results.xlsx",
                                    mime="application/vnd.ms-excel",
                                    use_container_width=True
                                )
                            else:
                                # Convert to CSV
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Full Results (CSV)",
                                    data=csv,
                                    file_name="batch_prediction_results.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        else:
                            st.error("Prediction failed. Please check data format.")
        
        except Exception as e:
            st.error(f"Failed to read file: {str(e)}")

elif app_mode == "Model Information":
    st.header("Model Information")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Feature Information")
        st.write(f"**Continuous Features**: {len(con_feature_line)}")
        st.write(f"**Categorical Features**: {len(object_feature_line)}")
        st.write(f"**Total Features**: {len(con_feature_line) + len(object_feature_line)}")
    
    with col2:
        st.subheader("🤖 Model Information")
        st.write("**Model Type**: TabPFN (Tabular Prior-Data Fitted Network)")
        st.write("**Prediction Target**: Cardiac Risk Level")
        st.write("**Output Classes**: Low Risk (0), High Risk (1)")
    
    st.divider()
    
    # Feature details
    with st.expander("View All Features and Descriptions"):
        st.write("**Continuous Features**:")
        for feature in con_feature_line:
            st.write(f"- {feature}")
        
        st.write("\n**Categorical Features**:")
        for feature in object_feature_line:
            st.write(f"- {feature}")
    
    # Special examination values explanation
    with st.expander("Special Examination Value Definitions"):
        st.markdown("""
        ### Mitral Regurgitation (MR) Grade:
        - **4**: > Moderate (Moderate以上反流)
        - **3**: ≥ Moderate (中度反流)
        - **2**: > Mild (轻-中度反流)
        - **1**: ≥ Mild (轻度反流)
        - **0**: No regurgitation (无反流)
        
        ### Tricuspid Regurgitation (TR) Grade:
        - **4**: > Moderate (Moderate以上反流)
        - **3**: ≥ Moderate (中度反流)
        - **2**: > Mild (轻-中度反流)
        - **1**: ≥ Mild (轻度反流)
        - **0**: No regurgitation (无反流)
        
        ### LAA Spontaneous Contrast (LAA-SC) Grade:
        - **3**: Moderate or greater (中度或以上自发显影)
        - **2**: Mild to moderate (轻-中度自发显影)
        - **1**: Mild (轻度自发显影)
        - **0**: None (无自发显影)
        """)
    
    # Model file check
    st.subheader("🔍 Model File Status")
    required_files = ['minmax_scaler.pkl', 'onehot_encoder.pkl', 'feature_info.pkl', 'tabpfn.joblib']
    
    file_status = {}
    for file in required_files:
        file_status[file] = os.path.exists(file)
    
    for file, exists in file_status.items():
        if exists:
            st.success(f"✓ {file} - Loaded")
        else:
            st.error(f"✗ {file} - Missing")

elif app_mode == "Instructions":
    st.header("Instructions")
    
    st.markdown("""
    ## 📚 Application Usage Guide
    
    ### 1. Single Patient Prediction
    - Select "Single Patient Prediction" from the left sidebar
    - Fill in all patient information in four organized tabs:
        1. **Clinical Information**: Basic patient demographics and history
        2. **Echocardiography**: Cardiac measurements and findings
        3. **Laboratory Tests**: Blood test results
        4. **Special Examinations**: Direct input of MR, TR, and LAA-SC grades
    - Click "Start Prediction" button
    - View prediction results and risk probabilities
    
    ### 2. Batch Prediction
    - Select "Batch Prediction" from the left sidebar
    - Upload Excel or CSV file containing patient data
    - Ensure file contains all required feature columns with correct values
    - System will perform batch prediction automatically
    - Download complete prediction results
    
    ### 3. Data Requirements
    - **File Format**: CSV or Excel
    - **Feature Order**: No special requirement, system will automatically match
    - **Data Format**:
        - Continuous Features: Numeric type
        - Categorical Features: Use predefined option values
    
    ### 4. Special Examination Value Definitions
    
    **Mitral Regurgitation (MR) Grade Values:**
    - **4**: > Moderate regurgitation
    - **3**:  Moderate regurgitation
    - **2**:  Mild - Moderate regurgitation
    - **1**:  Mild   regurgitation
     - **0**: No regurgitation 
    
    **Tricuspid Regurgitation (TR) Grade Values:**
    - **4**: > Moderate regurgitation
    - **3**:  Moderate regurgitation
    - **2**:  Mild - Moderate regurgitation
    - **1**:  Mild   regurgitation  
     - **0**: No regurgitation 
    
    **LAA Spontaneous Contrast (LAA-SC) Grade Values:**
    - **3**: Moderate or greater Spontaneous Contrast
    - **2**: Mild to moderate Spontaneous Contrast
    - **1**: Mild Spontaneous Contrast
    - **0**: No Spontaneous Contrast    
    ### 5. Result Interpretation
    - **Low Risk (0)**: Prediction probability < 50%
    - **High Risk (1)**: Prediction probability ≥ 50%
    - **Risk Level**:
        - 🟢 Low Risk: Probability < 40%
        - 🟡 Moderate Risk: 40% ≤ Probability < 70%
        - 🔴 High Risk: Probability ≥ 70%
    """)

# Footer
st.divider()
st.caption("© 2023 Atrial Fibrillation Recurrence Prediction System | For medical professional reference only")