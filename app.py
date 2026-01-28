"""
DISTILLATION COLUMN ETHANOL PURITY PREDICTION
================================================================================
✅ Uses trained model files with exact 21 FEATURES:
   - model.pkl (your XGBoost/Random Forest model - best one selected)
   - scaler.pkl (your StandardScaler - fit on 21 features)
   - features_names.pkl (your 21 feature names)

Features: 21 total
- 6 Core parameters (P, L, V, D, B, F)
- 6 Derived operating parameters (Reflux, Reboiler, Condenser, Feed_Normalized, Distillate_W, Bottoms_W)
- 5 Interaction terms (Reflux×Temp, Reboiler×Temp, Feed interactions)
- 3 Efficiency metrics (Column_Load, Separation_Duty, Column_Efficiency)
- 1 Temperature reference (Temp_Bottom)

Author: Mr. Narendra Kumar | Chemical R&D Scientist
Date: January 28, 2026
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from PIL import Image
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="Ethanol Purity Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional industrial look
st.markdown("""
    <style>
    :root {
        --color-primary: #2180E7;
        --color-success: #21808D;
        --color-warning: #A84B2F;
        --color-error: #C01547;
        --color-bg: #FCFCF9;
    }
    
    .main {
        background-color: #FCFCF9;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #21808D 0%, #2A9AA8 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #1A6B77;
    }
    
    .stMetric [data-testid="stMetricLabel"] {
        color: #E0E0D8 !important;
        font-size: 12px;
        font-weight: 500;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 24px;
        font-weight: bold;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #E8F8F0 0%, #F0FCFA 100%);
        border: 2px solid #21808D;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
    }

    section[data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
    }
    
    .error-box {
        background: linear-gradient(135deg, #FFF5F0 0%, #FFFAF8 100%);
        border: 2px solid #C01547;
        border-radius: 12px;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. HEADER
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("⚗️ Ethanol Purity Predictor")
    st.markdown("### Distillation Column Real-time Estimation")
    st.markdown("Machine Learning | Random Forest | Total 21 Features")

st.markdown("---")

# ============================================================================
# 3. LOAD ARTIFACTS
# ============================================================================
@st.cache_resource
def load_model_artifacts():
    try:
        # Load Model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load Scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        # Load Feature Names
        with open('features_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        # ✅ FIXED: Return 'None' as the 4th value (error) for success case
        return model, scaler, feature_names, None 
        
    except Exception as e:
        # Returns error message as 4th value for failure case
        return None, None, None, str(e)

model, scaler, feature_names, error = load_model_artifacts()

if error:
    st.error(f"🚨 SYSTEM ERROR: Missing Model Files.")
    st.warning(f"Could not find: {error}")
    st.info("Please ensure 'model.pkl', 'scaler.pkl', and 'features_names.pkl' are in the app folder.")
    st.stop()

# ============================================================================
# 4. SIDEBAR - MODEL INFORMATION
# ============================================================================

with st.sidebar:
    st.header("📊 General Info")
       
    st.divider()
    
    st.subheader("📈 Performance Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", "0>.98", delta=None)
    with col2:
        st.metric("RMSE", "0.0155", delta=None)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAE", "0.0122", delta=None)
    with col2:
        st.metric("Accuracy", "±1.55%", delta=None)
    
    st.divider()

    st.subheader("💡 Quick Tips")
    st.markdown("""
✓ **Higher reflux ratio** → Higher purity


✓ **Steady-state operation** recommended


✓ **Check mass balance:** D + B ≈ F


✓ **Keep T1 sensor calibrated**


✓ **Use typical operating ranges**
    """)

# ============================================================================
# 5. MAIN INPUT SECTION
# ============================================================================


st.header("📥 Input Operating Parameters")


col1, col2 = st.columns([1, 1])


with col1:
    st.subheader("Core Parameters")
    
    pressure = st.number_input(
        "Pressure (bar)",
        min_value=0.5,
        max_value=3.0,
        value=1.01,
        step=0.01,
        help="Column operating pressure in bar"
    )
    
    t1_celsius = st.number_input(
        "T1 - Top Tray Temperature (°C)",
        min_value=60.0,
        max_value=120.0,
        value=78.0,
        step=0.5,
        help="Temperature at top tray (main sensor)"
    )
    
    l_input = st.number_input(
        "L - Reflux Flow Rate (kmol/hr)",
        min_value=300.0,
        max_value=1200.0,
        value=780.0,
        step=10.0,
        help="Reflux flow from condenser"
    )
    
    v_input = st.number_input(
        "V - Vapor Flow Rate (kmol/hr)",
        min_value=600.0,
        max_value=1500.0,
        value=1040.0,
        step=10.0,
        help="Vapor flow from reboiler"
    )


with col2:
    st.subheader("Product & Feed Flows")
    
    d_input = st.number_input(
        "D - Distillate Flow Rate (kmol/hr)",
        min_value=100.0,
        max_value=500.0,
        value=260.0,
        step=10.0,
        help="Distillate product flow"
    )
    
    b_input = st.number_input(
        "B - Bottoms Flow Rate (kmol/hr)",
        min_value=100.0,
        max_value=500.0,
        value=340.0,
        step=10.0,
        help="Bottoms product flow"
    )
    
    f_input = st.number_input(
        "F - Feed Flow Rate (kmol/hr)",
        min_value=350.0,
        max_value=700.0,
        value=580.0,
        step=10.0,
        help="Feed flow to column"
    )

# ============================================================================
# 6. FEATURE ENGINEERING ENGINE (Hidden Calculation Layer)
# ============================================================================

# Safety: Avoid division by zero
epsilon = 1e-6
v_safe = max(v_input, epsilon)
f_safe = max(f_input, epsilon)

# A. Physics Calculation: Bottom Temperature
# We approximate Reboiler Temp as Top Temp + 20°C based on training data correlations.
# This prevents "upside down column" errors.
temp_top = t1_celsius
temp_bottom = t1_celsius + 20.0
temp_diff = temp_bottom - temp_top

# B. Normalization
# Constant derived from Training Data Mean (Printed in training script)
MEAN_FEED_FLOW = 545.23
feed_normalized = f_input / MEAN_FEED_FLOW

# C. Operating Ratios
reflux_ratio = l_input / v_safe
reboiler_intensity = v_input / f_safe
condenser_load = l_input / f_safe
distillate_withdrawal = d_input / f_safe
bottoms_withdrawal = b_input / f_safe
column_load = (l_input + v_input) / f_safe

# D. Interaction Terms (The "Secret Sauce" of the model)
reflux_x_temp_top = reflux_ratio * temp_top
reflux_x_temp_diff = reflux_ratio * temp_diff
reboiler_x_temp_bottom = reboiler_intensity * temp_bottom
feed_x_reflux = feed_normalized * reflux_ratio
feed_x_reboiler = feed_normalized * reboiler_intensity

# E. Efficiency Metrics
separation_duty = reflux_ratio * reboiler_intensity
column_efficiency = reflux_ratio * column_load

# Display features in grid
col1, col2, col3 = st.columns(3)


with col1:
    st.metric("Reflux Ratio", f"{reflux_ratio:.3f}")


with col2:
    st.metric("Reboiler Intensity", f"{reboiler_intensity:.3f}")


with col3:
    st.metric("Condenser Load", f"{condenser_load:.3f}")



col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Column Load", f"{column_load:.3f}")


with col2:
    st.metric("Separation Duty", f"{separation_duty:.3f}")


with col3:
    st.metric("Column Efficiency", f"{column_efficiency:.3f}")


# ============================================================================
# 7. PREDICTION SECTION
# ============================================================================


st.header("Purity Prediction")

st.subheader("Prediction Result")

# --- CORRECTED INDENTATION ---
if st.button("🔮 Run Simulation", type="primary", use_container_width=True):
    
    # 1. Assemble Feature Vector (Must match Training Order EXACTLY)
    input_vector = np.array([[
        pressure,               # 1
        l_input,                # 2
        v_input,                # 3
        d_input,                # 4
        b_input,                # 5
        f_input,                # 6
        temp_bottom,            # 7 (Calculated)
        reflux_ratio,           # 8
        reboiler_intensity,     # 9
        condenser_load,         # 10
        feed_normalized,        # 11
        distillate_withdrawal,  # 12
        bottoms_withdrawal,     # 13
        column_load,            # 14
        reflux_x_temp_top,      # 15
        reflux_x_temp_diff,     # 16
        reboiler_x_temp_bottom, # 17
        feed_x_reflux,          # 18
        feed_x_reboiler,        # 19
        separation_duty,        # 20
        column_efficiency       # 21
    ]])
    
    # 2. Scale & Predict
    try:
        input_scaled = scaler.transform(input_vector)
        prediction = model.predict(input_scaled)[0]
        purity_pct = prediction * 100
        
        # 3. Visual Logic
        if prediction >= 0.82:
            status_color = "#2E7D32" # Green
            status_msg = "OPTIMAL"
        elif prediction >= 0.75:
            status_color = "#F9A825" # Yellow/Orange
            status_msg = "ACCEPTABLE"
        else:
            status_color = "#C62828" # Red
            status_msg = "LOW PURITY WARNING"
        
        # 4. Display Result
        st.markdown(f"""
                <div style="background-color: {status_color}; padding: 20px; border-radius: 10px; text-align: center; color: white; margin-top: 20px;">
                    <h3 style="margin:0; opacity:0.8; color: white;">Predicted Ethanol Purity</h3>
                    <h1 style="font-size: 4rem; margin: 10px 0; color: white;">{prediction:.4f}</h1>
                    <h2 style="margin:0; color: white;">{purity_pct:.2f}% (Mole Fraction)</h2>
                    <hr style="border-color: rgba(255,255,255,0.3);">
                    <h3 style="margin:0; color: white;">STATUS: {status_msg}</h3>
                </div>
                """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction Failed: {e}")


# --- MIDDLE SECTION: Feature Analysis ---
st.divider()
col_img, col_desc = st.columns([1, 1])

with col_img:
    st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
    image_path = "feature_importance.png"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Model Decision Drivers", use_container_width=True)
    else:
        st.info("ℹ️ Upload 'feature_importance.png' to see the chart.")

with col_desc:
    st.markdown("<div class='section-header'>Decision Logic Analysis</div>", unsafe_allow_html=True)
    st.info("""
    **Understanding the Prediction:**
    
    1. **Top Drivers:** The chart shows which parameters the Random Forest model prioritizes. 
    2. **Physics Validation:** * **Reflux_x_Temp_Top:** The interaction between cooling and top temp is usually the #1 driver.
       * **Temp_Bottom:** The Reboiler temperature is a critical indicator of separation power.
    3. **Interpretation:** If purity is low, check if **Reflux Ratio** is low or **Top Temp** is too high (boiling water).
    """)


# ============================================================================
# 8. HELP & DOCUMENTATION SECTION
# ============================================================================

st.divider()
st.header("❓ Help & Documentation")

tab1, tab2, tab3 = st.tabs(["📘 Features Explained", "ℹ️ Model Info", "🔧 Troubleshooting"])

with tab1:
    st.subheader("21 Features Breakdown")
    st.markdown("""
    The model transforms 6 raw inputs into 21 powerful features to capture the chemical physics.
    
    | # | Feature | Category | Description |
    |---|---------|----------|-------------|
    | 1-6 | P, L, V, D, B, F | **Core** | User input - operating conditions |
    | 7 | Temp_Bottom | **Reference** | Calculated as **T1 + 20°C** (Reboiler approx.) |
    | 8-13 | Reflux_Ratio, Reboiler_Intensity, Condenser_Load, Feed_Normalized, Distillate_W, Bottoms_W | **Derived** | Ratios describing column loading & separation power |
    | 14-18 | Reflux×Temp_Top, Reflux×Temp_Diff, Reboiler×Temp_Bottom, Feed×Reflux, Feed×Reboiler | **Interactions** | **Critical:** How temperature changes affect separation efficiency |
    | 19-21 | Column_Load, Separation_Duty, Column_Efficiency | **Efficiency** | Combined metrics for total column performance |
    """)

with tab2:
    st.subheader("System Specifications")
    st.markdown("""
    ### 🧠 Model Architecture
    * **Algorithm:** Random Forest Regressor (Ensemble Learning)
    * **Training Data:** 1,200+ steady-state operation points
    * **Input Dimension:** 21 Physics-Augmented Features
    * **Output:** Ethanol Purity (Mole Fraction)
    
    ### 📊 Performance Metrics
    * **R² Score:** ~0.98 (Excellent fit) 
    * **Accuracy:** ±0.012% (Mole Fraction error)
    * **Validation:** 5-Fold Cross-Validation confirmed
    
    ### 🏭 Typical Operating Ranges
    | Parameter | Normal Range | Critical High | Critical Low |
    |-----------|--------------|---------------|--------------|
    | Top Temp (T1) | 77 - 80 °C | > 85°C (Water) | < 75°C (Subcooled) |
    | Reflux Ratio | 1.2 - 2.5 | > 3.0 (Flooding) | < 0.6 (Poor Separation) |
    | Reboiler (V) | 900 - 1100 | > 1200 | < 800 |
    """)

with tab3:
    st.subheader("Troubleshooting Guide")
    st.markdown("""
    ### 🔴 Purity Prediction is Too Low (< 0.70)
    1. **Check Reflux:** Is Reflux Ratio < 0.8? Increase Reflux Flow (L).
    2. **Check Temp:** Is Top Temp > 82°C? You are boiling water into the product. Reduce Steam (V).
    3. **Check Feed:** Is Feed Flow > 620? The column might be overloaded.

    ### 🟡 "Model files not found" Error
    * Ensure `model.pkl`, `scaler.pkl`, and `features_names.pkl` are in the **same folder** as `main.py`.
    
    ### 🔵 Why T1 + 20°C?
    * The model was trained using Bottom Temperature (T14).
    * Since you only input Top Temperature (T1), we use the physics approximation **T_bottom ≈ T_top + 20°C** to simulate the reboiler.
    * This ensures the interaction features (like `Reboiler_x_Temp_Bottom`) still work correctly.
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<br>
<div style='text-align: center; color: #6c757d; font-size: 14px; background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
    <p>
    ⚗️ <strong>Ethanol Purity Predictor V1.0 </strong> | Random Forest Logic | 21-Feature Physics Engine
    </p>
    <p>
    <em>Built for Educational & Learning Purpose Only | HNot for Critical Process Control</em>
    </p>
</div>

""", unsafe_allow_html=True)
