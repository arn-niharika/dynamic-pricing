"""
Universal Prediction Model Deployment - Streamlit App
Upload any pickle file containing a trained model and use it for predictions.
Supports any regression or classification model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
import traceback

# ============================================================================
# Page Configuration
# ============================================================================

# HOTFIX: Handle pickles with 'LabelEncoder' module reference
try:
    import sys
    import numpy as np
    # Only inject if not already present or if it's the wrong one
    if 'LabelEncoder' not in sys.modules:
        try:
            import sklearn.preprocessing
            class MockLabelEncoderModule:
                pass
            mock_le = MockLabelEncoderModule()
            mock_le.LabelEncoder = sklearn.preprocessing.LabelEncoder
            mock_le.dtype = np.dtype
            sys.modules['LabelEncoder'] = mock_le
        except ImportError:
            pass # Sklearn maybe not installed
except Exception as e:
    pass

st.set_page_config(
    page_title="Dynamic Bus Pricing Predictor",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS for Modern UI
# ============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Card-like containers */
    .stApp {
        background: transparent;
    }
    
    /* Header styling */
    h1 {
        color: #000000 !important;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.3rem;
      
    
    h2 {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #e0e7ff;
        font-weight: 500;
        font-size: 1.3rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e7ff;
    }
    
    /* Input containers */
    .stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stCheckbox {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stTextInput:hover, .stNumberInput:hover, .stSelectbox:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Labels */
    label {
        color: #1e1e2e !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        border-left: 4px solid #667eea;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Success message */
    .element-container:has(.stSuccess) {
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 2rem 0;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed rgba(255,255,255,0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(255,255,255,0.6);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white !important;
        font-weight: 500;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Caption text */
    .stCaption {
        color: rgba(255,255,255,0.8) !important;
        font-size: 0.85rem !important;
        font-style: italic;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card effect for columns */
    .element-container {
        animation: fadeIn 0.6s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Result card */
    .result-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin: 2rem 0;
        animation: scaleIn 0.5s ease;
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Checkbox */
    .stCheckbox {
        background: transparent !important;
    }
    
    /* Select box dropdown */
    [data-baseweb="select"] {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# Load Model from Uploaded File
# ============================================================================

def load_model_from_file(uploaded_file):
    """Load model from uploaded pickle file"""
    try:
        model = pickle.load(uploaded_file)
        return model, None
    except Exception as e:
        return None, str(e)

def load_encoders(uploaded_file):
    """Load encoders from uploaded pickle file"""
    try:
        encoders = pickle.load(uploaded_file)
        
        # Validation: Ensure it's a dictionary
        if not isinstance(encoders, dict):
             # Heuristic: if it's an array, it might be a single encoder or classes
             return None, f"Expected a dictionary {{'feature': encoder}}, but got {type(encoders).__name__}. You might have uploaded a single encoder or just an array of classes."
             
        return encoders, None
    except Exception as e:
        return None, str(e)

def get_model_info(model):
    """Extract basic info about the model"""
    info = {
        "type": type(model).__name__,
        "module": type(model).__module__,
        "has_predict": hasattr(model, 'predict'),
        "has_predict_proba": hasattr(model, 'predict_proba'),
        "has_feature_names": hasattr(model, 'feature_names_in_'),
    }
    
    # Try to get feature names
    if hasattr(model, 'feature_names_in_'):
        info["feature_names"] = list(model.feature_names_in_)
    elif hasattr(model, 'feature_names'):
        info["feature_names"] = list(model.feature_names)
    
    # Try to get number of features
    if hasattr(model, 'n_features_in_'):
        info["n_features"] = model.n_features_in_
    elif hasattr(model, 'feature_names_in_'):
        info["n_features"] = len(model.feature_names_in_)
    
    return info

# ============================================================================
# Feature Engineering and Preparation
# ============================================================================

def engineer_features(df):
    """
    Apply the same feature engineering as the notebook.
    """
    df = df.copy()
    
    # ─── Temporal Patterns ──────────────────────────────────────────────
    if 'journey_weekday' in df.columns:
        df['journey_is_weekend'] = df['journey_weekday'].isin([5, 6]).astype(int)
    
    if 'departure_hour' in df.columns:
        # Night departure (8 PM - 5 AM)
        df['is_night_departure'] = ((df['departure_hour'] >= 20) | (df['departure_hour'] <= 5)).astype(int)
        # Peak hours (6-9 AM, 5-8 PM)
        df['is_peak_hour'] = (df['departure_hour'].isin([6,7,8,9,17,18,19,20])).astype(int)
    
    # ─── Booking Window ─────────────────────────────────────────────────
    if 'hours_to_departure' in df.columns:
        df['is_last_minute'] = (df['hours_to_departure'] <= 6).astype(int)
        df['is_advance_booking'] = (df['hours_to_departure'] >= 168).astype(int)  # 7+ days
    
    # ─── Demand & Scarcity Signals ──────────────────────────────────────
    if 'available_seats' in df.columns:
        df['low_availability'] = (df['available_seats'] <= 5).astype(int)
        df['very_low_availability'] = (df['available_seats'] <= 2).astype(int)
        # Use dynamic capacity if available, else default to 50
        total_capacity = df.get('total_capacity', 50)
        df['seats_sold_ratio'] = (1 - (df['available_seats'] / total_capacity).clip(upper=1))
    
    # ─── Seat Characteristics ───────────────────────────────────────────
    if 'seat_is_upper' in df.columns:
        df['is_lower_berth'] = (~df['seat_is_upper'].astype(bool)).astype(int)
    
    if 'window_seats' in df.columns and 'seat_is_upper' in df.columns:
        df['is_premium_seat'] = ((~df['seat_is_upper'].astype(bool)) & (df['window_seats'] > 0)).astype(int)
    
    # ─── Bus Type Features ──────────────────────────────────────────────
    bus_source = None
    if 'bus_type' in df.columns:
        bus_source = df['bus_type']
    elif 'bus_type_le' in df.columns:
        bus_source = df['bus_type_le']

    if bus_source is not None:
        bus_type_lower = bus_source.astype(str).str.lower().fillna('')
        df['is_volvo'] = bus_type_lower.str.contains('volvo').astype(int)
        df['is_sleeper'] = bus_type_lower.str.contains('sleeper').astype(int)
        df['is_seater'] = bus_type_lower.str.contains('seater').astype(int)
        df['is_multi_axle'] = bus_type_lower.str.contains('multi|axle').astype(int)
        df['is_AC'] = bus_type_lower.str.contains('ac').astype(int)
        # Fix mutual exclusivity for seater/sleeper if both found (prioritize sleeper if it's a mix or adjust logic)
        # For now, simplistic string matching as per notebook is fine.

    return df

def prepare_input_data(input_dict, feature_names, encoders=None):
    """
    Convert input dictionary to DataFrame with required features.
    Automatically handles missing features by creating sensible defaults.
    """
    df = pd.DataFrame([input_dict])
    
    
    # Apply automatic feature engineering
    df = engineer_features(df)
    
    # Ensure all required features exist (fill remaining with defaults)
    for feature in feature_names:
        if feature not in df.columns:
            # Try to infer default values based on feature name
            if 'age' in feature.lower():
                df[feature] = 0
            elif 'price' in feature.lower() or 'cost' in feature.lower():
                df[feature] = 0
            elif 'count' in feature.lower() or 'number' in feature.lower() or 'quantity' in feature.lower():
                df[feature] = 0
            elif 'is_' in feature.lower() or 'has_' in feature.lower():
                df[feature] = 0
            elif 'ratio' in feature.lower() or 'percent' in feature.lower():
                df[feature] = 0.5
            else:
                df[feature] = 0  # Default to 0 for unknown numeric features
    
    # Apply Encoders if available
    if encoders:
        for feature in feature_names:
            if feature.endswith('_le'):
                base_feature = feature[:-3] # Remove '_le'
                if base_feature in encoders:
                    le = encoders[base_feature]
                    # Get the current value which is likely a string
                    display_val = df.iloc[0][feature]
                    try:
                        # Transform
                        # Note: LabelEncoder expects a list/array
                        encoded_val = le.transform([str(display_val)])[0]
                        df.at[0, feature] = encoded_val
                    except Exception:
                        # If value not seen in training, assign a default (e.g. 0 or unknown)
                        # For now, we'll try to use 0 or verify if there's a better fallback
                        st.warning(f"Value '{display_val}' for '{base_feature}' was not seen in training. using 0.")
                        df.at[0, feature] = 0
            # Also handle case where model might use non-le named features but encoding is needed
            # (less likely given the notebook structure, but good for safety)
            elif feature in encoders:
                 le = encoders[feature]
                 display_val = df.iloc[0][feature]
                 try:
                    df.at[0, feature] = le.transform([str(display_val)])[0]
                 except:
                    df.at[0, feature] = 0
                                
    # Make sure all data is numeric for XGBoost/LightGBM (unless native cat support is enabled)
    # The error message specifically mentioned object columns were the issue.
    # We should convert the dataframe to numeric, coercing errors.
    # But only after we tried encoding.
    
    # Select only required features in correct order
    
    # Select only required features in correct order
    X = df[feature_names].copy()
    
    # Force conversion to numeric to ensure no object types remain
    # XGBoost raises error if object types are present
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X

# ============================================================================
# Main Streamlit App
# ============================================================================

def main():
    # Inject custom CSS
    inject_custom_css()
    
    # Hero Section
    st.markdown("""
        <div style='text-align: center; padding: 0.5rem 0 1rem 0;'>
            <h1 style='font-size: 2.5rem; color: #000000;'>
                🚌 Dynamic Bus Pricing
            </h1>
            <p style='color: #000000; font-size: 1.1rem; margin-top: 0.3rem;'>
                Predict bus ticket prices with AI-powered precision
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model upload
    with st.sidebar:
        st.markdown("### 📦 Model Configuration")
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Upload Trained Model",
            type=["pkl", "pickle"],
            help="Upload your trained model in pickle format"
        )
        
        uploaded_encoder = st.file_uploader(
            "Upload Encoders (Optional)",
            type=["pkl", "pickle"],
            help="Upload the categorical_encoders.pkl file if your model uses encoded features"
        )
        
        if uploaded_file is not None:
            st.success("✅ Model loaded successfully!")
    
    # Main content
    if uploaded_file is None:
        # Welcome screen with better structure
        st.markdown("""
            <div style='background: rgba(255,255,255,0.95); border-radius: 20px; padding: 2rem; margin: 2rem 0; box-shadow: 0 10px 40px rgba(0,0,0,0.2);'>
                <h2 style='color: #667eea; text-align: center; margin-bottom: 2rem;'>Welcome to the Bus Pricing Predictor</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Step-by-step guide using Streamlit columns
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div style='background: rgba(255,255,255,0.9); border-radius: 15px; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 250px;'>
                    <div style='font-size: 3.5rem; margin-bottom: 1rem;'>📤</div>
                    <h3 style='color: #764ba2; font-size: 1.2rem; margin-bottom: 0.5rem;'>Step 1: Upload Model</h3>
                    <p style='color: #666; font-size: 0.95rem;'>Select your trained model file from the sidebar</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style='background: rgba(255,255,255,0.9); border-radius: 15px; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 250px;'>
                    <div style='font-size: 3.5rem; margin-bottom: 1rem;'>✍️</div>
                    <h3 style='color: #764ba2; font-size: 1.2rem; margin-bottom: 0.5rem;'>Step 2: Enter Details</h3>
                    <p style='color: #666; font-size: 0.95rem;'>Fill in journey and bus information</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div style='background: rgba(255,255,255,0.9); border-radius: 15px; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 250px;'>
                    <div style='font-size: 3.5rem; margin-bottom: 1rem;'>🎯</div>
                    <h3 style='color: #764ba2; font-size: 1.2rem; margin-bottom: 0.5rem;'>Step 3: Get Prediction</h3>
                    <p style='color: #666; font-size: 0.95rem;'>Receive instant price predictions</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Features section
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background: rgba(255,255,255,0.95); border-radius: 20px; padding: 2.5rem; margin: 2rem 0; box-shadow: 0 10px 40px rgba(0,0,0,0.2);'>
                <h3 style='color: #667eea; margin-bottom: 1.5rem; font-size: 1.5rem;'>✨ Features</h3>
                <div style='color: #666; line-height: 2.2; font-size: 1.05rem;'>
                    🤖 <strong>AI-powered price predictions</strong> - Advanced machine learning models<br>
                    📊 <strong>Smart feature engineering</strong> - Automatic calculation of derived features<br>
                    🎨 <strong>Beautiful, intuitive interface</strong> - Modern design for better experience<br>
                    ⚡ <strong>Real-time calculations</strong> - Instant predictions with no delays<br>
                    🔒 <strong>Secure and private</strong> - Your data stays on your machine
                </div>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Load the model
    model, error = load_model_from_file(uploaded_file)
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        return

    # Load encoders if provided
    encoders = None
    if uploaded_encoder:
        encoders, enc_error = load_encoders(uploaded_encoder)
        if enc_error:
            st.sidebar.warning(f"⚠️ Could not load encoders: {enc_error}")
            st.sidebar.info("ℹ️ App will work without encoders! You'll enter numeric codes directly.")
        else:
            st.sidebar.success(f"✅ Loaded {len(encoders)} encoders")
    else:
        st.sidebar.info("ℹ️ No encoders uploaded. Using default encoding.")
    
    
    # Get model information
    model_info = get_model_info(model)
    
    # Display model information in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                <p style='margin: 0.5rem 0; color: black;'><strong>Type:</strong> {model_info['type']}</p>
                <p style='margin: 0.5rem 0; color: black;'><strong>Library:</strong> {model_info['module'].split('.')[0]}</p>
                <p style='margin: 0.5rem 0; color: black;'><strong>Features:</strong> {model_info.get('n_features', 'N/A')}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Get feature names
    feature_names = model_info.get("feature_names", [])

    # Check for missing encoders
    if encoders is None and any(f.endswith('_le') for f in feature_names):
        st.warning("⚠️ Model uses encoded features (ending in _le) but no encoders were uploaded. These features will default to 0, which may affect prediction accuracy. Please upload 'categorical_encoders.pkl'.")
    
    if not feature_names:
        st.warning("⚠️ Could not automatically detect feature names from the model.")
        st.markdown("**Please enter feature names manually (comma-separated)**")
        manual_features = st.text_input(
            "Feature names",
            help="Enter feature names separated by commas"
        )
        
        if manual_features:
            feature_names = [f.strip() for f in manual_features.split(",")]
            st.success(f"Using {len(feature_names)} features")
        else:
            st.info("👈 Please provide feature names to continue")
            return
    
    # ===== INPUT SECTION =====
    st.markdown("""
        <div style='background: rgba(255,255,255,0.95); border-radius: 20px; padding: 2rem; margin: 2rem 0; box-shadow: 0 10px 40px rgba(0,0,0,0.2);'>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>📝 Journey Details</h2>
            <p style='color: #666;'>Fill in the information below to get your price prediction</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature descriptions and help text
    feature_descriptions = {
        # Seat features
        'seat_is_seater': 'Regular seating (like airplane) vs bed/sleeper (like train)',
        'seat_is_upper': 'Upper bunk (top bed) vs lower bunk (bottom bed)',
        'seat_is_ladies': 'Reserved for women vs general seat',
        'seat_is_horizontal': 'Lying down (horizontal bed) vs sitting upright',
        'seat_is_available': 'Available to book vs already sold',
        'seat_name': 'Your seat code (e.g., SL1, U5, L3)',
        'seat_name_le': 'Select your seat from the dropdown',
        
        # Journey features
        'journey_weekday': 'Day of the week for your journey',
        'departure_hour': 'What time does the bus leave?',
        'hours_to_departure': 'How many hours until departure?',
        'days_to_journey': 'How many days in advance are you booking?',
        'duration_hours': 'How long is the journey?',
        'scrape_hour': 'Current hour (for booking time analysis)',
        
        # Bus features
        'operator_name': 'Bus company/operator',
        'operator_name_le': 'Select bus company',
        'bus_type': 'Type of bus (AC/Non-AC, Sleeper/Seater)',
        'bus_type_le': 'Select bus type',
        'source_collection': 'Route (e.g., Hyderabad-Bangalore)',
        'source_collection_le': 'Select route',
        
        # Availability features
        'available_seats': 'Number of seats still available',
        'window_seats': 'Number of window seats available',
    }
    
    # Predefined options
    seat_names = [
                "0",
                "01LW",
                "02L",
                "03L",
                "03UW",
                "04LW",
                "05L",
                "06L",
                "06UW",
                "07LW",
                "08L",
                "09L",
                "09UW",
                "1",
                "10",
                "10ALB",
                "10AUB",
                "10B",
                "10BLB",
                "10BUB",
                "10C",
                "10L",
                "10LW",
                "10U",
                "10UW",
                "10W",
                "11",
                "11ALB",
                "11AUB",
                "11B",
                "11BLB",
                "11BUB",
                "11C",
                "11L",
                "11U",
                "11W",
                "12",
                "12ALB",
                "12AUB",
                "12B",
                "12BLB",
                "12BUB",
                "12C",
                "12L",
                "12U",
                "12UW",
                "12W",
                "13",
                "13ALB",
                "13AUB",
                "13BLB",
                "13BUB",
                "13L",
                "13LW",
                "13U",
                "13UW",
                "13W",
                "14",
                "14L",
                "14U",
                "14W",
                "15",
                "15L",
                "15U",
                "15UW",
                "15W",
                "16",
                "16L",
                "16LW",
                "16U",
                "16UW",
                "16W",
                "17",
                "17L",
                "17U",
                "17W",
                "18",
                "18L",
                "18U",
                "18UW",
                "18W",
                "19",
                "19L",
                "19W",
                "1A",
                "1B",
                "1C",
                "1D",
                "1E",
                "1F",
                "1G",
                "1L",
                "1LA",
                "1LB",
                "1LC",
                "1LDA",
                "1LDB",
                "1LS",
                "1U",
                "1UA",
                "1UB",
                "1UC",
                "1UDA",
                "1UDB",
                "1US",
                "1UW",
                "1W",
                "2",
                "20",
                "20L",
                "20W",
                "21",
                "21L",
                "21W",
                "22",
                "22U",
                "22W",
                "23",
                "23U",
                "23W",
                "24",
                "24U",
                "24W",
                "25",
                "25L",
                "26",
                "26L",
                "26W",
                "27",
                "27L",
                "27W",
                "28",
                "28U",
                "29",
                "29U",
                "2A",
                "2B",
                "2C",
                "2D",
                "2E",
                "2F",
                "2G",
                "2L",
                "2LA",
                "2LB",
                "2LC",
                "2LDA",
                "2LDB",
                "2LS",
                "2U",
                "2UA",
                "2UB",
                "2UC",
                "2UDA",
                "2UDB",
                "2US",
                "2W",
                "3",
                "30",
                "30U",
                "30W",
                "31",
                "31L",
                "31W",
                "32",
                "32L",
                "33",
                "33L",
                "34",
                "34U",
                "34W",
                "35",
                "35U",
                "35W",
                "36",
                "36U",
                "37",
                "38",
                "39",
                "39W",
                "3A",
                "3B",
                "3C",
                "3D",
                "3E",
                "3F",
                "3G",
                "3L",
                "3LA",
                "3LB",
                "3LC",
                "3LDA",
                "3LDB",
                "3LS",
                "3U",
                "3UA",
                "3UB",
                "3UC",
                "3UDA",
                "3UDB",
                "3US",
                "3W",
                "4",
                "40",
                "41",
                "42",
                "43",
                "44",
                "45",
                "4A",
                "4B",
                "4C",
                "4D",
                "4E",
                "4F",
                "4G",
                "4L",
                "4LA",
                "4LB",
                "4LC",
                "4LDA",
                "4LDB",
                "4LS",
                "4U",
                "4UA",
                "4UB",
                "4UC",
                "4UDA",
                "4UDB",
                "4US",
                "4UW",
                "4W",
                "5",
                "5A",
                "5B",
                "5C",
                "5D",
                "5E",
                "5F",
                "5G",
                "5L",
                "5LA",
                "5LB",
                "5LC",
                "5LDA",
                "5LDB",
                "5LS",
                "5U",
                "5UA",
                "5UB",
                "5UC",
                "5UDA",
                "5UDB",
                "5US",
                "5W",
                "6",
                "6A",
                "6ALB",
                "6AUB",
                "6B",
                "6BLB",
                "6BUB",
                "6C",
                "6D",
                "6E",
                "6F",
                "6G",
                "6L",
                "6LA",
                "6LB",
                "6LC",
                "6LDA",
                "6LDB",
                "6LS",
                "6U",
                "6UA",
                "6UB",
                "6UC",
                "6UDA",
                "6UDB",
                "6US",
                "6W",
                "7",
                "7ALB",
                "7AUB",
                "7B",
                "7BLB",
                "7BUB",
                "7C",
                "7L",
                "7LB",
                "7LDA",
                "7LDB",
                "7LS",
                "7U",
                "7UB",
                "7UDA",
                "7UDB",
                "7US",
                "7UW",
                "7W",
                "8",
                "8ALB",
                "8AUB",
                "8B",
                "8BLB",
                "8BUB",
                "8C",
                "8L",
                "8U",
                "8W",
                "9",
                "9ALB",
                "9AUB",
                "9B",
                "9BLB",
                "9BUB",
                "9C",
                "9L",
                "9U",
                "9W",
                "A",
                "A1",
                "A10",
                "A11",
                "A12",
                "A13",
                "A1K1",
                "A1Q1",
                "A2",
                "A2K7",
                "A2Q7",
                "A3",
                "A4",
                "A5",
                "A6",
                "A7",
                "A8",
                "A9",
                "B",
                "B1",
                "B10",
                "B11",
                "B12",
                "B13",
                "B1K2",
                "B1Q2",
                "B2",
                "B2K8",
                "B2Q8",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B9",
                "C",
                "C1",
                "C10",
                "C11",
                "C12",
                "C13",
                "C1K3",
                "C1Q3",
                "C2",
                "C2K9",
                "C2Q9",
                "C3",
                "C4",
                "C5",
                "C6",
                "C7",
                "C8",
                "C9",
                "D",
                "D1",
                "D10",
                "D11",
                "D12",
                "D13",
                "D1K4",
                "D1Q4",
                "D2",
                "D2K10",
                "D2Q10",
                "D3",
                "D4",
                "D5",
                "D6",
                "D7",
                "D8",
                "D9",
                "DL1",
                "DL10",
                "DL11",
                "DL12",
                "DL13",
                "DL14",
                "DL2",
                "DL3",
                "DL4",
                "DL5",
                "DL6",
                "DL7",
                "DL8",
                "DL9",
                "DU1",
                "DU10",
                "DU11",
                "DU12",
                "DU13",
                "DU14",
                "DU2",
                "DU3",
                "DU4",
                "DU5",
                "DU6",
                "DU7",
                "DU8",
                "DU9",
                "E",
                "E1",
                "E1K5",
                "E1Q5",
                "E2",
                "E2K11",
                "E2Q11",
                "E3",
                "E4",
                "E5",
                "E6",
                "F",
                "F1",
                "F1K6",
                "F1Q6",
                "F2",
                "F2K12",
                "F2Q12",
                "F3",
                "F4",
                "F5",
                "F6",
                "G",
                "G1",
                "G2",
                "G3",
                "G4",
                "G5",
                "G6",
                "H",
                "H1",
                "H2",
                "H3",
                "H4",
                "I",
                "I1",
                "I2",
                "I3",
                "I4",
                "I5",
                "J",
                "J1",
                "J12",
                "J13",
                "J2",
                "J3",
                "J4",
                "J5",
                "K",
                "K1",
                "K2",
                "K3",
                "K4",
                "L",
                "L1",
                "L10",
                "L10W",
                "L11",
                "L12",
                "L12W",
                "L13",
                "L14",
                "L15",
                "L16",
                "L17",
                "L18",
                "L19",
                "L2",
                "L20",
                "L21",
                "L22",
                "L23",
                "L24",
                "L25",
                "L26",
                "L27",
                "L28",
                "L29",
                "L2W",
                "L3",
                "L30",
                "L31",
                "L32",
                "L33",
                "L35",
                "L37",
                "L38",
                "L39",
                "L4",
                "L41",
                "L4W",
                "L5",
                "L6",
                "L6W",
                "L7",
                "L8",
                "L8W",
                "L9",
                "LD1",
                "LD10",
                "LD11",
                "LD12",
                "LD13",
                "LD14",
                "LD15",
                "LD16",
                "LD17",
                "LD18",
                "LD19",
                "LD2",
                "LD20",
                "LD21",
                "LD2W",
                "LD3",
                "LD4",
                "LD4W",
                "LD5",
                "LD6",
                "LD6W",
                "LD7",
                "LD8",
                "LD8W",
                "LD9",
                "LS1",
                "LS2",
                "LS3",
                "LS4",
                "LS5",
                "LU1",
                "LU2",
                "LU3",
                "LU4",
                "LU5",
                "M",
                "M1",
                "M2",
                "M3",
                "N",
                "N1",
                "N2",
                "O",
                "P",
                "Q",
                "R",
                "R1",
                "R10",
                "R11",
                "R12",
                "R13",
                "R14",
                "R15",
                "R16",
                "R17",
                "R18",
                "R19",
                "R2",
                "R20",
                "R21",
                "R22",
                "R23",
                "R24",
                "R25",
                "R26",
                "R3",
                "R4",
                "R5",
                "R6",
                "R7",
                "R8",
                "R9",
                "RL1",
                "RL10",
                "RL11",
                "RL12",
                "RL2",
                "RL3",
                "RL4",
                "RL5",
                "RL6",
                "RL7",
                "RL8",
                "RL9",
                "RU1",
                "RU10",
                "RU11",
                "RU12",
                "RU2",
                "RU3",
                "RU4",
                "RU5",
                "RU6",
                "RU7",
                "RU8",
                "RU9",
                "S",
                "S1",
                "S10",
                "S11",
                "S12",
                "S13",
                "S14",
                "S15",
                "S16",
                "S17",
                "S18",
                "S19",
                "S2",
                "S20",
                "S21",
                "S22",
                "S23",
                "S24",
                "S25",
                "S26",
                "S27",
                "S28",
                "S29",
                "S3",
                "S30",
                "S31",
                "S32",
                "S33",
                "S34",
                "S35",
                "S36",
                "S4",
                "S47",
                "S5",
                "S6",
                "S7",
                "S8",
                "S9",
                "SL1",
                "SL10",
                "SL12",
                "SL13",
                "SL15",
                "SL2",
                "SL3",
                "SL4",
                "SL5",
                "SL6",
                "SL7",
                "SL9",
                "SU1",
                "SU10",
                "SU12",
                "SU13",
                "SU15",
                "SU2",
                "SU3",
                "SU4",
                "SU5",
                "SU6",
                "SU7",
                "SU9",
                "T",
                "U",
                "U1",
                "U10",
                "U11",
                "U12",
                "U12W",
                "U13",
                "U14",
                "U14W",
                "U15",
                "U15W",
                "U16",
                "U17",
                "U18",
                "U18W",
                "U19",
                "U2",
                "U20",
                "U21",
                "U22",
                "U23",
                "U24",
                "U25",
                "U26",
                "U27",
                "U28",
                "U29",
                "U3",
                "U30",
                "U31",
                "U33",
                "U34",
                "U35",
                "U36",
                "U37",
                "U3W",
                "U4",
                "U40",
                "U41",
                "U42",
                "U5",
                "U6",
                "U6W",
                "U7",
                "U8",
                "U9",
                "U9W",
                "UA1",
                "UA2",
                "UA3",
                "UA4",
                "UA5",
                "UA6",
                "UD1",
                "UD10",
                "UD11",
                "UD12",
                "UD13",
                "UD14",
                "UD15",
                "UD16",
                "UD17",
                "UD18",
                "UD19",
                "UD2",
                "UD20",
                "UD21",
                "UD2W",
                "UD3",
                "UD4",
                "UD4W",
                "UD5",
                "UD6",
                "UD6W",
                "UD7",
                "UD8",
                "UD8W",
                "UD9",
                "US1",
                "US2",
                "US3",
                "US4",
                "US5",
                "US6",
                "V",
                "V1",
                "V10",
                "V11",
                "V12",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "V7",
                "V8",
                "V9",
                "W",
                "X",
                "Y",
                "Z"
            ]
    operators = [
                "7Hills roadways",
                "A1 Transports",
                "A1 Travels",
                "ATR Bus",
                "AVM Tours And Travels",
                "AZ Travels",
                "Anmol Tours & Travels",
                "B R Travels",
                "BMCC Travels",
                "BR Travels",
                "BSR TOURS & TRAVELS",
                "BSR Tours And Travels",
                "BTR Travels",
                "Balaji Cabs",
                "Basanth Tours",
                "Bharathi Travels",
                "Big Bus",
                "BigBus",
                "Bmcc Travels",
                "CMR Express",
                "Choudhary Travels Bhilwara",
                "DEGA TRAVELS",
                "DNR Express",
                "Dakshin Travels",
                "Delta Transport Pvt Ltd",
                "Delta Transports Pvt Ltd",
                "Dhanunjaya Travels",
                "Dream Line Travels Pvt Ltd",
                "Dreamlinetravels pvt ltd",
                "EXPRESS LINE",
                "Express Line",
                "Flixbus",
                "GAJRAJ BUS SERVICE ",
                "GEE PEE TRAVELS",
                "GPM Travels",
                "GRT Travels",
                "Gajraj bus service",
                "Geepee Travels",
                "Go Tour Travels And Holidays",
                "Go Tour Travels and Holidays",
                "HASH BUS",
                "Highline Transports",
                "IRA TRANSPORTS",
                "Intercity travels",
                "IntrCity SmartBus",
                "Ira Transport",
                "J J YATRA",
                "JAI MARUTHI TRAVELS",
                "Jabbar  Travels",
                "Jabbar Travels",
                "K.L.A Travels",
                "KAMAKSHI TOURS AND TRAVELS",
                "KBN Travels",
                "KGN INDIA",
                "KKaveri Travels",
                "KSM Roadlines",
                "KSM Roadways",
                "KVR Tours and Travels",
                "Kallada Tours and Travels",
                "Kallada Tours and Travels (VKLDA)",
                "Kallada Travels (Suresh Kallada)",
                "Kamakshi Tours And Travels",
                "Kaveri Tours and Travels",
                "Kaveri Travels",
                "Khaja Sardar Travels Hyd",
                "LVP Travels",
                "MMK Travels",
                "MRM Travels",
                "MSM Tours & Travels",
                "Medikonda Travels",
                "Medikonda Trravels",
                "Meghana Travels",
                "Morning Star Travels",
                "Mythri Tours And Travels",
                "N S Holidays",
                "NWAY HOLIDAYS",
                "National  travels",
                "National Travels(nts)",
                "Naveen Transport",
                "Naveen Travels (Durg)",
                "Northern Travels",
                "November Travels",
                "Orange Tours and Travels",
                "POOJA TRAVELS NAGPUR",
                "Pooja Travels (Nagpur)",
                "Pramukh Travels",
                "RAJESH TRANSPORTS",
                "RMS Transports",
                "Rajdhani Travels",
                "Rajdhani Travels (rjds)",
                "Rajesh Transports",
                "Raj’s Travels and Transports",
                "Ram Dalal Holidays Pvt Ltd",
                "Ramana Tours And Travels",
                "Ramana Travels",
                "Renuka Bus Service",
                "Royal Rich India",
                "Royal Rich India R No. 208",
                "S L Travels",
                "S.L  Travels",
                "SHYAMOLI PARIBAHAN PRIVATE LIMITED",
                "SL Travels",
                "SLC Road Lines",
                "SREE KVR TRAVELS",
                "SRI SVR travels",
                "SRS TRAVELS",
                "SRS Travels(srsr)",
                "STREAMLINE TOURS AND TRAVELS",
                "SVR Tours and Travels",
                "Saleem Tours and Travels",
                "Saleem Travels",
                "Saleem Travels (Hyderabad)",
                "Shama Sardar Travels HPM",
                "Shree Savariya Travels & Transport",
                "Shyamoli Paribahan Pvt Ltd",
                "Siva AMR Tours and Travels",
                "Skyline Transport",
                "Skyline Transports",
                "Sree Rama Tours & Travels",
                "Sri K.V.R Travels",
                "Sri KVR Travels",
                "Sri Krishna Travels",
                "Sri Sai Anjana Tours & Travels",
                "Sri Sai Anjana Tours and Travels",
                "Sri Tulasi Tours and Travels",
                "Streamline tours and travels",
                "Sugama  Tourist",
                "Sugama  Tourists",
                "Swamy Ayyappa Travels",
                "TVK TRAVELS",
                "TVK Travels",
                "Tranz India",
                "Tranzindia Travels",
                "Travel Point World LLP",
                "UNIVERSAL TRANSPORT SERVICE (UTS)",
                "Universal Bus( UTS)",
                "V Kaveri Travels",
                "VKV Travels",
                "VKaveri Travels",
                "VRL Travel",
                "VRL Travels",
                "VSR Tours and Travels",
                "Vasireddy Travels",
                "Vega Bus",
                "Vega Bus (hyderabad)",
                "Vikram Travels",
                "YAS Tours & Travels",
                "YAS Travels",
                "YOLOBUS",
                "Yolo Bus",
                "zingbus Maxx",
                "zingbus maxx",
                "zingbus plus"
            ]
    bus_types = [
                "A/C Seater / Sleeper (2+1)",
                "A/C Seater / Sleeper (2+2)",
                "A/C Seater/Sleeper (2+1)",
                "A/C Sleeper",
                "A/C Sleeper (2+1)",
                "A/C Volvo B11R Multi-Axle Sleeper (2+1)",
                "A/C, Seater Semi Sleeper, Premium Ishift, Multi Axle",
                "A/C, Seater Sleeper",
                "A/C, Seater Sleeper, Bharat Benz",
                "A/C, Seater Sleeper, Deluxe",
                "A/C, Seater Sleeper, Premium",
                "A/C, Semi Sleeper, Premium",
                "A/C, Semi Sleeper, Premium, Multi Axle",
                "A/C, Semi Sleeper, Scania, Multi Axle",
                "A/C, Sleeper",
                "A/C, Sleeper, Bharat Benz",
                "A/C, Sleeper, Bharat Benz Business Class",
                "A/C, Sleeper, Deluxe",
                "A/C, Sleeper, Mercedes Benz",
                "A/C, Sleeper, Premium",
                "A/C, Sleeper, Premium B9r Multi Axle, Multi Axle",
                "A/C, Sleeper, Premium Ishift B11r, Multi Axle",
                "A/C, Sleeper, Premium Ishift, Multi Axle",
                "A/C, Sleeper, Premium, Multi Axle",
                "A/C, Sleeper, Scania",
                "A/C, Sleeper, Scania, Multi Axle",
                "AC Sleeper (2+1)",
                "Benz A/C Sleeper (2+1)",
                "Bharat Benz A/C Seater /Sleeper (2+1)",
                "Bharat Benz A/C Sleeper (1+1)",
                "Bharat Benz A/C Sleeper (2+1)",
                "Bharat Benz NON A/C Seater / Sleeper (2+1)",
                "Mercedes Benz A/C Sleeper (2+1)",
                "Mercedes Benz Multi-Axle A/C Sleeper (2+1)",
                "NON A/C Seater Push Back (2+2)",
                "NON A/C Sleeper (2+1)",
                "NON AC Seater / Sleeper 2+1",
                "Non A/C Seater / Sleeper (2+1)",
                "Non A/C, Seater",
                "Non A/C, Seater Sleeper",
                "Non A/C, Seater Sleeper, Bharat Benz",
                "Non A/C, Seater Sleeper, Deluxe",
                "Non A/C, Sleeper",
                "Non A/C, Sleeper, Deluxe",
                "Non A/C, Sleeper, Premium",
                "Scania AC Multi Axle Sleeper (2+1)",
                "Scania Multi-Axle AC Semi Sleeper (2+2)",
                "VE A/C Seater / Sleeper (2+1)",
                "VE A/C Sleeper (2+1)",
                "VE Non A/C Sleeper (2+1)",
                "Volvo 9600 Multi-Axle A/C Sleeper (2+1)",
                "Volvo 9600 SLX Multi-Axle AC Sleeper (2+1)",
                "Volvo A/C B11R Multi Axle Semi Sleeper (2+2)",
                "Volvo A/C Semi Sleeper (2+2)",
                "Volvo Multi Axle A/C Sleeper I-Shift B11R (2+1)",
                "Volvo Multi-Axle A/C Sleeper (2+1)",
                "Volvo Multi-Axle A/C Sleeper (2+1) ",
                "Volvo Multi-Axle I-Shift A/C Semi Sleeper (2+2)",
                "Volvo Multi-Axle I-Shift A/C Sleeper (2+1)"
            ]
    
    routes = [
                "hyderabad_bangalore",
                "hyderabad_chennai"
            ]
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    input_data = {}
    
    # Auto-calculated fields to Hide
    auto_calculated = [
        'journey_is_weekend', 'is_night_departure', 'is_peak_hour',
        'is_last_minute', 'is_advance_booking', 
        'low_availability', 'very_low_availability', 'seats_sold_ratio',
        'is_lower_berth', 'is_premium_seat', 'total_capacity',
        'is_volvo', 'is_sleeper', 'is_seater', 'is_multi_axle', 'is_AC'
    ]
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🚌 Bus & Route", "🪑 Seat Details", "📅 Journey Timing"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'operator_name_le' in feature_names or 'operator_name' in feature_names:
                feature = 'operator_name_le' if 'operator_name_le' in feature_names else 'operator_name'
                input_data[feature] = st.selectbox(
                    "🚌 Bus Operator",
                    options=operators,
                    help=feature_descriptions.get(feature, "Select bus operator"),
                    key=f"input_{feature}"
                )
            
            if 'source_collection_le' in feature_names or 'source_collection' in feature_names:
                feature = 'source_collection_le' if 'source_collection_le' in feature_names else 'source_collection'
                input_data[feature] = st.selectbox(
                    "🗺️ Route",
                    options=routes,
                    help=feature_descriptions.get(feature, "Select route"),
                    key=f"input_{feature}"
                )
        
        with col2:
            if 'bus_type_le' in feature_names or 'bus_type' in feature_names:
                feature = 'bus_type_le' if 'bus_type_le' in feature_names else 'bus_type'
                input_data[feature] = st.selectbox(
                    "🚍 Bus Type",
                    options=bus_types,
                    help=feature_descriptions.get(feature, "Select bus type"),
                    key=f"input_{feature}"
                )
            
            if 'duration_hours' in feature_names:
                input_data['duration_hours'] = st.slider(
                    "🕐 Journey Duration (hours)",
                    min_value=1.0, max_value=24.0, value=10.0, step=0.5,
                    help=feature_descriptions.get('duration_hours', "Journey duration"),
                    key="input_duration_hours"
                )
                hrs = int(input_data['duration_hours'])
                mins = int((input_data['duration_hours'] % 1) * 60)
                st.caption(f"→ {hrs}h {mins}m travel time")

            # Add Total Capacity Input
            input_data['total_capacity'] = st.number_input(
                "🚌 Total Bus Capacity",
                min_value=10, max_value=100, value=50, step=1,
                help="Total number of seats in the bus (for occupancy calculation)",
                key="input_total_capacity"
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'seat_name_le' in feature_names or 'seat_name' in feature_names:
                feature = 'seat_name_le' if 'seat_name_le' in feature_names else 'seat_name'
                input_data[feature] = st.selectbox(
                    "🪑 Seat Number",
                    options=seat_names,
                    help=feature_descriptions.get(feature, "Select seat"),
                    key=f"input_{feature}"
                )
            
            if 'seat_is_upper' in feature_names:
                input_data['seat_is_upper'] = st.checkbox(
                    "Upper Berth",
                    value=False,
                    help=feature_descriptions.get('seat_is_upper', "Is upper berth?"),
                    key="input_seat_is_upper"
                )
            
            if 'seat_is_ladies' in feature_names:
                input_data['seat_is_ladies'] = st.checkbox(
                    "Ladies Seat",
                    value=False,
                    help=feature_descriptions.get('seat_is_ladies', "Reserved for women?"),
                    key="input_seat_is_ladies"
                )
        
        with col2:
            if 'available_seats' in feature_names:
                input_data['available_seats'] = st.slider(
                    "🪑 Available Seats",
                    min_value=0, max_value=50, value=25, step=1,
                    help=feature_descriptions.get('available_seats', "Seats available"),

                    key="input_available_seats"
                )
                
                booked_seats = input_data['total_capacity'] - input_data['available_seats']
                # Ensure non-negative
                booked_seats = max(0, booked_seats)
                occupancy = (booked_seats / input_data['total_capacity']) * 100
                st.caption(f"→ {booked_seats} booked ({occupancy:.0f}% full), {input_data['available_seats']} available")
            
            if 'window_seats' in feature_names:
                input_data['window_seats'] = st.slider(
                    "🪟 Window Seats Available",
                    min_value=0, max_value=25, value=10, step=1,
                    help=feature_descriptions.get('window_seats', "Window seats"),
                    key="input_window_seats"
                )
            
            if 'seat_is_horizontal' in feature_names:
                input_data['seat_is_horizontal'] = st.checkbox(
                    "Horizontal Sleeper",
                    value=False,
                    help=feature_descriptions.get('seat_is_horizontal', "Horizontal bed?"),
                    key="input_seat_is_horizontal"
                )
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'journey_weekday' in feature_names:
                day_selected = st.selectbox(
                    "📅 Journey Day",
                    options=weekdays,
                    help=feature_descriptions.get('journey_weekday', "Day of journey"),
                    key="input_journey_weekday"
                )
                input_data['journey_weekday'] = weekdays.index(day_selected)
            
            if 'departure_hour' in feature_names:
                hours = [f"{h:02d}:00" for h in range(24)]
                hour_selected = st.selectbox(
                    "⏰ Departure Time",
                    options=hours,
                    index=18,  # Default to 6 PM
                    help=feature_descriptions.get('departure_hour', "Departure time"),
                    key="input_departure_hour"
                )
                input_data['departure_hour'] = int(hour_selected.split(":")[0])
            
            if 'scrape_hour' in feature_names:
                current_hour = datetime.now().hour
                hours = [f"{h:02d}:00" for h in range(24)]
                hour_selected = st.selectbox(
                    "🕐 Current Time",
                    options=hours,
                    index=current_hour,
                    help=feature_descriptions.get('scrape_hour', "Current time"),
                    key="input_scrape_hour"
                )
                input_data['scrape_hour'] = int(hour_selected.split(":")[0])
        
        with col2:
            if 'hours_to_departure' in feature_names:
                input_data['hours_to_departure'] = st.slider(
                    "⏱️ Hours to Departure",
                    min_value=0.0, max_value=720.0, value=72.0, step=6.0,
                    help=feature_descriptions.get('hours_to_departure', "Hours until departure"),
                    key="input_hours_to_departure"
                )
                days = int(input_data['hours_to_departure'] / 24)
                hrs = int(input_data['hours_to_departure'] % 24)
                st.caption(f"→ {days} days, {hrs} hours from now")
            
            if 'days_to_journey' in feature_names:
                input_data['days_to_journey'] = st.slider(
                    "📆 Days to Journey",
                    min_value=0, max_value=90, value=7, step=1,
                    help=feature_descriptions.get('days_to_journey', "Days in advance"),
                    key="input_days_to_journey"
                )
    
    # Handle remaining features not in tabs
    remaining_features = [f for f in feature_names if f not in input_data and f not in auto_calculated]
    
    if remaining_features:
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            for idx, feature in enumerate(remaining_features):
                current_col = col1 if idx % 2 == 0 else col2
                with current_col:
                    feature_lower = feature.lower()
                    help_text = feature_descriptions.get(feature, f"Enter {feature}")
                    
                    if any(word in feature_lower for word in ['is_', 'has_', 'bool', 'flag']):
                        input_data[feature] = st.checkbox(
                            f"{feature}",
                            value=False,
                            help=help_text,
                            key=f"input_{feature}"
                        )
                    else:
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            value=0.0,
                            step=0.1,
                            help=help_text,
                            key=f"input_{feature}"
                        )
    
    # Prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🎯 Predict Price",
            type="primary",
            use_container_width=True
        )
    
    # Auto-predict on parameter change (check if any input has changed)
    # Create a hash of current inputs to detect changes
    current_input_hash = hash(str(sorted(input_data.items())))
    if 'last_input_hash' not in st.session_state:
        st.session_state['last_input_hash'] = None
    
    # Auto-trigger prediction if inputs changed and model is loaded
    auto_predict = False
    if st.session_state['last_input_hash'] is not None and current_input_hash != st.session_state['last_input_hash']:
        auto_predict = True
    
    st.session_state['last_input_hash'] = current_input_hash
    
    # Make prediction
    if predict_button or auto_predict:
        with st.spinner("🔮 Analyzing journey details..."):
            try:
                # Prepare features
                X = prepare_input_data(input_data, feature_names, encoders)
                
                # Make prediction
                if model_info["has_predict_proba"]:
                    # Classification model with probabilities
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)
                    
                    st.markdown("""
                        <div class='result-card'>
                            <h2 style='color: #667eea; text-align: center;'>💡 Prediction Result</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Class", predictions[0])
                    
                    if hasattr(model, 'classes_'):
                        with col2:
                            st.metric("Number of Classes", len(model.classes_))
                        
                        # Show probabilities
                        st.subheader("Class Probabilities")
                        prob_data = {
                            'Class': model.classes_,
                            'Probability': probabilities[0]
                        }
                        prob_df = pd.DataFrame(prob_data)
                        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                        
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                        
                        # Visualize
                        chart_data = pd.DataFrame({
                            'Class': [str(c) for c in model.classes_],
                            'Probability': probabilities[0]
                        })
                        st.bar_chart(chart_data.set_index('Class'), height=300)
                else:
                    # Regression model
                    prediction = model.predict(X)[0]
                    
                    # Inverse Log Transform to get Price in currency
                    final_price = np.expm1(prediction)
                    
                    # Beautiful result display
                    st.markdown("""
                        <div style='background: rgba(255,255,255,0.98); border-radius: 20px; padding: 3rem; margin: 2rem 0; box-shadow: 0 10px 40px rgba(0,0,0,0.2); text-align: center;'>
                            <h2 style='color: #667eea; margin-bottom: 2rem;'>✅ Price Prediction</h2>
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 2rem; margin: 1rem 0;'>
                                <p style='color: white; font-size: 1.2rem; margin: 0; opacity: 0.9;'>Predicted Ticket Price</p>
                                <h1 style='color: white; font-size: 4rem; margin: 1rem 0; font-weight: 700;'>₹ {:.2f}</h1>
                            </div>
                        </div>
                    """.format(final_price), unsafe_allow_html=True)
                    
                    # Additional insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Price Category",
                            "Premium" if final_price > 1000 else "Standard" if final_price > 500 else "Economy"
                        )
                    
                    with col2:
                        st.metric(
                            "Model Raw Output (Log Scale)",
                            f"{prediction:.4f}"
                        )
                
                # Store in session state
                st.session_state['last_prediction'] = {
                    'input': input_data,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Show full feature set
                with st.expander("🔍 View All Calculated Features"):
                    st.write("These are the exact values sent to the model after feature engineering:")
                    st.dataframe(X.T, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: rgba(255,255,255,0.6); padding: 2rem;'>
            <p>Made with ❤️ using Streamlit | Powered by Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
