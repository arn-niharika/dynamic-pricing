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
    page_title="Model Prediction App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        # Assuming 50 total seats for ratio calculation if not provided
        df['seats_sold_ratio'] = (1 - (df['available_seats'] / 50).clip(upper=1))
    
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
    st.set_page_config(
        page_title="Model Prediction App",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Model Prediction App")
    st.markdown("Upload a pickle file with your trained model and make predictions easily")
    
    # Sidebar for model upload
    with st.sidebar:
        st.header("📦 Model Upload")
        uploaded_file = st.file_uploader(
            "1. Choose Model (.pkl)",
            type=["pkl", "pickle"],
            help="Upload your trained model in pickle format"
        )
        
        uploaded_encoder = st.file_uploader(
            "2. Choose Encoders (.pkl)",
            type=["pkl", "pickle"],
            help="Upload the categorical_encoders.pkl file if your model uses encoded features"
        )
        
        if uploaded_file is not None:
            st.success("✅ Model uploaded successfully!")
    
    # Main content
    if uploaded_file is None:
        st.info("👈 **Please upload a model pickle file to get started.**")
        st.markdown("""
        ### How to use this app:
        1. **Upload a Model**: Click the file uploader on the left to select your trained model (in `.pkl` format)
        2. **Enter Features**: Provide values for the features your model expects
        3. **Get Predictions**: Click the predict button to see results
        
        ### Supported Models:
        - ✅ XGBoost, LightGBM, CatBoost
        - ✅ Scikit-learn (RandomForest, LinearRegression, LogisticRegression, etc.)
        - ✅ Neural Networks (Keras/TensorFlow)
        - ✅ Any custom Python model with `predict()` method
        """)
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
            st.sidebar.info("ℹ️ **App will work without encoders!** You'll enter numeric codes directly for categorical features (e.g., operator_name_le, bus_type_le).")
        else:
            st.sidebar.success(f"✅ Loaded {len(encoders)} encoders")
    else:
        st.sidebar.info("ℹ️ **No encoders uploaded.** You'll enter numeric codes directly for categorical features (e.g., 0-152 for operator_name_le).")
    
    
    # Get model information
    model_info = get_model_info(model)
    
    # Display model information in sidebar
    with st.sidebar:
        st.divider()
        st.header("📊 Model Details")
        st.write(f"**Type:** {model_info['type']}")
        st.write(f"**Library:** {model_info['module']}")
        if "n_features" in model_info:
            st.write(f"**Expected Features:** {model_info['n_features']}")
        if model_info.get("has_predict"):
            st.write("**Prediction Type:** ✅ Regression" if not model_info.get("has_predict_proba") else "**Prediction Type:** ✅ Classification")
    
    # Get feature names
    feature_names = model_info.get("feature_names", [])
    
    if not feature_names:
        st.warning("⚠️ Could not automatically detect feature names from the model.")
        st.markdown("**Option 1**: Enter feature names manually (comma-separated)")
        manual_features = st.text_input(
            "Feature names",
            help="Enter feature names separated by commas, e.g., age,income,credit_score"
        )
        
        if manual_features:
            feature_names = [f.strip() for f in manual_features.split(",")]
            st.success(f"Using {len(feature_names)} features")
        else:
            st.info("👈 Please provide feature names to continue")
            return
    else:
        st.success(f"✅ Detected {len(feature_names)} features from model")
        with st.expander("View feature names"):
            st.write(feature_names)
    
    # ===== IMPROVED INPUT SECTION WITH DESCRIPTIONS =====
    st.header("📝 Input Features")
    
    if not encoders:
        st.info("ℹ️ **Working without encoders:** The app will use default values (0) for categorical features like `operator_name_le`, `bus_type_le`, etc. Predictions will still work but may be less accurate.")
    
    st.markdown(f"Provide values for **{len(feature_names)}** features. **Hover over labels** for detailed descriptions.")
    
    # Feature descriptions and help text
    feature_descriptions = {
        # Seat features
        'seat_is_seater': '✓ Regular SEATING (like airplane) | ✗ BED/SLEEPER (like train)',
        'seat_is_upper': '✓ UPPER BUNK (top bed) | ✗ LOWER BUNK (bottom bed)',
        'seat_is_ladies': '✓ RESERVED FOR WOMEN | ✗ General seat for anyone',
        'seat_is_horizontal': '✓ LYING DOWN (horizontal bed) | ✗ Sitting upright',
        'seat_is_available': '✓ AVAILABLE TO BOOK | ✗ Already sold/booked',
        'seat_name': 'Your SEAT CODE (e.g., SL1, U5, L3) - SL=Sleeper-Lower, U=Upper, L=Lower, W=Window',
        'seat_name_le': 'Pick your SEAT CODE from dropdown',
        
        # Journey features
        'journey_weekday': 'Pick which day (Monday-Sunday)',
        'departure_hour': 'What TIME does bus leave? (00:00 to 23:00)',
        'hours_to_departure': 'How many HOURS from NOW until departure?',
        'days_to_journey': 'How many DAYS from TODAY? (advance booking)',
        'duration_hours': 'How LONG is the journey? (in hours)',
        'scrape_hour': 'What is the CURRENT HOUR? (0=midnight, 12=noon, 18=evening)',
        
        # Bus features
        'operator_name': 'Which BUS COMPANY?',
        'operator_name_le': 'Pick BUS COMPANY from list',
        'bus_type': 'What TYPE of bus? (AC/Non-AC, Sleeper/Seater)',
        'bus_type_le': 'Pick BUS TYPE from list',
        'source_collection': 'Which ROUTE? (Hyderabad-Bangalore, etc.)',
        'source_collection_le': 'Pick ROUTE from list',
        
        # Availability features
        'available_seats': 'How many SEATS STILL AVAILABLE? (out of 50 total)',
        'window_seats': 'How many WINDOW SEATS available? (seats with nice view)',
        'window_seats': 'How many WINDOW SEATS available? (seats with nice view)',
        'low_availability': '✓ YES = 5 or FEWER seats left | ✗ NO = more seats available (Auto-calculated)',
        'very_low_availability': '✓ YES = 2 or FEWER seats left | ✗ NO = more seats available (Auto-calculated)',
        'seats_sold_ratio': 'What % SOLD? (0=empty, 0.5=half full, 1=completely sold)',
        
        # Demand features
        'is_lower_berth': '✓ LOWER BED (bottom, usually more expensive) | ✗ Upper bed',
        'is_premium_seat': '✓ PREMIUM = Lower bed + Window view (BEST SEATS) | ✗ Regular',
        'is_AC': '✓ AIR-CONDITIONED (cool) | ✗ Non-AC (hot)',
        'is_sleeper': '✓ SLEEPER/BED seats | ✗ Normal sitting seats',
        'is_volvo': '✓ LUXURY Volvo/Premium brand | ✗ Regular bus',
        'is_night_departure': '✓ Leaves at NIGHT (8PM-5AM) | ✗ Day time',
        'journey_is_weekend': '✓ WEEKEND trip (Sat/Sun) | ✗ Weekday trip',
    }
    
    # Predefined options
    seat_names = ['SL1', 'SL2', 'SL3', 'SL4', 'SL5', 'L1', 'L2', 'L3', 'U1', 'U2', 'U3', 'U4', 'U5', 'W1', 'W2']
    operators = ['SRS Travels', 'VRL Travels', 'Orange Travels', 'Kallada Travels', 'GRT Travels', 'BigBus', 'Jabbar Travels', 'Sharma Travels', 'Parveen Travels', 'Express Line', 'A1 Travels', 'ATR Bus', 'AZ Travels', '7Hills roadways']
    bus_types = ['AC Sleeper (2+1)', 'Non-AC Sleeper (2+1)', 'AC Seater (2+2)', 'Non-AC Seater (2+2)', 'Volvo AC Sleeper (2+1)', 'Mercedes AC Sleeper (2+1)', 'AC Semi Sleeper (2+2)']
    routes = ['hyderabad_chennai', 'hyderabad_bangalore']
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    input_data = {}
    col1, col2 = st.columns(2)
    
    # Dynamic input based on feature name
    col1, col2 = st.columns(2)
    
    # Auto-calculated fields to Hide
    auto_calculated = [
        'journey_is_weekend', 'is_night_departure', 'is_peak_hour',
        'is_last_minute', 'is_advance_booking', 
        'low_availability', 'very_low_availability', 'seats_sold_ratio',
        'is_lower_berth', 'is_premium_seat',
        'is_volvo', 'is_sleeper', 'is_seater', 'is_multi_axle', 'is_AC'
    ]
    
    # Dynamic input based on feature name
    col_idx = 0
    for feature in feature_names:
        # Skip auto-calculated features
        if feature in auto_calculated:
            continue
            
        feature_lower = feature.lower()
        help_text = feature_descriptions.get(feature, f"Enter {feature}")
        
        current_col = col1 if col_idx % 2 == 0 else col2
        
        with current_col:
            # ===== SEAT NAME - DROPDOWN =====
            if 'seat_name' in feature_lower:
                if feature not in input_data:
                    input_data[feature] = st.selectbox(
                        f"🪑 {feature}",
                        options=seat_names,
                        help=help_text,
                        key=f"input_{feature}"
                    )
                    col_idx += 1
            
            # ===== OPERATOR - DROPDOWN =====
            elif 'operator' in feature_lower:
                if feature not in input_data:
                    input_data[feature] = st.selectbox(
                        f"🚌 {feature}",
                        options=operators,
                        help=help_text,
                        key=f"input_{feature}"
                    )
                    col_idx += 1
            
            # ===== BUS TYPE - DROPDOWN =====
            elif 'bus_type' in feature_lower:
                if feature not in input_data:
                    input_data[feature] = st.selectbox(
                        f"🚍 {feature}",
                        options=bus_types,
                        help=help_text,
                        key=f"input_{feature}"
                    )
                    col_idx += 1
            
            # ===== ROUTE/SOURCE - DROPDOWN =====
            elif 'source_collection' in feature_lower or 'route' in feature_lower:
                if feature not in input_data:
                    input_data[feature] = st.selectbox(
                        f"🗺️ {feature}",
                        options=routes,
                        help=help_text,
                        key=f"input_{feature}"
                    )
                    col_idx += 1
            
            # ===== WEEKDAY - DROPDOWN =====
            elif feature == 'journey_weekday':
                day_selected = st.selectbox(
                    f"📅 {feature}",
                    options=weekdays,
                    help=help_text,
                    key=f"input_{feature}"
                )
                input_data[feature] = weekdays.index(day_selected)  # Convert to 0-6
                col_idx += 1
            
            # ===== DEPARTURE HOUR - DROPDOWN =====
            elif feature == 'departure_hour':
                hours = [f"{h:02d}:00" for h in range(24)]
                hour_selected = st.selectbox(
                    f"⏰ {feature}",
                    options=hours,
                    help=help_text,
                    key=f"input_{feature}"
                )
                input_data[feature] = int(hour_selected.split(":")[0])
                col_idx += 1
            
            # ===== SCRAPE HOUR - DROPDOWN =====
            elif feature == 'scrape_hour':
                hours = [f"{h:02d}:00" for h in range(24)]
                hour_selected = st.selectbox(
                    f"🕐 {feature}",
                    options=hours,
                    help=help_text,
                    key=f"input_{feature}"
                )
                input_data[feature] = int(hour_selected.split(":")[0])
                col_idx += 1
            
            # ===== AVAILABLE SEATS - SLIDER (0-50) =====
            elif feature == 'available_seats':
                input_data[feature] = st.slider(
                    f"🪑 {feature}",
                    min_value=0, max_value=50, value=25, step=1,
                    help=help_text,
                    key=f"input_{feature}"
                )
                st.caption(f"→ {50-input_data[feature]} booked, {input_data[feature]} available")
                col_idx += 1
            
            # ===== WINDOW SEATS - SLIDER (0-25) =====
            elif feature == 'window_seats':
                input_data[feature] = st.slider(
                    f"🪟 {feature}",
                    min_value=0, max_value=25, value=10, step=1,
                    help=help_text,
                    key=f"input_{feature}"
                )
                col_idx += 1
            
            # ===== HOURS TO DEPARTURE - SLIDER =====
            elif feature == 'hours_to_departure':
                input_data[feature] = st.slider(
                    f"⏱️ {feature}",
                    min_value=0.0, max_value=720.0, value=72.0, step=6.0,
                    help=help_text,
                    key=f"input_{feature}"
                )
                days = int(input_data[feature] / 24)
                hrs = int(input_data[feature] % 24)
                st.caption(f"→ {days} days, {hrs} hours from now")
                col_idx += 1
            
            # ===== DAYS TO JOURNEY - SLIDER =====
            elif feature == 'days_to_journey':
                input_data[feature] = st.slider(
                    f"📆 {feature}",
                    min_value=0, max_value=90, value=7, step=1,
                    help=help_text,
                    key=f"input_{feature}"
                )
                col_idx += 1
            
            # ===== DURATION HOURS - SLIDER =====
            elif feature == 'duration_hours':
                input_data[feature] = st.slider(
                    f"🕐 {feature}",
                    min_value=1.0, max_value=24.0, value=10.0, step=0.5,
                    help=help_text,
                    key=f"input_{feature}"
                )
                hrs = int(input_data[feature])
                mins = int((input_data[feature] % 1) * 60)
                st.caption(f"→ {hrs}h {mins}m travel")
                col_idx += 1
            
            # ===== SEATS SOLD RATIO - SLIDER (0-1) =====
            elif feature == 'seats_sold_ratio':
                input_data[feature] = st.slider(
                    f"📊 {feature}",
                    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                    help=help_text,
                    key=f"input_{feature}"
                )
                pct = int(input_data[feature] * 100)
                st.caption(f"→ {pct}% seats sold")
                col_idx += 1
            
            # ===== BOOLEAN - CHECKBOX =====
            elif any(word in feature_lower for word in ['is_', 'has_', 'bool', 'flag']):
                input_data[feature] = st.checkbox(
                    f"✓ {feature}",
                    value=False,
                    help=help_text,
                    key=f"input_{feature}"
                )
                col_idx += 1
            
            # ===== DEFAULT - NUMBER INPUT =====
            else:
                input_data[feature] = st.number_input(
                    f"🔢 {feature}",
                    value=0.0,
                    step=0.1,
                    help=help_text,
                    key=f"input_{feature}"
                )
                col_idx += 1
    
    # Define categorical options for common fields
    
    # Prediction button
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button(
            "🎯 Make Prediction",
            type="primary",
            use_container_width=True
        )
    
    # Make prediction
    if predict_button:
        try:
            # Prepare features
            X = prepare_input_data(input_data, feature_names, encoders)
            
            # Make prediction
            if model_info["has_predict_proba"]:
                # Classification model with probabilities
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)
                
                st.header("💡 Prediction Result")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Class", predictions[0])
                
                if hasattr(model, 'classes_'):
                    with col2:
                        st.metric("Number of Classes", len(model.classes_))
                    
                    # Show probabilities for each class
                    st.subheader("Class Probabilities")
                    prob_data = {
                        'Class': model.classes_,
                        'Probability': probabilities[0]
                    }
                    prob_df = pd.DataFrame(prob_data)
                    prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
                    # Visualize probabilities
                    chart_data = pd.DataFrame({
                        'Class': [str(c) for c in model.classes_],
                        'Probability': probabilities[0]
                    })
                    st.bar_chart(chart_data.set_index('Class'), height=300)
            else:
                # Regression model
                prediction = model.predict(X)[0]
                
                # Inverse Log Transform to get Price in currency
                # Assuming the model was trained on log(price)
                final_price = np.expm1(prediction)
                
                st.divider()
                st.header("✅ Prediction Result")
                
                # Display with better formatting
                st.subheader(f"Predicted Price")
                st.metric("🎯 Result", f"₹ {final_price:,.2f}")
                st.caption(f"(Model Raw Output: {prediction:.4f})")
                
                st.info(
                    f"💡 **What this means:**\n\n"
                    f"Your model predicted: **{prediction:.4f}**\n\n"
                    f"This value depends on:\n"
                    f"- What your model was trained for (e.g., price prediction, score, count)\n"
                    f"- The scale of your training data\n\n"
                    f"**Check with your data scientist or model documentation** to understand what this number represents!"
                )
            
            # Store in session state
            st.session_state['last_prediction'] = {
                'input': input_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Show full feature set used for prediction
            with st.expander("🔍 View All Model Features (26 Calculated Features)"):
                st.write("These are the exact values sent to the model after processing:")
                st.dataframe(X.T, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    # Additional tools
    st.divider()
    st.header("🛠️ Additional Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Show Input as DataFrame"):
            input_df = pd.DataFrame([input_data])
            st.dataframe(input_df, use_container_width=True)
    
    with col2:
        if st.button("💾 Download Prediction Template"):
            template_df = pd.DataFrame([input_data])
            csv = template_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="prediction_template.csv",
                mime="text/csv"
            )



if __name__ == "__main__":
    main()
