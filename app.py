"""
Bus Ticket Price Prediction - Streamlit App (Simplified)
Uses only the XGBoost model file and manual feature engineering
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# Load Model and Metadata
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load model and metadata"""
    production_dir = Path("models/production")
    
    # Load XGBoost model directly
    with open(production_dir / "bus_price_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names
    with open(production_dir / "feature_names.json", 'r') as f:
        feature_names = json.load(f)
    
    # Load metrics
    with open(production_dir / "model_metrics.json", 'r') as f:
        metrics = json.load(f)
    
    # Load metadata
    with open(production_dir / "model_info.json", 'r') as f:
        metadata = json.load(f)
    
    return model, feature_names, metrics, metadata

# ============================================================================
# Feature Engineering Functions
# ============================================================================

def engineer_features(input_data):
    """Apply feature engineering to input data"""
    df = pd.DataFrame([input_data])
    
    # Extract is_AC from bus_type
    df['is_AC'] = df['bus_type'].str.contains('A/C|AC', case=False, na=False, regex=True).astype(int)
    
    # Extract is_sleeper from bus_type (if not provided directly)
    if 'seat_is_seater' in df.columns:
        df['is_sleeper'] = (~df['seat_is_seater']).astype(int)
    else:
        # Fallback: infer from bus_type if seat_is_seater not provided
        df['is_sleeper'] = df['bus_type'].str.contains('sleeper', case=False, na=False, regex=True).astype(int)
    
    # Temporal patterns
    df['journey_is_weekend'] = df['journey_weekday'].isin([5, 6]).astype(int)
    df['is_night_departure'] = df['departure_hour'].between(20, 5).astype(int)
    
    # Demand & scarcity signals
    df['low_availability'] = (df['available_seats'] <= 5).astype(int)
    df['very_low_availability'] = (df['available_seats'] <= 2).astype(int)
    df['seats_sold_ratio'] = 1 - (df['available_seats'] / 50).clip(upper=1)
    
    # Seat preference
    df['is_lower_berth'] = (~df['seat_is_upper']).astype(int)
    df['is_premium_seat'] = (df['is_lower_berth'] & (df['window_seats'] > 0)).astype(int)
    
    # Bus flags
    df['is_volvo'] = df['bus_type'].str.contains('volvo', case=False, na=False).astype(int)
    
    return df

def simple_label_encode(value, category_type):
    """Simple hash-based encoding for categorical variables"""
    # Use hash to create consistent numeric encoding
    return abs(hash(str(value))) % 1000

def prepare_features(input_data, feature_names):
    """Prepare features for prediction"""
    # Apply feature engineering
    df = engineer_features(input_data)
    
    # Add label encoded features using simple hash encoding
    df['operator_name_le'] = simple_label_encode(input_data['operator_name'], 'operator')
    df['bus_type_le'] = simple_label_encode(input_data['bus_type'], 'bus_type')
    df['source_collection_le'] = simple_label_encode(input_data['source_collection'], 'route')
    df['seat_name_le'] = simple_label_encode(input_data['seat_name'], 'seat')
    
    # Select only the features needed by the model
    X = df[feature_names]
    
    return X

# ============================================================================
# Prediction Function
# ============================================================================

def predict_price(model, feature_names, input_data):
    """Make price prediction from input data"""
    # If seat is not available, return 0
    if not input_data.get('seat_is_available', True):
        return 0.0
    
    X = prepare_features(input_data, feature_names)
    
    # Make prediction (model predicts log price)
    log_price = model.predict(X)[0]
    price = np.expm1(log_price)  # Convert back from log
    
    return max(0, price)  # Ensure non-negative

# ============================================================================
# Streamlit App
# ============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Bus Price Predictor",
        page_icon="🚌",
        layout="wide"
    )
    
    # Load model artifacts
    try:
        model, feature_names, metrics, metadata = load_model_artifacts()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()
    
    # Header
    st.title("🚌 Bus Ticket Price Prediction")
    st.markdown("### Dynamic Pricing Model - Predict bus ticket prices based on various factors")
    
    # Model info in sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Model Type", metrics['model_name'])
        st.metric("R² Score", f"{metrics['metrics']['r2_log']:.4f}")
        st.metric("MAPE", f"{metrics['metrics']['mape_pct']:.2f}%")
        st.metric("MAE (₹)", f"₹{metrics['metrics']['mae_inr']:.2f}")
        
        st.divider()
        st.caption(f"Training Date: {metadata['training_timestamp']}")
        st.caption(f"Training Samples: {metrics['training_samples']:,}")
        st.caption(f"Features: {metadata['num_features']}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Input Features")
        
        # Create tabs for organized input
        tab1, tab2, tab3, tab4 = st.tabs(["🪑 Seat Details", "⏰ Timing", "🚍 Bus Info", "📊 Availability"])
        
        with tab1:
            st.subheader("Seat Characteristics")
            col_a, col_b = st.columns(2)
            with col_a:
                seat_type = st.selectbox("Seat Type", ["Seater", "Sleeper"], index=1, help="Seater or Sleeper seat")
                seat_is_upper = st.selectbox("Seat Position", ["Lower", "Upper"], index=0)
                seat_is_ladies = st.selectbox("Ladies Seat", ["No", "Yes"], index=0)
            with col_b:
                seat_is_horizontal = st.selectbox("Horizontal Seat", ["No", "Yes"], index=1)
                seat_is_available = st.selectbox("Seat Available", ["Yes", "No"], index=0)
                seat_name = st.text_input("Seat Name", value="D1", help="e.g., D1, U5, L3")
        
        with tab2:
            st.subheader("Journey Timing")
            col_a, col_b = st.columns(2)
            with col_a:
                journey_date = st.date_input("Journey Date", value=datetime.now() + timedelta(days=3))
                departure_time = st.time_input("Departure Time", value=datetime.strptime("20:00", "%H:%M").time())
            with col_b:
                duration_hours = st.number_input("Journey Duration (hours)", min_value=1.0, max_value=24.0, value=10.0, step=0.5)
                scrape_hour = st.number_input("Current Hour (0-23)", min_value=0, max_value=23, value=datetime.now().hour)
        
        with tab3:
            st.subheader("Bus Details")
            col_a, col_b = st.columns(2)
            with col_a:
                operator_name = st.selectbox("Operator", [
                    "7Hills roadways", "A1 Travels", "ATR Bus", "AZ Travels",
                    "Anmol Tours & Travels", "B R Travels", "BSR Tours And Travels",
                    "Balaji Cabs", "Bharathi Travels", "BigBus", "Bmcc Travels",
                    "CMR Express", "DEGA TRAVELS", "DNR Express",
                    "Delta Transports Pvt Ltd", "Dhanunjaya Travels",
                    "Dream Line Travels Pvt Ltd", "Express Line",
                    "GEE PEE TRAVELS", "GRT Travels", "Gajraj bus service",
                    "Go Tour Travels and Holidays", "HASH BUS", "Highline Transports",
                    "IRA TRANSPORTS", "Jabbar  Travels", "Kallada Travels",
                    "Orange Travels", "Parveen Travels", "SRS Travels",
                    "Sharma Travels", "VRL Travels"
                ], index=25)
                bus_type = st.selectbox("Bus Type", [
                    "A/C Seater / Sleeper (2+1)",
                    "A/C Seater / Sleeper (2+2)",
                    "A/C Seater/Sleeper (2+1)",
                    "A/C Sleeper (2+1)",
                    "A/C Volvo B11R Multi-Axle Sleeper (2+1)",
                    "AC Sleeper (2+1)",
                    "Benz A/C Sleeper (2+1)",
                    "Bharat Benz A/C Seater /Sleeper (2+1)",
                    "Bharat Benz A/C Semi Sleeper (2+2)",
                    "Bharat Benz A/C Sleeper (1+1)",
                    "Bharat Benz A/C Sleeper (2+1)",
                    "Bharat Benz NON A/C Seater / Sleeper (2+1)",
                    "Mercedes Benz A/C Sleeper (2+1)",
                    "Mercedes Benz Multi-Axle A/C Sleeper (2+1)",
                    "NON A/C Seater Push Back (2+2)",
                    "NON A/C Sleeper (2+1)",
                    "NON AC Seater / Sleeper 2+1",
                    "Non A/C Seater / Sleeper (2+1)",
                    "Scania AC Multi Axle Sleeper (2+1)",
                    "Scania Multi-Axle AC Semi Sleeper (2+2)",
                    "Volvo A/C B11R Multi Axle Semi Sleeper (2+2)",
                    "Volvo A/C Sleeper (2+1)"
                ], index=20)
            with col_b:
                source_collection = st.selectbox("Route", [
                    "hyderabad_bangalore",
                    "hyderabad_chennai"
                ])
                window_seats = st.number_input("Window Seats Available", min_value=0, max_value=50, value=20)
        
        with tab4:
            st.subheader("Availability Metrics")
            col_a, col_b = st.columns(2)
            with col_a:
                available_seats = st.number_input("Available Seats", min_value=0, max_value=50, value=25)
            with col_b:
                st.info(f"Seats Sold: {50 - available_seats} / 50")
                if available_seats <= 2:
                    st.warning("⚠️ Very Low Availability")
                elif available_seats <= 5:
                    st.warning("⚠️ Low Availability")
        
        # Calculate derived features
        now = datetime.now()
        journey_datetime = datetime.combine(journey_date, departure_time)
        hours_to_departure = (journey_datetime - now).total_seconds() / 3600
        days_to_journey = (journey_date - now.date()).days
        journey_weekday = journey_date.weekday()
        
        # Prepare input data
        input_data = {
            'seat_is_seater': seat_type == "Seater",
            'seat_is_upper': seat_is_upper == "Upper",
            'seat_is_ladies': seat_is_ladies == "Yes",
            'seat_is_horizontal': seat_is_horizontal == "Yes",
            'seat_is_available': seat_is_available == "Yes",
            'hours_to_departure': max(0, hours_to_departure),
            'duration_hours': duration_hours,
            'days_to_journey': max(0, days_to_journey),
            'scrape_hour': scrape_hour,
            'journey_weekday': journey_weekday,
            'departure_hour': departure_time.hour,
            'operator_name': operator_name,
            'bus_type': bus_type,
            'source_collection': source_collection,
            'seat_name': seat_name,
            'available_seats': available_seats,
            'window_seats': window_seats
        }
        
        # Predict button
        st.divider()
        if st.button("🎯 Predict Price", type="primary", use_container_width=True):
            try:
                predicted_price = predict_price(model, feature_names, input_data)
                
                # Store in session state
                st.session_state['last_prediction'] = predicted_price
                st.session_state['last_input'] = input_data
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    with col2:
        st.header("💰 Prediction Result")
        
        if 'last_prediction' in st.session_state:
            predicted_price = st.session_state['last_prediction']
            
            # Check if seat is not available
            if predicted_price == 0.0:
                st.error("❌ Seat Not Available")
                st.warning("Price cannot be predicted for unavailable seats.")
            else:
                # Display prediction
                st.metric(
                    label="Predicted Price",
                    value=f"₹{predicted_price:.2f}",
                    delta=None
                )
                
                # Price range (±MAPE)
                mape = metrics['metrics']['mape_pct'] / 100
                lower_bound = predicted_price * (1 - mape)
                upper_bound = predicted_price * (1 + mape)
                
                st.info(f"**Expected Range:** ₹{lower_bound:.2f} - ₹{upper_bound:.2f}")
            
            # Additional insights (only show if seat is available)
            if predicted_price > 0:
                st.divider()
                st.subheader("📈 Insights")
                
                input_data = st.session_state['last_input']
                
                if input_data['hours_to_departure'] < 6:
                    st.warning("⏰ Last-minute booking - prices may be higher")
                elif input_data['hours_to_departure'] > 168:
                    st.success("✅ Early booking - better prices expected")
                
                if input_data['available_seats'] <= 5:
                    st.warning("🔥 High demand - limited seats available")
                
                if input_data['journey_weekday'] in [4, 5, 6]:
                    st.info("📅 Weekend travel - prices may vary")
                
                if "volvo" in input_data['bus_type'].lower():
                    st.success("⭐ Premium Volvo bus")
                
                # Show seat type info
                if input_data.get('seat_is_seater', False):
                    st.info("🪑 Seater seat selected")
                else:
                    st.info("🛏️ Sleeper seat selected")
        
        else:
            st.info("👈 Fill in the details and click 'Predict Price' to see the prediction")

if __name__ == "__main__":
    main()
