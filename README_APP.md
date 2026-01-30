# Bus Ticket Price Prediction - Streamlit App

An interactive web application for predicting bus ticket prices using machine learning.

## Features

- **Real-time Price Predictions**: Get instant price estimates based on journey details
- **Organized Input Interface**: 4 intuitive tabs for entering seat, timing, bus, and availability information
- **Smart Insights**: Contextual recommendations based on booking patterns
- **Model Transparency**: View model performance metrics in the sidebar

## Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```powershell
   
   streamlit run app.py
   or .venv\Scripts\streamlit.exe run app.py
   ```

3. **Open in browser**: Navigate to `http://localhost:8501`

## Usage

1. **Enter Journey Details** across 4 tabs:
   - 🪑 **Seat Details**: Position, type, and availability
   - ⏰ **Timing**: Date, time, and duration
   - 🚍 **Bus Info**: Operator, bus type, and route
   - 📊 **Availability**: Seat availability metrics

2. **Click "Predict Price"** to get your estimate

3. **View Results**:
   - Predicted price in ₹
   - Expected price range
   - Contextual insights and recommendations

## Model Information

- **Algorithm**: XGBoost Regressor
- **Performance**:
  - R² Score: 0.8487
  - MAPE: 8.27%
  - MAE: ₹123.57
- **Features**: 24 engineered features including temporal patterns, seat preferences, and availability metrics
- **Training Data**: 377,290 samples

## Project Structure

```
dynamic-pricing/
├── app.py                    # Streamlit application
├── export_model.py           # Model export script
├── requirements.txt          # Python dependencies
├── models/
│   └── production/          # Exported model artifacts
│       ├── bus_price_model.pkl
│       ├── feature_names.json
│       ├── model_metrics.json
│       └── model_info.json
└── notebooks/               # Training notebooks
```

## Example Prediction

**Input**:
- Journey: Hyderabad → Bangalore
- Operator: Jabbar Travels
- Bus: Volvo A/C Semi Sleeper
- Departure: 3 days from now at 20:00
- Available Seats: 25/50

**Output**:
- **Price**: ₹1,563.39
- **Range**: ₹1,434 - ₹1,693
- **Insights**: Weekend travel, Premium Volvo bus

## Technical Details

### Feature Engineering

The app automatically creates 24 features:
- **Direct inputs**: Seat details, timing, availability
- **Engineered features**: Weekend indicator, night departure, scarcity signals
- **Encoded features**: Operator, bus type, route, seat name

### Model Export

Run `export_model.py` to consolidate the latest trained model:
```powershell
python export_model.py
```

This copies model artifacts to `models/production/` for use by the Streamlit app.

## License

This project is for demonstration purposes.

## Contact

For questions or feedback, please contact the development team.

## Prerequsites

1. paste the converted  parquet data under data/complete data
2.create venv and install all the requirements
3.run all cells

