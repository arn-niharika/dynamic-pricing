# Universal Model Prediction - Streamlit App

A universal web application for making predictions using any trained machine learning model.
Upload any pickle file containing your trained model and use it to make predictions on new data.

## Features

- **Universal Model Support**: Works with any pickled Python model (XGBoost, LightGBM, Scikit-learn, etc.)
- **Easy Model Upload**: Simply upload your `.pkl` or `.pickle` file
- **Auto Feature Detection**: Automatically detects feature names from the model
- **Smart Input Types**: Intelligently infers input types (boolean, integer, float) from feature names
- **Flexible Predictions**: Supports both regression and classification models
- **Probability Display**: Shows class probabilities for classification models
- **CSV Export**: Download prediction templates for batch processing
- **Error Handling**: Comprehensive error messages and debugging info

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
   ```

3. **Open in browser**: Navigate to `http://localhost:8501`

## Usage

### Step 1: Upload Your Model
Click the file uploader in the sidebar and select your pickle file (`.pkl` or `.pickle`).

### Step 2: Review Model Information
The app automatically displays:
- Model type (e.g., XGBoostRegressor, RandomForestClassifier)
- Number of features expected
- Feature names (if available in the model)

### Step 3: Provide Input Features
- Enter values for each feature
- The app intelligently infers input types:
  - **Checkboxes**: For boolean features (containing "is_", "has_", "bool", "flag")
  - **Integer inputs**: For count/ID features (containing "count", "number", "age", "id")
  - **Float inputs**: For continuous features

### Step 4: Make Predictions
Click the "Make Prediction" button to see results:
- **Regression**: Shows predicted value and formatted output
- **Classification**: Shows predicted class and probability distribution

## Supported Model Types

✅ **Regression Models**:
- XGBoost, LightGBM, CatBoost
- Scikit-learn (LinearRegression, RandomForestRegressor, etc.)
- Custom regression models

✅ **Classification Models**:
- XGBoost, LightGBM, CatBoost
- Scikit-learn (LogisticRegression, RandomForestClassifier, etc.)
- Custom classification models

## Example Predictions

### Regression Example (Bus Price Prediction)
**Input**:
- 24 features related to bus journey details
- Model: XGBoost

**Output**:
- Predicted price: ₹1,563.39

### Classification Example (Customer Churn)
**Input**:
- Customer features (age, income, tenure, etc.)
- Model: Random Forest Classifier

**Output**:
- Predicted class: Churned or Retained
- Probability: 75% confidence

## Project Structure

```
dynamic-pricing/
├── app.py                   # Streamlit application
├── requirements.txt         # Python dependencies
├── README_APP.md            # This file
├── test.json                # Sample test data
├── data/
│   └── complete_data/       # Training data
├── models/
│   ├── production/          # Production models
│   └── saved_runs/          # Historical trained models
└── notebooks/               # Training notebooks
```

## Technical Details

### Automatic Feature Detection

The app attempts to detect feature names from:
1. `model.feature_names_in_` (Scikit-learn models)
2. `model.feature_names` (Custom models)
3. User-provided feature names (if auto-detection fails)

### Smart Input Type Inference

Based on feature names, the app infers input types:
```
Feature Contains        → Input Type
is_, has_, bool, flag   → Checkbox (boolean)
count, number, qty      → Integer spinner
age, id, age            → Integer spinner
(default)               → Float slider
```

### Model Information Display

For each uploaded model, the app shows:
- **Type**: Model class name
- **Module**: Which library provides it
- **Prediction Capability**: Has `predict()` method
- **Classification Capability**: Has `predict_proba()` method
- **Features**: Number and names of expected features

## Troubleshooting

### "Could not automatically detect feature names"
**Solution**: Manually enter feature names in the text input (comma-separated)

### "Error loading model"
**Solution**: Ensure your file is a valid pickle file containing a trained model

### "Prediction Error"
**Solution**: Check that:
- All required features are provided
- Feature values are in expected ranges
- Model was trained with similar data

## Deployment

To deploy this app:

1. **Streamlit Cloud**:
   - Push to GitHub
   - Connect to [Streamlit Cloud](https://streamlit.io/cloud)
   - Select `app.py` as the main file

2. **Docker**:
   ```dockerfile
   FROM python:3.9-slim
   RUN pip install -r requirements.txt
   CMD ["streamlit", "run", "app.py"]
   ```

3. **Local/Server**:
   ```bash
   streamlit run app.py --server.port 8501
   ```

## Tips for Best Results

- **Feature Scaling**: If your model expects scaled features, scale inputs accordingly
- **Categorical Features**: For text/categorical features, the model should handle encoding
- **Missing Values**: Ensure all required features are provided
- **Feature Format**: Use correct data types (int, float, bool) matching training data

## Sharing Models

To share a trained model with others:

1. Save as pickle file:
   ```python
   import pickle
   with open('my_model.pkl', 'wb') as f:
       pickle.dump(model, f)
   ```

2. Share the `.pkl` file with others
3. They can upload it to this app and make predictions immediately

## License

This project is for demonstration purposes.

## Contact

For questions or feedback, please contact the development team.

