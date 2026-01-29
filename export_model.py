"""
Model Export Script for Bus Price Prediction
Copies model files to production directory and creates a simple wrapper
"""

import shutil
import json
from datetime import datetime
from pathlib import Path

def export_model():
    """Export the latest trained model with all dependencies"""
    
    print("=" * 60)
    print("Bus Price Prediction Model Export")
    print("=" * 60)
    
    # Define paths
    saved_runs_dir = Path("models/saved_runs")
    production_dir = Path("models/production")
    
    # Create production directory if it doesn't exist
    production_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Production directory: {production_dir}")
    
    # Find the latest model files
    model_files = sorted(saved_runs_dir.glob("bus_price_model_XGBoost_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No model files found in models/saved_runs/")
    
    latest_model_file = model_files[-1]
    timestamp = latest_model_file.stem.split("_")[-2] + "_" + latest_model_file.stem.split("_")[-1]
    
    print(f"\n✓ Latest model: {latest_model_file.name}")
    print(f"  Timestamp: {timestamp}") 
    
    # Copy model artifacts to production
    print("\nCopying model artifacts to production...")
    
    # 1. Copy model
    model_dest = production_dir / "bus_price_model.pkl"
    shutil.copy2(latest_model_file, model_dest)
    print(f"  ✓ Model copied: {model_dest.name}")
    
    # 2. Copy categorical encoder
    encoder_file = saved_runs_dir / f"categorical_encoder_{timestamp}.pkl"
    encoder_dest = production_dir / "categorical_encoder.pkl"
    shutil.copy2(encoder_file, encoder_dest)
    print(f"  ✓ Encoder copied: {encoder_dest.name}")
    
    # 3. Copy feature engineer (if exists)
    feature_engineer_file = saved_runs_dir / f"feature_engineer_{timestamp}.pkl"
    if feature_engineer_file.exists():
        fe_dest = production_dir / "feature_engineer.pkl"
        shutil.copy2(feature_engineer_file, fe_dest)
        print(f"  ✓ Feature engineer copied: {fe_dest.name}")
    
    # 4. Copy feature names
    feature_names_file = saved_runs_dir / f"feature_names_{timestamp}.json"
    fn_dest = production_dir / "feature_names.json"
    shutil.copy2(feature_names_file, fn_dest)
    print(f"  ✓ Feature names copied: {fn_dest.name}")
    
    # 5. Copy model metrics
    metrics_file = saved_runs_dir / f"model_metrics_{timestamp}.json"
    metrics_dest = production_dir / "model_metrics.json"
    shutil.copy2(metrics_file, metrics_dest)
    print(f"  ✓ Model metrics copied: {metrics_dest.name}")
    
    # Load metrics for display
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Load feature names
    with open(feature_names_file, 'r') as f:
        feature_names = json.load(f)
    
    # Create metadata file
    metadata = {
        'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training_timestamp': timestamp,
        'model_type': 'XGBoost',
        'version': '1.0',
        'num_features': len(feature_names),
        'files': {
            'model': 'bus_price_model.pkl',
            'encoder': 'categorical_encoder.pkl',
            'feature_engineer': 'feature_engineer.pkl',
            'feature_names': 'feature_names.json',
            'metrics': 'model_metrics.json'
        }
    }
    
    metadata_path = production_dir / "model_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata created: {metadata_path.name}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Model Type: {metrics['model_name']}")
    print(f"Training Date: {timestamp}")
    print(f"Export Date: {metadata['export_date']}")
    print(f"\nPerformance Metrics:")
    print(f"  R² Score: {metrics['metrics']['r2_log']:.4f}")
    print(f"  RMSE (INR): ₹{metrics['metrics']['rmse_inr']:.2f}")
    print(f"  MAE (INR): ₹{metrics['metrics']['mae_inr']:.2f}")
    print(f"  MAPE: {metrics['metrics']['mape_pct']:.2f}%")
    print(f"\nTraining Data:")
    print(f"  Training samples: {metrics['training_samples']:,}")
    print(f"  Testing samples: {metrics['testing_samples']:,}")
    print(f"  Number of features: {metrics['num_features']}")
    print(f"\nProduction Directory: {production_dir.absolute()}")
    print("=" * 60)
    print("\n✓ Model export completed successfully!")
    
    return production_dir

if __name__ == "__main__":
    try:
        export_model()
    except Exception as e:
        print(f"\n✗ Error during export: {str(e)}")
        raise
