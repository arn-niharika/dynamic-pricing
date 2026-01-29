"""
Simple script to inspect the encoder pickle file
"""
import pickle
import sys

encoder_file = "models/saved_runs/categorical_encoder_20260127_104450.pkl"

try:
    with open(encoder_file, 'rb') as f:
        data = pickle.load(f)
    print(f"Successfully loaded: {type(data)}")
    print(f"Attributes: {dir(data)}")
    if hasattr(data, 'encoders'):
        print(f"\nEncoders dict keys: {data.encoders.keys()}")
        for key, encoder in data.encoders.items():
            print(f"\n{key}:")
            print(f"  Type: {type(encoder)}")
            print(f"  Classes: {len(encoder.classes_)} unique values")
            print(f"  Sample classes: {list(encoder.classes_[:5])}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
