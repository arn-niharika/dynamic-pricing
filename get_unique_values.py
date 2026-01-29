import pandas as pd
from pathlib import Path

# Read a sample of the data
data_dir = Path("data/complete_data")
df = pd.read_parquet(data_dir / "data_part_0.parquet")

# Get unique values using correct column names
if 'travels' in df.columns:
    operators = sorted(df['travels'].dropna().unique().tolist())
    print("=== OPERATORS ===")
    for op in operators[:25]:  # First 25
        print(f'    "{op}",')
    if len(operators) > 25:
        print(f"... Total: {len(operators)} operators")

if 'busType' in df.columns:
    bus_types = sorted(df['busType'].dropna().unique().tolist())
    print("\n=== BUS TYPES ===")
    for bt in bus_types[:20]:  # First 20
        print(f'    "{bt}",')
    if len(bus_types) > 20:
        print(f"... Total: {len(bus_types)} bus types")

if 'source_collection' in df.columns:
    routes = sorted(df['source_collection'].dropna().unique().tolist())
    print("\n=== ROUTES ===")
    for route in routes:
        print(f'    "{route}",')
