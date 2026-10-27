import pandas as pd
import json
import os

base_path = "."
data_dir = os.path.join("website", "data")
os.makedirs(data_dir, exist_ok=True)

samples = {}

# 1. USGS Query
query_path = "query.csv"
if os.path.exists(query_path):
    df = pd.read_csv(query_path, nrows=20)
    samples['query.csv'] = {'columns': list(df.columns), 'data': df.astype(str).fillna("").values.tolist()}

# 2. NOAA Significant
output_path = "output.csv"
if os.path.exists(output_path):
    df = pd.read_csv(output_path, skiprows=[1], nrows=20)
    samples['output.csv'] = {'columns': list(df.columns), 'data': df.astype(str).fillna("").values.tolist()}

# 3. USGS Macroseismic Intensity
intensity_path = "eqint_tsqp.csv"
if os.path.exists(intensity_path):
    df = pd.read_csv(intensity_path, nrows=20, low_memory=False)
    samples['eqint_tsqp.csv'] = {'columns': list(df.columns), 'data': df.astype(str).fillna("").values.tolist()}

# 4. India Regional Catalog
india_path = os.path.join("archive", "Earthquakes.csv")
if os.path.exists(india_path):
    df = pd.read_csv(india_path, nrows=20)
    samples['Earthquakes.csv'] = {'columns': list(df.columns), 'data': df.astype(str).fillna("").values.tolist()}

out_file = os.path.join(data_dir, "dataset_samples.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(samples, f)
print(f"All 4 dataset samples generated at {out_file}! Keys: {list(samples.keys())}")
