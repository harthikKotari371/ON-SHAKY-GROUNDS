import os

base_dir = r"d:\GRIND\projects\earthquake prediction (ml)"

files = {
    "src/__init__.py": "",
    "src/models/__init__.py": "",
    
    "src/data_preprocessing.py": """import pandas as pd
import numpy as np
import os

def load_data(base_path="."):
    query_path = os.path.join(base_path, "query.csv")
    output_path = os.path.join(base_path, "output.csv")
    
    query_df = pd.read_csv(query_path) if os.path.exists(query_path) else pd.DataFrame()
    output_df = pd.read_csv(output_path, skiprows=[1]) if os.path.exists(output_path) else pd.DataFrame()
    return query_df, output_df

def preprocess_query(df):
    if df.empty: return df
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce', utc=True)
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour
    df['day_of_year'] = df['time'].dt.dayofyear
    
    df['depth'] = df['depth'].fillna(df['depth'].median())
    df['mag'] = df['mag'].fillna(df['mag'].median())
    df['gap'] = pd.to_numeric(df['gap'], errors='coerce').fillna(0)
    df['dmin'] = pd.to_numeric(df['dmin'], errors='coerce').fillna(0)
    df['rms'] = pd.to_numeric(df['rms'], errors='coerce').fillna(0)
    df['nst'] = pd.to_numeric(df['nst'], errors='coerce').fillna(0)
    
    return df

def preprocess_output(df):
    if df.empty: return df
    df = df.copy()
    df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce').fillna(0)
    df['Damage ($Mil)'] = pd.to_numeric(df['Damage ($Mil)'], errors='coerce').fillna(0)
    df['MMI Int'] = pd.to_numeric(df['MMI Int'], errors='coerce').fillna(0)
    df['Mag'] = pd.to_numeric(df['Mag'], errors='coerce').fillna(df['Mag'].median())
    df['Focal Depth (km)'] = pd.to_numeric(df['Focal Depth (km)'], errors='coerce').fillna(df['Focal Depth (km)'].median())
    
    severity_score = df['Deaths'] * 10 + df['Damage ($Mil)'] + df['MMI Int'] * 5
    
    def classify_severity(score):
        if score < 10: return 'Low'
        elif score < 50: return 'Medium'
        elif score < 200: return 'High'
        else: return 'Catastrophic'
        
    df['Severity'] = severity_score.apply(classify_severity)
    df['Tsu'] = df['Tsu'].notna().astype(int)
    
    return df

if __name__ == '__main__':
    q, o = load_data()
    print('Loaded:', q.shape, o.shape)
""",

    "src/models/magnitude_predictor.py": """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_magnitude_models(df):
    if df.empty: return {}
    features = ['latitude', 'longitude', 'depth', 'gap', 'dmin', 'rms', 'year', 'month', 'hour', 'day_of_year', 'nst']
    X = df[features].fillna(0)
    y = df['mag'].fillna(5.0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10),
        'XGBoost': GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5),
        'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, preds))),
            'mae': float(mean_absolute_error(y_test, preds)),
            'r2': float(r2_score(y_test, preds))
        }
        if hasattr(model, 'feature_importances_'):
            results[name]['feature_importances'] = {f: float(imp) for f, imp in zip(features, model.feature_importances_)}
            
    return results
""",

    "src/models/severity_classifier.py": """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train_severity_models(df):
    if df.empty: return {}
    features = ['Mag', 'Focal Depth (km)', 'Latitude', 'Longitude', 'MMI Int', 'Tsu', 'Year']
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(2000)
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['Severity']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
        'XGBoost': GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=5),
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, preds, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, preds, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, preds).tolist()
        }
    results['classes'] = le.classes_.tolist()
    return results
""",


    "src/models/clustering_analysis.py": """import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_clustering(df):
    if df.empty: return {}
    sample_df = df.sample(min(10000, len(df)), random_state=42).copy()
    features = ['latitude', 'longitude', 'depth', 'mag']
    X = sample_df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    for k in [5, 7, 10]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        stats = []
        for i in range(k):
            cluster_mask = (labels == i)
            stats.append({
                'cluster_id': i,
                'count': int(np.sum(cluster_mask)),
                'avg_mag': float(sample_df.loc[cluster_mask, 'mag'].mean()),
                'avg_depth': float(sample_df.loc[cluster_mask, 'depth'].mean())
            })
            
        results[f'kmeans_{k}'] = {
            'centers': centers.tolist(),
            'stats': stats
        }
    return results
""",

    "src/models/time_series_analysis.py": """import pandas as pd
import numpy as np

def run_time_series_analysis(df):
    if df.empty: return {}
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)
    
    monthly_counts = df.resample('ME').size()
    monthly_mag_mean = df['mag'].resample('ME').mean().fillna(0)
    
    depths = df['depth'].dropna()
    mags = df['mag'].dropna()
    
    results = {
        'monthly_counts': {
            'dates': [d.strftime('%Y-%m') for d in monthly_counts.index],
            'counts': monthly_counts.tolist()
        },
        'monthly_mag_mean': {
            'dates': [d.strftime('%Y-%m') for d in monthly_mag_mean.index],
            'mag_mean': monthly_mag_mean.tolist()
        },
        'depth_distribution': np.histogram(depths, bins=20)[0].tolist() if not depths.empty else [],
        'depth_bins': np.histogram(depths, bins=20)[1].tolist() if not depths.empty else [],
        'mag_distribution': np.histogram(mags, bins=20)[0].tolist() if not mags.empty else [],
        'mag_bins': np.histogram(mags, bins=20)[1].tolist() if not mags.empty else []
    }
    return results
""",

    "src/generate_results.py": """import os
import json
import numpy as np
from src.data_preprocessing import load_data, preprocess_query, preprocess_output
from src.models.magnitude_predictor import train_magnitude_models
from src.models.severity_classifier import train_severity_models
from src.models.clustering_analysis import run_clustering
from src.models.time_series_analysis import run_time_series_analysis

def convert_np(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_np(v) for v in obj]
    else: return obj

def main():
    print("Starting pipeline...")
    base_path = "."
    out_dir = os.path.join(base_path, "website", "data")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data...")
    q_df, o_df = load_data(base_path)
    
    print("Preprocessing data...")
    q_df = preprocess_query(q_df)
    o_df = preprocess_output(o_df)
    
    dataset_stats = {
        'query': {'rows': len(q_df), 'cols': len(q_df.columns)},
        'output': {'rows': len(o_df), 'cols': len(o_df.columns)},
        'intensity': {'rows': len(i_df), 'cols': len(i_df.columns)}
    }
    with open(os.path.join(out_dir, "dataset_stats.json"), "w") as f:
        json.dump(convert_np(dataset_stats), f)
        
    print("Running Magnitude Predictor...")
    mag_results = train_magnitude_models(q_df)
    with open(os.path.join(out_dir, "magnitude_results.json"), "w") as f:
        json.dump(convert_np(mag_results), f)
        
    print("Running Severity Classifier...")
    sev_results = train_severity_models(o_df)
    with open(os.path.join(out_dir, "severity_results.json"), "w") as f:
        json.dump(convert_np(sev_results), f)
        
        json.dump(convert_np(int_results), f)
        
    print("Running Clustering Analysis...")
    clus_results = run_clustering(q_df)
    with open(os.path.join(out_dir, "clustering_results.json"), "w") as f:
        json.dump(convert_np(clus_results), f)
        
    print("Running Time Series Analysis...")
    ts_results = run_time_series_analysis(q_df)
    with open(os.path.join(out_dir, "timeseries_results.json"), "w") as f:
        json.dump(convert_np(ts_results), f)
        
    print("Generating Model Comparison...")
    comp_results = {
        'magnitude': {k: v['rmse'] for k, v in mag_results.items()},
        'severity': {k: v['accuracy'] for k, v in sev_results.items() if k != 'classes'},
    }
    with open(os.path.join(out_dir, "model_comparison.json"), "w") as f:
        json.dump(convert_np(comp_results), f)
        
    print("Generating Map Data...")
    if not q_df.empty:
        map_df = q_df.sample(min(2000, len(q_df)), random_state=42)
        map_data = map_df[['latitude', 'longitude', 'mag', 'depth', 'place']].to_dict(orient='records')
        map_data_times = map_df.index.astype(str).tolist() if map_df.index.name == 'time' else map_df['time'].astype(str).tolist()
        for i, md in enumerate(map_data): md['time'] = map_data_times[i]
    else:
        map_data = []
    with open(os.path.join(out_dir, "earthquake_map_data.json"), "w") as f:
        json.dump(convert_np(map_data), f)
        
    print("Generating Risk Analysis...")
    risk_results = {}
    if not q_df.empty:
        risk_results['mag_dist'] = ts_results.get('mag_distribution', [])
        risk_results['depth_dist'] = ts_results.get('depth_distribution', [])
    with open(os.path.join(out_dir, "risk_analysis.json"), "w") as f:
        json.dump(convert_np(risk_results), f)
        
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
""",

    "run_pipeline.py": """import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    from src.generate_results import main
    try:
        main()
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        sys.exit(1)
"""
}

for rel, content in files.items():
    p = os.path.join(base_dir, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
