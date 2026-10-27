"""
Earthquake Prediction Pipeline — Master Entry Point
Run this script from the project root:
    python run_pipeline.py
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time

def main():
    start = time.time()
    
    # Ensure we run from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    sys.path.insert(0, project_root)
    
    print("=" * 58)
    print("  EARTHQUAKE PREDICTION & DETECTION SYSTEM")
    print("  Machine Learning Pipeline v2.0")
    print("=" * 58)
    print()
    
    # Check datasets exist
    datasets = ['query.csv', 'output.csv', 'eqint_tsqp.csv']
    for ds in datasets:
        if not os.path.exists(ds):
            print(f"❌ Missing dataset: {ds}")
            sys.exit(1)
        else:
            size_mb = os.path.getsize(ds) / (1024 * 1024)
            print(f"  ✓ Found {ds} ({size_mb:.1f} MB)")
    
    print()
    
    try:
        from src.generate_results import main as run_pipeline
        run_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    elapsed = time.time() - start
    print(f"\n⏱  Total time: {elapsed:.1f} seconds")
    print(f"\n🌐 To view the dashboard, open:")
    print(f"   {os.path.join(project_root, 'website', 'index.html')}")
    print(f"\n   Or start a local server:")
    print(f"   cd website && python -m http.server 8000")
    print(f"   Then visit: http://localhost:8000")


if __name__ == "__main__":
    main()
