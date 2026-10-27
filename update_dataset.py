"""
Live Earthquake Dataset Updater
================================
Fetches real-time earthquake data from the USGS (United States Geological Survey) API,
deduplicates against existing records, and updates both:
  1. `query.csv` (the primary ML training and evaluation catalog)
  2. `query.xlsx` (Excel spreadsheet dataset for direct viewing/editing)

Usage:
  # One-time update:
  python update_dataset.py

  # Update and re-run ML pipeline to refresh website dashboard:
  python update_dataset.py --pipeline

  # Run continuously in background (checks every 15 minutes by default):
  python update_dataset.py --loop --interval 900
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import argparse
import urllib.request
import urllib.parse
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(PROJECT_ROOT, "query.csv")
EXCEL_FILE = os.path.join(PROJECT_ROOT, "query.xlsx")
USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_latest_usgs_events(start_time_iso, min_magnitude=5.0):
    """
    Fetch new earthquakes from the USGS API occurring after start_time_iso
    with magnitude >= min_magnitude.
    """
    params = {
        "format": "csv",
        "starttime": start_time_iso,
        "minmagnitude": min_magnitude,
        "orderby": "time"
    }
    url = f"{USGS_API_URL}?{urllib.parse.urlencode(params)}"
    print(f"  [+] Querying USGS API: {url}")
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "EarthquakeML-LiveUpdater/1.0 (Academic/Research Project)"}
    )
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8")
        
    df_new = pd.read_csv(pd.io.common.StringIO(content))
    return df_new


def update_dataset(min_magnitude=5.0, export_excel=True, run_ml_pipeline=False):
    """
    Syncs query.csv and query.xlsx with the latest live earthquakes from USGS.
    """
    print("=" * 60)
    print("  EARTHQUAKE DATASET LIVE UPDATER (USGS -> CSV / EXCEL)")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found!")
        return 0

    # 1. Load existing dataset
    print(f"\n[1/4] Reading existing dataset: {os.path.basename(CSV_FILE)}...")
    existing_df = pd.read_csv(CSV_FILE)
    initial_count = len(existing_df)
    print(f"  Existing records: {initial_count:,}")

    # Determine latest timestamp
    time_series = pd.to_datetime(existing_df['time'], format='mixed', errors='coerce', utc=True)
    latest_time = time_series.max()
    latest_iso = latest_time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"  Latest record in dataset: {latest_iso} UTC")

    # 2. Fetch new earthquakes
    print(f"\n[2/4] Fetching real-time earthquakes (M >= {min_magnitude}) since {latest_iso}...")
    try:
        new_df = fetch_latest_usgs_events(latest_iso, min_magnitude=min_magnitude)
        print(f"  Received {len(new_df)} candidate records from USGS.")
    except Exception as e:
        print(f"  ❌ Error fetching from USGS API: {e}")
        return 0

    if new_df.empty:
        print("  ✓ Dataset is already 100% up-to-date with current real-time earthquake feeds.")
        new_records_added = 0
    else:
        # 3. Deduplicate
        existing_ids = set(existing_df['id'].dropna().astype(str))
        filtered_new = new_df[~new_df['id'].astype(str).isin(existing_ids)].copy()
        new_records_added = len(filtered_new)

        if new_records_added == 0:
            print("  ✓ No new unique earthquakes found (all candidate IDs already present).")
        else:
            print(f"  Found {new_records_added:,} NEW unique earthquakes to insert!")
            
            # Combine: new records placed at top (or sorted chronologically descending)
            combined_df = pd.concat([filtered_new, existing_df], ignore_index=True)
            
            # Sort descending by time
            combined_df['time_dt'] = pd.to_datetime(combined_df['time'], format='mixed', errors='coerce', utc=True)
            combined_df = combined_df.sort_values(by='time_dt', ascending=False).drop(columns=['time_dt'])
            
            # Save updated CSV
            combined_df.to_csv(CSV_FILE, index=False)
            print(f"  ✓ Updated {os.path.basename(CSV_FILE)}: {len(combined_df):,} total rows (+{new_records_added})")

    # 4. Export / Update Excel dataset
    if export_excel:
        print(f"\n[3/4] Exporting to Excel workbook: {os.path.basename(EXCEL_FILE)}...")
        try:
            # Refresh df from CSV to ensure clean formatting
            full_df = pd.read_csv(CSV_FILE)
            # Write to Excel with openpyxl
            full_df.to_excel(EXCEL_FILE, sheet_name="USGS_Earthquakes", index=False)
            size_mb = os.path.getsize(EXCEL_FILE) / (1024 * 1024)
            print(f"  ✓ Saved {os.path.basename(EXCEL_FILE)} ({size_mb:.2f} MB, {len(full_df):,} rows)")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not save Excel file: {e}")

    # 5. Optionally run ML pipeline
    if run_ml_pipeline and new_records_added > 0:
        print("\n[4/4] Triggering ML training and dashboard update pipeline...")
        try:
            from src.generate_results import main as run_pipeline
            run_pipeline()
            print("  ✓ Machine learning models & dashboard data successfully updated!")
        except Exception as e:
            print(f"  ❌ Error executing ML pipeline: {e}")
    else:
        print(f"\n[4/4] Status check complete. Dataset is synchronized.")

    print(f"\nDone! Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    return new_records_added


def run_continuous_loop(interval_sec=900, min_magnitude=5.0, run_ml_pipeline=False):
    """
    Runs continuous background polling to update datasets every interval_sec.
    """
    print(f"Starting continuous real-time updater (interval: {interval_sec} seconds / {interval_sec/60:.1f} minutes)...")
    print("Press Ctrl+C to terminate.")
    while True:
        try:
            update_dataset(
                min_magnitude=min_magnitude,
                export_excel=True,
                run_ml_pipeline=run_ml_pipeline
            )
        except Exception as e:
            print(f"⚠️ Exception during update cycle: {e}")
            
        print(f"Sleeping for {interval_sec} seconds before next USGS check...\n")
        time.sleep(interval_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Earthquake Dataset Updater (USGS)")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a background loop")
    parser.add_argument("--interval", type=int, default=900, help="Check interval in seconds (default: 900s / 15 min)")
    parser.add_argument("--minmag", type=float, default=5.0, help="Minimum earthquake magnitude (default: 5.0)")
    parser.add_argument("--no-excel", action="store_true", help="Skip updating the Excel (.xlsx) file")
    parser.add_argument("--pipeline", action="store_true", help="Automatically retrain ML models and update web dashboard JSONs")

    args = parser.parse_args()

    if args.loop:
        run_continuous_loop(
            interval_sec=args.interval,
            min_magnitude=args.minmag,
            run_ml_pipeline=args.pipeline
        )
    else:
        update_dataset(
            min_magnitude=args.minmag,
            export_excel=not args.no_excel,
            run_ml_pipeline=args.pipeline
        )
