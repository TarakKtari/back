# scheduler_jobs.py
import os
import re
import requests

# Track the last processed timestamps
# (we keep separate variables for midmarket & bctx)
last_processed_timestamp_midmarket = None
last_processed_timestamp_bctx = None

def check_for_new_files():
    global last_processed_timestamp_midmarket
    global last_processed_timestamp_bctx

    data_dir = "/app/data"  # Where your JSON files live in Docker

    try:
        all_files = os.listdir(data_dir)
    except Exception as e:
        print(f"Error accessing directory: {e}")
        return

    # ============================
    # 1) Check for Midmarket Files
    # ============================
    mid_pattern = re.compile(r"midmarket_rates_(\d{4}-\d{2}-\d{2}_\d{4})\.json")
    mid_timestamps = []
    for filename in all_files:
        match = mid_pattern.search(filename)
        if match:
            mid_timestamps.append(match.group(1))

    if mid_timestamps:
        latest_mid_ts = max(mid_timestamps)
        if (last_processed_timestamp_midmarket is None 
                or latest_mid_ts != last_processed_timestamp_midmarket):
            print(f"New midmarket file detected: {latest_mid_ts} "
                  f"(last processed: {last_processed_timestamp_midmarket})")
            last_processed_timestamp_midmarket = latest_mid_ts

            # Trigger the upsert-exchange-data endpoint
            try:
                response = requests.post("http://localhost:5001/admin/api/upsert-exchange-data")
                print("Triggered upsert-exchange-data, response:", response.json())
            except Exception as api_err:
                print("Error triggering upsert-exchange-data:", api_err)
        else:
            print("No new midmarket files detected.")
    else:
        print("No matching midmarket files found at all.")

    # =====================
    # 2) Check for BCTX Files
    # =====================
    bctx_pattern = re.compile(r"bctx_data_(\d{4}-\d{2}-\d{2}_\d{4})\.json")
    bctx_timestamps = []
    for filename in all_files:
        match = bctx_pattern.search(filename)
        if match:
            bctx_timestamps.append(match.group(1))

    if bctx_timestamps:
        latest_bctx_ts = max(bctx_timestamps)
        if (last_processed_timestamp_bctx is None 
                or latest_bctx_ts != last_processed_timestamp_bctx):
            print(f"New BCTX file detected: {latest_bctx_ts} "
                  f"(last processed: {last_processed_timestamp_bctx})")
            last_processed_timestamp_bctx = latest_bctx_ts

            # Trigger the upsert-bctx endpoint
            try:
                response = requests.post("http://localhost:5001/admin/api/upsert-bctx")
                print("Triggered upsert-bctx, response:", response.json())
            except Exception as api_err:
                print("Error triggering upsert-bctx:", api_err)
        else:
            print("No new BCTX files detected.")
    else:
        print("No matching BCTX files found at all.")