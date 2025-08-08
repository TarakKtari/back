import os
import datetime
import json
import eikon as ek
import pandas as pd
import refinitiv.data as rd
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set Eikon app key and port number
ek.set_app_key("20af0572a6364fe8abf9a35cdd16bd367057564a")
ek.set_port_number(9000)  # Default proxy port for Eikon Desktop

# Open a session (ensure Eikon or Workspace is running)
rd.open_session()


# 4) Define BCTX RICs
bctx_rics = [
    "TND=BCTX",
    "EURTNDX=BCTX",
    "GBPTNDX=BCTX",
    "JPYTNDX=BCTX"
]

# 5) Date Range
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime.today()

# 6) Ensure data folder (mapped to /app/data in Docker)
data_dir = "data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 7) Generate a timestamp
timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

try:
    # 8) Fetch 1-min data
    df = rd.get_history(
        universe=bctx_rics,
        fields=["BID", "ASK"],
        start=start_date,
        end=end_date,
        interval="1min"
    )

    if df.empty:
        print("No data available for BCTX in the specified range.")
    else:
        print("BCTX data retrieved successfully.")

        # Flatten index
        df = df.reset_index()

        # 9) Build a filename like bctx_data_2024-10-30_1630.json
        filename = f"bctx_data_{timestamp_str}.json"
        full_path = os.path.join(data_dir, filename)

        # 10) Write to JSON
        df.to_json(full_path, orient="records", date_format="iso")
        print(f"BCTX data saved to {full_path}")

except Exception as e:
    print(f"Error fetching BCTX data: {e}")

# 11) Close session
rd.close_session()
