import os
import datetime
import json
import eikon as ek
import pandas as pd
import refinitiv.data as rd

# Set Eikon app key and port number
ek.set_app_key("20af0572a6364fe8abf9a35cdd16bd367057564a")
ek.set_port_number(9000)  # Default proxy port for Eikon Desktop

# Open session
rd.open_session()

# Define RICs for Midmarket rates and Yields
rics = ['EUR02H=', 'TND02H=']
yield_rics = [
    # USD
    'USDOND=', 'USD1MD=', 'USD3MD=', 'USD6MD=', 'USD1YD=',
    
    # EUR
    'EUROND=', 'EUR1MD=', 'EUR3MD=', 'EUR6MD=', 'EUR1YD=',
    
    # TND
    'TNDOND=', 'TND1MD=', 'TND3MD=', 'TND6MD=', 'TND1YD='
]

# Set start and end dates
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime.today()

# Create the "data" folder if it doesn't exist
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Get a timestamp string with date and time (e.g., "2025-02-18_0930")
timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

try:
    ###### Fetch and Process Midmarket Rates ######
    df = rd.get_history(
        universe=rics,
        fields=['BID', 'ASK'],
        start=start_date,
        end=end_date,
        interval='hourly'
    )

    if df.empty:
        print("No data available for Midmarket rates.")
    else:
        print("Midmarket rates data retrieved successfully.")
        df = df.reset_index()
        df.columns = ['Timestamp', 'Bid_EUR02H', 'Ask_EUR02H', 'Bid_TND02H', 'Ask_TND02H']
        df['Mid_EUR02H'] = (df['Bid_EUR02H'] + df['Ask_EUR02H']) / 2
        df['Mid_TND02H'] = (df['Bid_TND02H'] + df['Ask_TND02H']) / 2
        df['spotUSD'] = df['Mid_TND02H']
        df['spotEUR'] = df['Mid_TND02H'] * df['Mid_EUR02H']
        df = df.infer_objects()

        # Save to JSON with timestamp in the filename
        result_json = df[['Timestamp', 'spotUSD', 'spotEUR']].to_json(orient="records", date_format="iso")
        mid_filename = os.path.join(data_dir, f"midmarket_rates_{timestamp_str}.json")
        with open(mid_filename, "w") as f:
            f.write(result_json)
        print(f"Midmarket rates saved to {mid_filename}.")

    ###### Fetch and Process Yield Data ######
     # fetch daily yields (no DATE field needed—you get the date as the index)
    yield_df = rd.get_history(
        universe=yield_rics,
        fields=['BID', 'ASK'],
        start=start_date,
        end=end_date,
        interval='daily'
    )

    # name the index 'Timestamp' so reset_index() will give you exactly that column
    yield_df.index      = pd.to_datetime(yield_df.index)
    yield_df.index.name = 'Timestamp'
    yield_df = yield_df.reset_index()

    if yield_df.empty:
        print("No data available for Yields.")
    else:
        print("Yield data retrieved successfully.")
        yield_df = yield_df.reset_index()
        yield_df.columns = [
            '_'.join(col).strip() if isinstance(col, tuple) else col
            for col in yield_df.columns
        ]
        if 'Timestamp' not in yield_df.columns and 'Timestamp_' in yield_df.columns:
            yield_df.rename(columns={'Timestamp_': 'Timestamp'}, inplace=True)

        # Convert to date only for daily grouping
        yield_df['Timestamp'] = pd.to_datetime(yield_df['Timestamp']).dt.date

        # Compute mid yields
        # USD yields
        yield_df['Mid_USDOND'] = (yield_df['USDOND=_BID'] + yield_df['USDOND=_ASK']) / 2
        yield_df['Mid_USD1M'] = (yield_df['USD1MD=_BID'] + yield_df['USD1MD=_ASK']) / 2
        yield_df['Mid_USD3M'] = (yield_df['USD3MD=_BID'] + yield_df['USD3MD=_ASK']) / 2
        yield_df['Mid_USD6M'] = (yield_df['USD6MD=_BID'] + yield_df['USD6MD=_ASK']) / 2
        yield_df['Mid_USD1Y'] = (yield_df['USD1YD=_BID'] + yield_df['USD1YD=_ASK']) / 2
        
        # EUR yields
        yield_df['Mid_EUROND'] = (yield_df['EUROND=_BID'] + yield_df['EUROND=_ASK']) / 2
        yield_df['Mid_EUR1M'] = (yield_df['EUR1MD=_BID'] + yield_df['EUR1MD=_ASK']) / 2
        yield_df['Mid_EUR3M'] = (yield_df['EUR3MD=_BID'] + yield_df['EUR3MD=_ASK']) / 2
        yield_df['Mid_EUR6M'] = (yield_df['EUR6MD=_BID'] + yield_df['EUR6MD=_ASK']) / 2
        yield_df['Mid_EUR1Y'] = (yield_df['EUR1YD=_BID'] + yield_df['EUR1YD=_ASK']) / 2
        
        # TND yields
        yield_df['Mid_TNDOND'] = (yield_df['TNDOND=_BID'] + yield_df['TNDOND=_ASK']) / 2
        yield_df['Mid_TND1M'] = (yield_df['TND1MD=_BID'] + yield_df['TND1MD=_ASK']) / 2
        yield_df['Mid_TND3M'] = (yield_df['TND3MD=_BID'] + yield_df['TND3MD=_ASK']) / 2
        yield_df['Mid_TND6M'] = (yield_df['TND6MD=_BID'] + yield_df['TND6MD=_ASK']) / 2
        yield_df['Mid_TND1Y'] = (yield_df['TND1YD=_BID'] + yield_df['TND1YD=_ASK']) / 2

        # Helper function: select the first non-null value in a series
        def first_non_null(series):
            non_null = series.dropna()
            return non_null.iloc[0] if not non_null.empty else None

        # Group by date and take first non-null for each column
        daily_yield = yield_df.groupby('Timestamp').agg({
            'Mid_USDOND': first_non_null,
            'Mid_USD1M': first_non_null,
            'Mid_USD3M': first_non_null,
            'Mid_USD6M': first_non_null,
            'Mid_USD1Y': first_non_null,
            'Mid_EUROND': first_non_null,
            'Mid_EUR1M': first_non_null,
            'Mid_EUR3M': first_non_null,
            'Mid_EUR6M': first_non_null,
            'Mid_EUR1Y': first_non_null,
            'Mid_TNDOND': first_non_null,
            'Mid_TND1M': first_non_null,
            'Mid_TND3M': first_non_null,
            'Mid_TND6M': first_non_null,
            'Mid_TND1Y': first_non_null
        }).reset_index()

        # ***** NEW STEP: Convert percentages to decimal values *****
        # (Divide each yield column by 100)
        for col in ['Mid_USDOND','Mid_USD1M','Mid_USD3M','Mid_USD6M','Mid_USD1Y',
                    'Mid_EUROND','Mid_EUR1M','Mid_EUR3M','Mid_EUR6M','Mid_EUR1Y',
                    'Mid_TNDOND','Mid_TND1M','Mid_TND3M','Mid_TND6M','Mid_TND1Y']:
            daily_yield[col] = daily_yield[col] / 100

        # Convert final DataFrame to JSON
        daily_yield_json = daily_yield.to_json(orient="records", date_format="iso")
        yield_filename = os.path.join(data_dir, f"daily_yield_rates_{timestamp_str}.json")
        with open(yield_filename, "w") as f:
            f.write(daily_yield_json)
        print(f"Yield rates saved to {yield_filename}.")

except Exception as e:
    print(f"Error fetching data: {e}")

# -------- Fetch Historical ATM Vols for EURUSD --------
print("\n📈 Fetching historical ATM vols for EURUSD...")
atm_rics = ["EUR1WO=R", "EUR1MO=", "EUR2MO=", "EUR3MO=", "EUR6MO=", "EUR9MO=", "EUR1YO="]
try:
    hist_atm_df = rd.get_history(
        universe=atm_rics,
        fields=["BID", "ASK", "CF_LAST"],
        start=start_date,
        end=end_date,
        interval="daily"
    )
    hist_atm_df = hist_atm_df.reset_index()
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    hist_atm_filename = os.path.join(data_dir, f"eur_usd_hist_atm_vols_{timestamp_str}.json")
    hist_atm_df.to_json(hist_atm_filename, orient="records", date_format="iso")
    print(f"EUR/USD historical ATM vols saved to {hist_atm_filename}.")
except Exception as e:
    print(f"Error fetching historical ATM vols: {e}")

# Close session
rd.close_session()

#===================================================================================
# import eikon as ek
# import pandas as pd
# import datetime
# import json
# import refinitiv.data as rd

# # Set Eikon app key and port number
# ek.set_app_key("20af0572a6364fe8abf9a35cdd16bd367057564a")
# ek.set_port_number(9000)  # Default proxy port for Eikon Desktop

# # Open session
# rd.open_session()

# # Define RICs for Midmarket rates and Yields
# rics = ['EUR02H=', 'TND02H=']
# yield_rics = ['USD1MD=', 'USD3MD=', 'USD6MD=', 'EUR1MD=', 'EUR3MD=', 'EUR6MD=', 'TNDOND=']

# # Set start and end dates
# start_date = datetime.datetime(2025, 1, 28)
# end_date = datetime.datetime.today()

# try:
#     ###### Fetch and Process Midmarket Rates ######
#     df = rd.get_history(
#         universe=rics,
#         fields=['BID', 'ASK'],
#         start=start_date,
#         end=end_date,
#         interval='hourly'  # Hourly data to capture multiple observations
#     )

#     if df.empty:
#         print("No data available for Midmarket rates.")
#     else:
#         print("Midmarket rates data retrieved successfully.")

#         # Reset index to bring Timestamp into a column
#         df = df.reset_index()

#         # Rename columns (assumes order: Timestamp, BID/ASK for EUR02H, then BID/ASK for TND02H)
#         df.columns = ['Timestamp', 'Bid_EUR02H', 'Ask_EUR02H', 'Bid_TND02H', 'Ask_TND02H']

#         # Calculate mid-prices for each currency pair
#         df['Mid_EUR02H'] = (df['Bid_EUR02H'] + df['Ask_EUR02H']) / 2
#         df['Mid_TND02H'] = (df['Bid_TND02H'] + df['Ask_TND02H']) / 2

#         # Save Mid_TND02H as spotUSD and calculate spotEUR (Mid_TND02H * Mid_EUR02H)
#         df['spotUSD'] = df['Mid_TND02H']
#         df['spotEUR'] = df['Mid_TND02H'] * df['Mid_EUR02H']

#         # Fix downcasting warning by inferring object types
#         df = df.infer_objects()

#         # Save midmarket rates to JSON
#         result_json = df[['Timestamp', 'spotUSD', 'spotEUR']].to_json(orient="records", date_format="iso")
#         with open("data/midmarket_rates.json", "w") as f:
#             f.write(result_json)


#         print("Midmarket rates saved to midmarket_rates.json.")

#     ###### Fetch and Process Yield Data ######
#     # Fetch hourly yield data to later aggregate to one record per day
#     yield_df = rd.get_history(
#         universe=yield_rics,
#         fields=['BID', 'ASK', 'DATE'],
#         start=start_date,
#         end=end_date,
#         interval='hourly'
#     )

#     if yield_df.empty:
#         print("No data available for Yields.")
#     else:
#         print("Yield data retrieved successfully.")

#         # Reset index to include Timestamp as a column
#         yield_df = yield_df.reset_index()

#         # Flatten MultiIndex columns (e.g. ('USD1MD=', 'BID') becomes 'USD1MD=_BID')
#         yield_df.columns = [
#             '_'.join(col).strip() if isinstance(col, tuple) else col
#             for col in yield_df.columns
#         ]

#         # Debug: Print flattened columns to verify names
#         print("Flattened yield data columns:", yield_df.columns)

#         # Ensure we have a proper Timestamp column
#         if 'Timestamp' not in yield_df.columns and 'Timestamp_' in yield_df.columns:
#             yield_df.rename(columns={'Timestamp_': 'Timestamp'}, inplace=True)

#         # Convert Timestamp to date-only for daily grouping
#         yield_df['Timestamp'] = pd.to_datetime(yield_df['Timestamp']).dt.date

#         # Calculate mid-prices for each yield field using the flattened column names
#         yield_df['Mid_USD1M'] = (yield_df['USD1MD=_BID'] + yield_df['USD1MD=_ASK']) / 2
#         yield_df['Mid_USD3M'] = (yield_df['USD3MD=_BID'] + yield_df['USD3MD=_ASK']) / 2
#         yield_df['Mid_USD6M'] = (yield_df['USD6MD=_BID'] + yield_df['USD6MD=_ASK']) / 2
#         yield_df['Mid_EUR1M'] = (yield_df['EUR1MD=_BID'] + yield_df['EUR1MD=_ASK']) / 2
#         yield_df['Mid_EUR3M'] = (yield_df['EUR3MD=_BID'] + yield_df['EUR3MD=_ASK']) / 2
#         yield_df['Mid_EUR6M'] = (yield_df['EUR6MD=_BID'] + yield_df['EUR6MD=_ASK']) / 2
#         yield_df['Mid_TND']   = (yield_df['TNDOND=_BID'] + yield_df['TNDOND=_ASK']) / 2

#         # Helper function: select the first non-null value in a series
#         def first_non_null(series):
#             non_null = series.dropna()
#             return non_null.iloc[0] if not non_null.empty else None

#         # Group by date (Timestamp) and aggregate each field by taking the first non-null value
#         daily_yield = yield_df.groupby('Timestamp').agg({
#             'Mid_USD1M': first_non_null,
#             'Mid_USD3M': first_non_null,
#             'Mid_USD6M': first_non_null,
#             'Mid_EUR1M': first_non_null,
#             'Mid_EUR3M': first_non_null,
#             'Mid_EUR6M': first_non_null,
#             'Mid_TND': first_non_null
#         }).reset_index()

#         # Save daily aggregated yield data to JSON
#         daily_yield_json = daily_yield.to_json(orient="records", date_format="iso")
#         with open("data/daily_yield_rates.json", "w") as f:
#             f.write(daily_yield_json)

#         print("Yield rates saved to daily_yield_rates.json.")

# except Exception as e:
#     print(f"Error fetching data: {e}")

# # Close session
# rd.close_session()
