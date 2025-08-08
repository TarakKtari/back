import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import psycopg2
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# === PRICING ENGINE ===
def strike_from_spot(spot: float, rd: float, rf: float, T: float) -> float:
    return spot * (1 + rd * T) / (1 + rf * T)

def garman_kohlhagen_price(option_type: str, S: float, K: float, T: float, rd: float, rf: float, sigma: float) -> float:
    if T == 0 or sigma == 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S/K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df_d = np.exp(-rd * T)
    df_f = np.exp(-rf * T)
    if option_type == 'call':
        return S * df_f * norm.cdf(d1) - K * df_d * norm.cdf(d2)
    else:
        return -S * df_f * norm.cdf(-d1) + K * df_d * norm.cdf(-d2)

def garman_kohlhagen_greeks(option_type: str, S: float, K: float, T: float, rd: float, rf: float, sigma: float) -> Tuple[float, float, float]:
    d1 = (np.log(S/K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df_d = np.exp(-rd * T)
    df_f = np.exp(-rf * T)
    gamma = df_f * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    if option_type == 'call':
        delta = df_f * norm.cdf(d1)
        theta = (
            - (S * df_f * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - rd * K * df_d * norm.cdf(d2)
            + rf * S * df_f * norm.cdf(d1)
        )
    else:
        delta = -df_f * norm.cdf(-d1)
        theta = (
            - (S * df_f * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + rd * K * df_d * norm.cdf(-d2)
            - rf * S * df_f * norm.cdf(-d1)
        )
    return delta, gamma, theta

def implied_volatility(option_type: str, S: float, K: float, T: float, rd: float, rf: float, annualized_premium: float, tol: float = 1e-8, max_iter: int = 400) -> float:
    def objective(sigma):
        unit_price = garman_kohlhagen_price(option_type, S, K, T, rd, rf, sigma)
        model_pct_premium = unit_price / K
        return model_pct_premium - (annualized_premium / T)

    try:
        sigma_low, sigma_high = 1e-8, 10.0
        f_low = objective(sigma_low)
        f_high = objective(sigma_high)

        if f_low * f_high > 0:
            print(f"[No bracket] T={T:.6f}, K={K:.6f}, AnnualizedPremium={annualized_premium:.8f}, "
                  f"Model%Low={f_low + (annualized_premium / T):.8f}, Model%High={f_high + (annualized_premium / T):.8f}")
            return np.nan

        return brentq(objective, sigma_low, sigma_high, xtol=tol, maxiter=max_iter)

    except Exception as e:
        print(f"[Error] T={T:.6f}, K={K:.6f}, AnnualizedPremium={annualized_premium:.8f}, Error: {e}")
        return np.nan

# === DATABASE AND INTERPOLATION ===
def get_db_connection() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        dbname=os.getenv('POSTGRES_DB', 'colombus'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432')
    )

def get_latest_exchange_data(conn, currency_pair: str) -> Dict[str, Any]:
    cur = conn.cursor()

    if currency_pair == "EUR/TND":
        yield_fields = [
            "TND Overnight", "1M TND", "3M TND", "6M TND", "1Y TND",
            "EUR Overnight", "1M EUR", "3M EUR", "6M EUR", "1Y EUR"
        ]
    elif currency_pair == "USD/TND":
        yield_fields = [
            "TND Overnight", "1M TND", "3M TND", "6M TND", "1Y TND",
            "USD Overnight", "1M USD", "3M USD", "6M USD", "1Y USD"
        ]
    else:
        raise ValueError("Unsupported currency pair")

    where_nonzero = " OR ".join([f'COALESCE("{field}",0) > 0' for field in yield_fields])
    sql = f'''SELECT * FROM exchange_data WHERE ({where_nonzero}) ORDER BY "Date" DESC LIMIT 1;'''
    cur.execute(sql)
    row = cur.fetchone()
    if row is None:
        raise ValueError("No nonzero-yield row found in exchange_data table.")

    colnames = [desc[0] for desc in cur.description]
    data = dict(zip(colnames, row))

    if currency_pair == "EUR/TND":
        dom_yields = [data['TND Overnight'], data['1M TND'], data['3M TND'], data['6M TND'], data['1Y TND']]
        for_yields = [data['EUR Overnight'], data['1M EUR'], data['3M EUR'], data['6M EUR'], data['1Y EUR']]
    elif currency_pair == "USD/TND":
        dom_yields = [data['TND Overnight'], data['1M TND'], data['3M TND'], data['6M TND'], data['1Y TND']]
        for_yields = [data['USD Overnight'], data['1M USD'], data['3M USD'], data['6M USD'], data['1Y USD']]

    known_tenors = [0, 1/12, 3/12, 6/12, 1]
    return {'domestic_yields': dom_yields, 'foreign_yields': for_yields, 'known_tenors': known_tenors}

# === MAIN BACKEND FUNCTION ===
def backend_price_fx_options(
    currency_pair: str,
    option_type: str,
    spot: Optional[float],
    date_premia: List[Tuple[str, float, Optional[float]]],
    valuation_date: str = None,
    conn: psycopg2.extensions.connection = None
) -> pd.DataFrame:
    """
    FX option pricing engine. Accepts optional strike per maturity.
    If strike is not provided, fallback to spot + interest rate parity.
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True
    if valuation_date is None:
        val_date = datetime.today()
    else:
        val_date = datetime.strptime(valuation_date, "%d/%m/%Y")

    try:
        data = get_latest_exchange_data(conn, currency_pair)
        known_tenors = data['known_tenors']
        results = []

        for date_str, premium_input, strike_input in date_premia:
            maturity_date = datetime.strptime(date_str.strip(), "%d/%m/%Y")
            T = (maturity_date - val_date).days / 365.0
            if T <= 0:
                continue

            rd = float(np.interp(T, known_tenors, data['domestic_yields']))
            rf = float(np.interp(T, known_tenors, data['foreign_yields']))

            # Use strike input if provided, otherwise derive from spot
            if strike_input is not None:
                K = strike_input
                if spot is None:
                    raise ValueError(f"Spot must be provided for volatility/gamma/theta calculations when strike is not used.")
            else:
                if spot is None:
                    raise ValueError(f"Spot must be provided if strike is not entered.")
                K = strike_from_spot(spot, rd, rf, T)

            annualized_premium = premium_input
            imp_vol = implied_volatility(option_type, spot, K, T, rd, rf, annualized_premium)
            theo_price = garman_kohlhagen_price(option_type, spot, K, T, rd, rf, imp_vol)
            delta, gamma, theta = garman_kohlhagen_greeks(option_type, spot, K, T, rd, rf, imp_vol)

            results.append({
                'Currency_Pair': currency_pair,
                'Option_Type': option_type,
                'Maturity_Date': date_str,
                'T_Years': T,
                'Domestic_Rate_rd': rd,
                'Foreign_Rate_rf': rf,
                'Strike': round(K, 6),
                'Bank_Premium_%': annualized_premium,
                'Target_Unit_Price': (annualized_premium / T) * K,
                'Implied_Vol': imp_vol,
                'Theo_Price': theo_price,
                'Delta': delta,
                'Gamma': gamma,
                'Theta': theta,
            })

        return pd.DataFrame.from_records(results)

    finally:
        if close_conn:
            conn.close()


# === INTERPOLATED PRICING FUNCTION ===
def interpolate_and_price(
    known_maturities: List[float],
    known_vols: List[float],
    target_maturity: float,
    currency_pair: str,
    option_type: str,
    spot: Optional[float] = None,
    strike: Optional[float] = None,
    conn: psycopg2.extensions.connection = None
) -> Dict[str, Any]:
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        data = get_latest_exchange_data(conn, currency_pair)
        known_tenors = data['known_tenors']
        rd = float(np.interp(target_maturity, known_tenors, data['domestic_yields']))
        rf = float(np.interp(target_maturity, known_tenors, data['foreign_yields']))
        sigma = float(np.interp(target_maturity, known_maturities, known_vols))

        if strike is not None:
            K = strike
            if spot is None:
                raise ValueError("Spot is required for delta/gamma/theta even if strike is given.")
        else:
            if spot is None:
                raise ValueError("Either strike or spot must be provided.")
            K = strike_from_spot(spot, rd, rf, target_maturity)

        theo_price = garman_kohlhagen_price(option_type, spot, K, target_maturity, rd, rf, sigma)
        delta, gamma, theta = garman_kohlhagen_greeks(option_type, spot, K, target_maturity, rd, rf, sigma)

        return {
            'Interpolated_Maturity': target_maturity,
            'Interpolated_Vol': sigma,
            'Strike': K,
            'Domestic_Rate_rd': rd,
            'Foreign_Rate_rf': rf,
            'Theo_Price': theo_price,
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta
        }

    finally:
        if close_conn:
            conn.close()


# === REVERSE PREMIUM CALCULATION ===
def reverse_premium_from_vol(
    option_type: str,
    S: float,
    K: float,
    T: float,
    rd: float,
    rf: float,
    sigma: float
) -> Dict[str, float]:
    unit_price = garman_kohlhagen_price(option_type, S, K, T, rd, rf, sigma)
    annualized_pct_premium = unit_price / K
    non_annualized_pct_premium = annualized_pct_premium * T
    return {
        'Unit_Price': unit_price,
        'Annualized_%_Premium': annualized_pct_premium,
        'Non_Annualized_%_Premium': non_annualized_pct_premium
    }

# === CLI ENTRY POINT ===
if __name__ == "__main__":
    print("=== FX Option Implied Volatility Calculator ===")
    currency_pair = input("Enter currency pair (EUR/TND or USD/TND): ").strip().upper()
    option_type = input("Option type (call/put): ").strip().lower()
    spot = float(input("Enter spot rate: ").strip().replace(',', '.'))
    print("Enter maturities and premiums as comma-separated pairs (e.g. 29/07/2025:0.05,15/08/2025:0.07):")
    pairs = input().split(",")
    date_premia = []
    for pair in pairs:
        if ":" in pair:
            components = pair.strip().split(":")
            d = components[0].strip()
            p = components[1].strip()
            premium_val = float(p[:-1].replace(',', '.')) / 100 if p.endswith('%') else float(p.replace(',', '.'))
            # Handle optional strike (3rd value)
            strike_val = float(components[2].replace(',', '.')) if len(components) == 3 else None
            date_premia.append((d, premium_val, strike_val))

    results = backend_price_fx_options(currency_pair, option_type, spot, date_premia)

    pd.set_option('display.float_format', lambda x: f'{x:,.8f}')
    print("\nResults:")

    print("Percentage Premium Inputs:")
    print(results[['Maturity_Date', 'T_Years', 'Domestic_Rate_rd', 'Foreign_Rate_rf', 'Bank_Premium_%', 'Target_Unit_Price']])

    print("\nFull Pricing Output:")
    print(results[['Strike','Bank_Premium_%','Target_Unit_Price','Implied_Vol','Theo_Price','Delta','Gamma','Theta']])
    print(results[['Maturity_Date', 'T_Years', 'Strike', 'Bank_Premium_%', 'Implied_Vol', 'Theo_Price']])

    # === Interpolation Test ===
    print("\n--- Interpolation Test ---")
    use_interp = input("Do you want to interpolate and price a new maturity? (yes/no): ").strip().lower()
    if use_interp == 'yes':
        known_maturities = results['T_Years'].tolist()
        known_vols = results['Implied_Vol'].tolist()
        while True:
            new_date_str = input("Enter new target maturity date (e.g. 15/12/2025): ").strip()
            try:
                new_maturity_date = datetime.strptime(new_date_str, "%d/%m/%Y")
                break
            except ValueError:
                print("⚠️ Invalid date format. Please use DD/MM/YYYY.")
        val_date = datetime.today()
        new_maturity_date = datetime.strptime(new_date_str, "%d/%m/%Y")
        T_target = (new_maturity_date - val_date).days / 365.0

        strike_input = input("Optional: Enter strike to use (press Enter to skip): ").strip()
        strike_val = float(strike_input.replace(',', '.')) if strike_input else None

        interp_result = interpolate_and_price(
            known_maturities, known_vols, T_target,
            currency_pair, option_type,
            spot=spot if strike_val is None else spot,  # still needed for Greeks
            strike=strike_val
        )


        print("\nInterpolated Result:")
        for k, v in interp_result.items():
            print(f"{k}: {v:.8f}" if isinstance(v, float) else f"{k}: {v}")

        print("\n--- Reverse Engineered Premium from Implied Vol ---")
        rev = reverse_premium_from_vol(
            option_type=option_type,
            S=spot,
            K=interp_result['Strike'],
            T=interp_result['Interpolated_Maturity'],
            rd=interp_result['Domestic_Rate_rd'],
            rf=interp_result['Foreign_Rate_rf'],
            sigma=interp_result['Interpolated_Vol']
        )
        for k, v in rev.items():
            print(f"{k}: {v:.8f}")