# option_pricer/main.py

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm

from .data_access import (
    load_eurusd_spot_from_excel,
    load_eurusd_vol_surface_from_excel,
    get_db_session,
    load_interbank_spot,
    load_atm_vol_surface,
    load_yield_curve,
)
from .utils   import tenor_to_tau, tenor_to_tau_days, fx_forward_discrete, ZeroCurve
from .ssvi    import SSVICalibrator
from .pricing import garman_kohlhagen
from models import db  # noqa: F401

# --- CONFIG ---
DB_URL           = "postgresql://postgres:pass123@localhost:5432/postgres"
EURUSD_SPOT_PATH = "data/eur usd spot_rates.xlsx"
EURUSD_VOL_PATH  = "data/vol surface.xlsx"

# --- DATA LOAD ---
session        = get_db_session(DB_URL)
eurusd_spot    = load_eurusd_spot_from_excel(EURUSD_SPOT_PATH)
eurusd_surf    = load_eurusd_vol_surface_from_excel(EURUSD_VOL_PATH)
usdtnd_spot    = load_interbank_spot(session, "USD")  # USD/TND
eurtnd_spot    = load_interbank_spot(session, "EUR")  # EUR/TND
atm_vol_raw    = load_atm_vol_surface(session)
yc_usd         = load_yield_curve(session, "USD")
yc_eur         = load_yield_curve(session, "EUR")
yc_tnd         = load_yield_curve(session, "TND")
print("Loaded TND curve:", yc_tnd)
print("Loaded EUR curve:", yc_eur)
print("Loaded USD curve:", yc_usd)

# ---------- base date & τ --------------
if "Date" in eurusd_surf.columns and eurusd_surf["Date"].notna().all():
    eurusd_surf["Date"] = pd.to_datetime(eurusd_surf["Date"])
    atm_vol_raw["Date"] = pd.to_datetime(atm_vol_raw["Date"])
    val_date            = eurusd_surf["Date"].iloc[0]
    eurusd_surf["tau"]  = eurusd_surf.apply(
        lambda r: tenor_to_tau(r["Tenor"], val_date=r["Date"]), axis=1)
    atm_vol_raw["tau"] = atm_vol_raw.apply(
        lambda r: tenor_to_tau(r["Tenor"], val_date=r["Date"]), axis=1)
else:
    val_date = datetime.today()
    eurusd_surf["tau"]  = eurusd_surf["Tenor"].apply(
        lambda x: tenor_to_tau(x, val_date=val_date))
    atm_vol_raw["tau"]  = atm_vol_raw["Tenor"].apply(
        lambda x: tenor_to_tau(x, val_date=val_date))

eurusd_surf.rename(columns={"Side": "side"}, inplace=True)
eurusd_surf["side"] = eurusd_surf["side"].str.upper()

# ---------- filter ATM vols to latest snapshot ----------
if "Date" in atm_vol_raw.columns:
    atm_vol_raw = atm_vol_raw[atm_vol_raw["Date"] == atm_vol_raw["Date"].max()]

if "Mid" not in atm_vol_raw.columns or atm_vol_raw["Mid"].isna().all():
    atm_vol_raw["Mid"] = (pd.to_numeric(atm_vol_raw["Bid"], errors="coerce") +
                          pd.to_numeric(atm_vol_raw["Ask"], errors="coerce"))/2
atm_vol = atm_vol_raw.dropna(subset=["Mid", "tau"])

atm_curve   = (atm_vol.groupby("tau", as_index=False)["Mid"].mean()
                       .sort_values("tau"))
theta_curve = pd.Series(((atm_curve["Mid"]/100)**2 * atm_curve["tau"]).values,
                        index=atm_curve["tau"].values)
eurusd_surf["atm_vol"]   = np.interp(eurusd_surf["tau"],
                                     atm_curve["tau"], atm_curve["Mid"])
eurusd_surf["theta_tau"] = (eurusd_surf["atm_vol"]/100)**2 * eurusd_surf["tau"]

# ---------- yield curves & forward ----------
zc_eur = ZeroCurve(yc_eur, val_date=val_date)
zc_usd = ZeroCurve(yc_usd, val_date=val_date)
zc_tnd = ZeroCurve(yc_tnd, val_date=val_date)

# *** FIXED: Correct domestic/foreign for EURUSD calibration ***
eurusd_surf["rd"] = eurusd_surf["tau"].apply(zc_usd.r)  # USD = domestic
eurusd_surf["rf"] = eurusd_surf["tau"].apply(zc_eur.r)  # EUR = foreign
spot_eurusd = eurusd_spot["Rate"].iloc[-1]
eurusd_surf["forward"] = eurusd_surf.apply(
    lambda r: fx_forward_discrete(spot_eurusd, r["rd"], r["rf"], int(r["tau"]*365)), axis=1)

# ---------- Black-76 helpers -------------
eurusd_surf["sigma"] = eurusd_surf["IV_pct"] / 100
eurusd_surf["z"]     = eurusd_surf["Delta"].apply(
    lambda d: norm.ppf(float(d)/100) if str(d).upper() != "ATM" else 0)
eurusd_surf["w"]     = eurusd_surf["sigma"]**2 * eurusd_surf["tau"]

def _lnFK(row):
    if str(row["Delta"]).upper()=="ATM":
        return 0.0
    sign = 1 if row["side"]=="CALL" else -1
    return sign*row["sigma"]*np.sqrt(row["tau"])*row["z"] - 0.5*row["sigma"]**2*row["tau"]

eurusd_surf["lnFK"]   = eurusd_surf.apply(_lnFK, axis=1)
eurusd_surf["strike"] = eurusd_surf["forward"]*np.exp(-eurusd_surf["lnFK"])
eurusd_surf["k"]      = np.log(eurusd_surf["strike"]/eurusd_surf["forward"])

# ---------- SSVI calibration --------------
ssvi = SSVICalibrator().fit(eurusd_surf, theta_curve)

# ---------- Use 1-year rolling window for all realised-vol --------
def rv30(df):
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    logret = np.log(df_sorted["Rate"]).diff()
    rv = (logret.rolling(30).std()*np.sqrt(252))
    rv = rv.dropna()
    rv.index = df_sorted["Date"].iloc[rv.index]
    return rv


# Ensure all date columns are pandas Timestamps
usdtnd_spot["Date"] = pd.to_datetime(usdtnd_spot["Date"])
eurusd_spot["Date"] = pd.to_datetime(eurusd_spot["Date"])
eurtnd_spot["Date"] = pd.to_datetime(eurtnd_spot["Date"])

# ---- Restrict history to 1 year only for realized-vol and ratios ----
window_days = 365  # or 730 for 2 years if you want
usd_dates = usdtnd_spot["Date"]
eur_dates = eurusd_spot["Date"]
latest_start = max(usd_dates.min(), eur_dates.min())
max_common_date = min(usd_dates.max(), eur_dates.max())
window_start = max(latest_start, max_common_date - pd.Timedelta(days=window_days))

print(f"Rolling vol window: {window_start} to {max_common_date}")

usdtnd_spot_cut = usdtnd_spot[(usdtnd_spot["Date"] >= window_start) & (usdtnd_spot["Date"] <= max_common_date)].reset_index(drop=True)
eurusd_spot_cut = eurusd_spot[(eurusd_spot["Date"] >= window_start) & (eurusd_spot["Date"] <= max_common_date)].reset_index(drop=True)
eurtnd_spot_cut = eurtnd_spot[(eurtnd_spot["Date"] >= window_start) & (eurtnd_spot["Date"] <= max_common_date)].reset_index(drop=True)

rv_eurusd = rv30(eurusd_spot_cut)
rv_usdtnd = rv30(usdtnd_spot_cut)
rv_eurtnd = rv30(eurtnd_spot_cut)

# ---------- Compute and debug alpha_usd for USD/TND ----------
def compute_alpha_usd(rv_usdtnd, rv_eurusd):
    overlap_usd = rv_usdtnd.index.intersection(rv_eurusd.index)
    print(f"rv_usdtnd n={len(rv_usdtnd)}, rv_eurusd n={len(rv_eurusd)}")
    print(f"overlap_usd n={len(overlap_usd)}")
    if len(overlap_usd) > 0:
        print(f"First overlap: {overlap_usd[0]}, Last overlap: {overlap_usd[-1]}")
        ratio_usd = (rv_usdtnd.loc[overlap_usd] / rv_eurusd.loc[overlap_usd]).dropna()
        print("ratio_usd:", ratio_usd.describe())
        if len(ratio_usd) > 0:
            alpha_usd = np.percentile(ratio_usd, 90)
        else:
            alpha_usd = np.nan
    else:
        alpha_usd = np.nan
    if np.isnan(alpha_usd) or alpha_usd < 0.1 or alpha_usd > 5.0:
        print("alpha_usd invalid or out of range, falling back to 1.0")
        alpha_usd = 1.0
    print(f"Final alpha_usd used: {alpha_usd}")
    return alpha_usd

alpha_usd = compute_alpha_usd(rv_usdtnd, rv_eurusd)

# ---------- Synthetic cross alpha for EUR/TND ----------
aligned = pd.DataFrame({
    "usd_tnd": rv_usdtnd,
    "eur_usd": rv_eurusd
}).dropna()

# 60-day rolling correlation between USD/TND and EUR/USD
roll_corr = (
    aligned.rolling(window=60)
    .corr().unstack().iloc[:,1]
)

syn_vol = np.sqrt(
    aligned["usd_tnd"] ** 2 +
    aligned["eur_usd"] ** 2 -
    2 * roll_corr * aligned["usd_tnd"] * aligned["eur_usd"]
)

alpha_syn = (syn_vol / aligned["eur_usd"]).dropna()
if len(alpha_syn) > 0:
    cut = alpha_syn.index[-1] - pd.Timedelta(days=365)
    alpha_eur = np.percentile(alpha_syn[alpha_syn.index >= cut], 90)
else:
    alpha_eur = 1.0  # fallback
print(f"Synthetic cross α for EUR/TND: {alpha_eur:.3f}")

# ---------- pricing function ---------------
def price_fx_option(currency: str, tenor: str, option_type="call", spot=None):
    """Price an ATM FX option via RSSVI."""
    currency = currency.upper()
    if currency not in {"USD/TND", "EUR/TND"}:
        raise ValueError("currency must be 'USD/TND' or 'EUR/TND'")

    # Use user-provided spot or database default
    if spot is None:
        S = usdtnd_spot["Rate"].iloc[-1] if currency=="USD/TND" else eurtnd_spot["Rate"].iloc[-1]
    else:
        S = spot

    tau, days = tenor_to_tau_days(tenor, val_date=val_date)
    rd = zc_tnd.r(tau)
    rf = zc_usd.r(tau) if currency=="USD/TND" else zc_eur.r(tau)
    alpha = alpha_usd if currency=="USD/TND" else alpha_eur

    print(f"rd (domestic): {rd}, rf (foreign): {rf}, tau: {tau}, days: {days}")
    F = fx_forward_discrete(S, rd, rf, days)
    total_var = alpha**2 * ssvi.total_variance(0.0, tau)
    sigma = np.sqrt(total_var / tau) if total_var > 0 else np.nan
    price = garman_kohlhagen(option_type, S, F, tau, rd, rf, sigma)

    return dict(currency=currency, tenor=tenor, option_type=option_type,
                spot=S, forward=F, tau=tau, implied_vol=sigma,
                price=price, alpha=alpha, val_date=val_date.date())

# ---------- Interactive user input ----------
if __name__ == "__main__":
    print("\nFX Option Pricer")
    print("Choose Currency Pair: USD/TND or EUR/TND")
    currency = input("Currency pair: ").strip().upper()
    while currency not in {"USD/TND", "EUR/TND"}:
        print("Invalid currency pair! Try 'USD/TND' or 'EUR/TND'.")
        currency = input("Currency pair: ").strip().upper()

    tenor = input("Tenor (e.g., 1M, 3M, 6M, 1Y): ").strip().upper()
    while not (len(tenor) >= 2 and tenor[:-1].isdigit() and tenor[-1] in "MWY"):
        print("Invalid tenor! Example valid: 1M, 3M, 6M, 1Y")
        tenor = input("Tenor (e.g., 1M, 3M, 6M, 1Y): ").strip().upper()

    option_type = input("Option type (call/put): ").strip().lower()
    while option_type not in {"call", "put"}:
        print("Invalid type! Enter 'call' or 'put'.")
        option_type = input("Option type (call/put): ").strip().lower()

    # Notional input (default 1000000 if blank)
    notional = input("Notional in base currency (e.g. 1000000): ").strip().replace(",", "")
    if not notional:
        notional = 1000000
    else:
        try:
            notional = float(notional)
        except Exception:
            print("Invalid notional, defaulting to 1,000,000")
            notional = 1000000

    # --- Print latest spot for each currency ---
    print(f"\nLatest USD/TND spot: {usdtnd_spot['Rate'].iloc[-1]:.5f} (Date: {usdtnd_spot['Date'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"Latest EUR/TND spot: {eurtnd_spot['Rate'].iloc[-1]:.5f} (Date: {eurtnd_spot['Date'].iloc[-1].strftime('%Y-%m-%d')})")

    # Decide which spot to override
    if currency == "USD/TND":
        db_spot = usdtnd_spot['Rate'].iloc[-1]
    elif currency == "EUR/TND":
        db_spot = eurtnd_spot['Rate'].iloc[-1]
    else:
        db_spot = None

    user_spot = input(f"Enter custom spot for {currency} (leave blank to use {db_spot:.5f}): ").strip()
    if user_spot:
        try:
            user_spot = float(user_spot)
        except Exception:
            print("Invalid input, using default spot.")
            user_spot = db_spot
    else:
        user_spot = db_spot

    quote = price_fx_option(currency, tenor, option_type, spot=user_spot)
    total_pv = quote['price'] * notional

    # Pretty output
    print("\n--- FX Option Valuation ---")
    print(f"Val Date   : {quote['val_date']}")
    print(f"Pair       : {quote['currency']}")
    print(f"Type/Tenor : {quote['option_type'].upper()} {quote['tenor']}")
    print(f"Spot       : {quote['spot']:.5f}")
    print(f"Forward    : {quote['forward']:.5f}")
    print(f"Implied σ  : {quote['implied_vol']:.4%}")
    print(f"Alpha      : {quote['alpha']:.3f}")
    print(f"PV per unit: {quote['price']:.5f} {quote['currency'].split('/')[-1]} per 1 {quote['currency'].split('/')[0]}")
    print(f"Notional   : {notional:,.0f} {quote['currency'].split('/')[0]}")
    print(f"Total PV   : {total_pv:,.2f} {quote['currency'].split('/')[-1]} (payable today)")

    # Optionally: show all details
    # print(quote)
