import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import InterbankRate, ATMVol, ExchangeData
from .utils import tenor_to_tau

# --- Excel Data Loaders ---
def load_eurusd_spot_from_excel(path):
    df = pd.read_excel(path, header=0)
    df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "Spot"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Standardize to ['Date', 'Rate'] for compatibility
    df = df.rename(columns={"Spot": "Rate"})
    return df

def load_eurusd_vol_surface_from_excel(path):
    df = pd.read_excel(path)
    # Canonical schema: ["Date","Tenor","Delta","Side","IV_pct"]
    col_map = {}
    for col in df.columns:
        c = col.strip().upper()
        if c in ["IV (%)", "VOLATILITY", "IV_PCT"]:
            col_map[col] = "IV_pct"
        elif c == "TYPE":
            col_map[col] = "Side"
        elif c == "DELTA":
            col_map[col] = "Delta"
        elif c == "TENOR":
            col_map[col] = "Tenor"
        elif c == "DATE":
            col_map[col] = "Date"
    df = df.rename(columns=col_map)
    # Canonicalize IV column: only IV_pct, float
    if "IV_pct" in df.columns:
        df["IV_pct"] = pd.to_numeric(df["IV_pct"], errors="coerce")
    # Robust Side tagging: infer Side from Delta sign if missing or invalid
    def infer_side(row):
        s = str(row.get("Side", "")).upper()
        d = row.get("Delta", None)
        if s in {"CALL", "PUT", "ATM"}:
            return s
        if pd.isna(d):
            return "PUT"  # fallback
        try:
            d = float(d)
            if abs(d) < 1e-6:
                return "ATM"
            return "CALL" if d > 0 else "PUT"
        except Exception:
            return "PUT"
    df["Side"] = df.apply(infer_side, axis=1)
    # Ensure all canonical columns exist
    for col in ["Date","Tenor","Delta","Side","IV_pct"]:
        if col not in df.columns:
            df[col] = pd.NA
    # Canonicalize Delta: always positive, sign only in Side, skip 'ATM'
    if "Delta" in df.columns:
        def canon_delta(d):
            if pd.isna(d):
                return d
            if str(d).upper() == "ATM":
                return d
            try:
                return abs(float(d))
            except Exception:
                return d
        df["Delta"] = df["Delta"].apply(canon_delta)
    return df[["Date","Tenor","Delta","Side","IV_pct"] + [c for c in df.columns if c not in ["Date","Tenor","Delta","Side","IV_pct"]]]

# --- Postgres Data Loaders ---
def get_db_session(db_url):
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()

def load_interbank_spot(session, currency):
    # currency: 'USD' or 'EUR'
    rates = session.query(InterbankRate).filter_by(currency=currency).order_by(InterbankRate.date.asc()).all()
    df = pd.DataFrame([{'Date': r.date, 'Rate': r.rate} for r in rates])
    # Standardize columns just in case
    df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "Rate"})
    return df

def load_atm_vol_surface(session):
    vols = session.query(ATMVol).order_by(ATMVol.date.asc()).all()
    df = pd.DataFrame([{
        'Date': v.date,
        'Tenor': v.tenor,
        'Bid': v.bid,
        'Ask': v.ask,
        'Mid': v.mid,
        'Tau_days': v.tau_days,
        'Tau_years': v.tau_years
    } for v in vols])
    return df


def load_yield_curve(session, currency):
    # Returns the curve for the previous valid row (skipping the last)
    rows = list(session.query(ExchangeData).order_by(ExchangeData.date.desc()).limit(10))
    # Skip the very latest row
    for row in rows[1:]:  # start from index 1, which is the second latest
        if currency == 'USD':
            curve = {
                'ON': row.usd_ond,
                '1M': row.usd_1m,
                '3M': row.usd_3m,
                '6M': row.usd_6m,
                '1Y': row.usd_1y,
            }
        elif currency == 'EUR':
            curve = {
                'ON': row.eur_ond,
                '1M': row.eur_1m,
                '3M': row.eur_3m,
                '6M': row.eur_6m,
                '1Y': row.eur_1y,
            }
        elif currency == 'TND':
            curve = {
                'ON': row.tnd_ond,
                '1M': row.tnd_1m,
                '3M': row.tnd_3m,
                '6M': row.tnd_6m,
                '1Y': row.tnd_1y,
            }
        else:
            curve = {}
        if any(v not in (None, 0, 0.0) for v in curve.values()):
            print(f"Loaded {currency} curve (date={row.date}):", curve)
            return curve
    return {}