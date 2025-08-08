
import numpy as np
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

def tenor_to_tau(label: str, daycount: str = "ACT/365", val_date: datetime = None) -> float:
    """
    Map tenor label to year fraction using actual calendar days (actual/365).
    If val_date is provided, computes expiry date and uses actual days.
    """
    label = label.strip().upper()
    if val_date is None:
        val_date = datetime.today()
    # Handle common variants
    if label in ("ON", "O/N", "1D", "OVERNIGHT"):
        expiry = val_date + relativedelta(days=1)
    else:
        match = re.match(r"(\d+)\s*WEEK", label)
        if match:
            expiry = val_date + relativedelta(weeks=int(match.group(1)))
        else:
            match = re.match(r"(\d+)\s*MONTH", label)
            if match:
                expiry = val_date + relativedelta(months=int(match.group(1)))
            else:
                match = re.match(r"(\d+)\s*YEAR", label)
                if match:
                    expiry = val_date + relativedelta(years=int(match.group(1)))
                else:
                    match = re.match(r"(\d+)\s*DAY", label)
                    if match:
                        expiry = val_date + relativedelta(days=int(match.group(1)))
                    else:
                        # Fallback for standard format (e.g., '1W', '2M', '3Y')
                        try:
                            num, unit = int(label[:-1]), label[-1]
                            if unit == "D": expiry = val_date + relativedelta(days=num)
                            elif unit == "W": expiry = val_date + relativedelta(weeks=num)
                            elif unit == "M": expiry = val_date + relativedelta(months=num)
                            elif unit == "Y": expiry = val_date + relativedelta(years=num)
                            else: raise ValueError()
                        except Exception:
                            raise ValueError(f"Unrecognised tenor: {label}")
    days = (expiry - val_date).days
    return days / 365.0


def tenor_to_tau_days(label: str, val_date: datetime = None) -> tuple:
    """
    Returns (tau, days) using actual calendar days.
    """
    if val_date is None:
        val_date = datetime.today()
    tau = tenor_to_tau(label, val_date=val_date)
    expiry = val_date + relativedelta(days=int(round(tau*365)))
    days = (expiry - val_date).days
    return tau, days


def fx_forward_discrete(spot, rd, rf, days):
    factor = (days - 2) / 360.0
    numerator = 1 + rd * factor
    denominator = 1 + rf * factor
    return spot * (numerator / denominator)

# --- Yield Curve Interpolation ---
import pandas as pd
class ZeroCurve:
    def __init__(self, curve_dict, val_date=None):
        self.val_date = val_date or datetime.today()
        # Convert curve_dict (pillar: rate) to DataFrame with tau (actual/365)
        taus = []
        rates = []
        for k, v in curve_dict.items():
            try:
                tau = tenor_to_tau(k, val_date=self.val_date)
                taus.append(tau)
                rates.append(v)
            except Exception:
                pass
        self.df = pd.DataFrame({'tau': taus, 'rate': rates}).sort_values('tau')
        self.taus = self.df['tau'].values
        self.rates = self.df['rate'].values
    def r(self, tau):
        # Log-linear interpolation in zero-coupon bond space
        if len(self.taus) == 0:
            return 0.0
        return float(np.interp(tau, self.taus, self.rates))
