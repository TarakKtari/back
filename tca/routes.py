# tca/routes.py
from . import tca_bp
from datetime import timedelta
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask import request, jsonify
from sqlalchemy import func
from models import db, User,  Order, AuditLog, ExchangeData, OpenPosition, PremiumRate, InterbankRate, BctxFixing, TcaSpotInput
from user.routes import calculate_forward_rate, get_interbank_rate_from_db, get_yield_period
from collections import defaultdict     


# helper ───────────────────────────────────────────────────────────
def _effective_client_id(current_user_id: int, query_id: str | None) -> int:
    me = User.query.get(current_user_id)
    if not me or not me.is_admin:                     
        return current_user_id

    if not query_id:                                 
        return current_user_id

    target = None
    if query_id.isdigit():
        target = User.query.get(int(query_id))

    # fall back to client_name search (case-insensitive)
    if not target:
        target = (User.query
                       .filter(func.lower(User.client_name) == query_id.lower())
                       .first())

    return target.id if target else current_user_id

def get_bctx_fixing_rate(transaction_date, currency):
    fixing = BctxFixing.query.filter_by(date=transaction_date, session='morning').first()
    if not fixing:
        fixing = BctxFixing.query.filter_by(date=transaction_date, session='afternoon').first()
    if not fixing:
        return None
    if currency == "EUR":
        if fixing.eur_bid is not None and fixing.eur_ask is not None:
            return (fixing.eur_bid + fixing.eur_ask) / 2
    elif currency == "GBP":
        if fixing.gbp_bid is not None and fixing.gbp_ask is not None:
            return (fixing.gbp_bid + fixing.gbp_ask) / 2
    elif currency == "JPY":
        if fixing.jpy_bid is not None and fixing.jpy_ask is not None:
            return (fixing.jpy_bid + fixing.jpy_ask) / 2
    elif currency == "USD":
        if fixing.tnd_bid is not None and fixing.tnd_ask is not None:
            return (fixing.tnd_bid + fixing.tnd_ask) / 2
    return None


def get_interbank_rate(d, c):
    return get_interbank_rate_from_db(d, c)

def compute_forward_rate_for_date(cur, hist, val):
    row = ExchangeData.query.filter_by(date=hist).first()
    if not row:
        return 0.0
    days = (val - hist).days
    p = get_yield_period(days)[0]          
    spot, r_for = (
        (row.spot_usd, getattr(row, f"usd_{p}m")) if cur.upper() == "USD"
        else (row.spot_eur, getattr(row, f"eur_{p}m"))
    )
    r_dom = getattr(row, f"tnd_{p}m")
    return calculate_forward_rate(spot, r_for, r_dom, days)

def get_option_pct(cur, opt_type, days):
    """
    Always returns the premium % for buying an option 
    (we never sell them), interpolated by the nearest tenor.
    """
    rows = PremiumRate.query.filter_by(
        currency=cur.upper(),
        option_type=opt_type,
        transaction_type="buy"      # hard-code to buy
    ).all()
    if not rows:
        return 0.0
    # pick the maturity_days closest to our target
    return min(rows, key=lambda r: abs(r.maturity_days - days)).premium_percentage

def get_fixing_rate_for_session(date, currency, session):
    """Returns (bid+ask)/2 for the given date, currency and session ('morning' or 'afternoon')."""
    fixing = BctxFixing.query.filter_by(date=date, session=session).first()
    if not fixing:
        return None

    if currency == 'USD':
        return (fixing.tnd_bid + fixing.tnd_ask) / 2
    elif currency == 'EUR':
        return (fixing.eur_bid + fixing.eur_ask) / 2
    elif currency == 'GBP':
        return (fixing.gbp_bid + fixing.gbp_ask) / 2
    elif currency == 'JPY':
        return (fixing.jpy_bid + fixing.jpy_ask) / 2
    return None
def get_fixing_components(date, currency, session):
    def _extract(fix, side):
        if not fix:
            return 0.0          
        col = {
            'USD': f"tnd_{side}",
            'EUR': f"eur_{side}",
            'GBP': f"gbp_{side}",
            'JPY': f"jpy_{side}",
        }[currency.upper()]
        return getattr(fix, col, 0.0) or 0.0

    fix = BctxFixing.query.filter_by(date=date, session=session).first()
    bid = _extract(fix, 'bid')
    ask = _extract(fix, 'ask')

    # fallback → morning
    if session == 'afternoon' and (bid == 0.0 or ask == 0.0):
        fix_am = BctxFixing.query.filter_by(date=date, session='morning').first()
        bid = bid or _extract(fix_am, 'bid')
        ask = ask or _extract(fix_am, 'ask')

    mid = (bid + ask) / 2 if bid and ask else 0.0
    return {"bid": bid, "ask": ask, "mid": mid}


@tca_bp.route('/spot', methods=['GET'])
@jwt_required()
def tca_spot():
    current_uid = get_jwt_identity()
    target_id   = request.args.get('client_id')
    client_id   = _effective_client_id(current_uid, target_id)
    currency = request.args.get('currency', type=str)
    q = TcaSpotInput.query.filter_by(client_id=client_id)
    if currency:
        q = q.filter(func.upper(TcaSpotInput.currency) == currency.upper())
    inputs = q.all()
    result    = []

    for inp in inputs:
        ccy = inp.currency
        comp_m = get_fixing_components(inp.transaction_date, ccy, 'morning')
        comp_a = get_fixing_components(inp.transaction_date, ccy, 'afternoon')
        irate  = get_interbank_rate(inp.transaction_date, ccy) or 0.0
        ex     = inp.execution_rate
        amt    = inp.amount

        # --- Sélection bid / ask pour le calcul P&L ---
        if inp.transaction_type == 'import':      
            fix_m = comp_m['ask']
            fix_a = comp_a['ask']
        else:                                    
            fix_m = comp_m['bid']
            fix_a = comp_a['bid']

        # --- P&L & spreads ----------------------------
        if inp.transaction_type == 'import':
            pnl_m = (ex - fix_m) * amt
            pnl_a = (ex - fix_a) * amt
            pnl_i = (ex - irate) * amt
            pct_m = (ex - fix_m) / fix_m * 100 if fix_m else 0
            pct_a = (ex - fix_a) / fix_a * 100 if fix_a else 0
            pct_i = (ex - irate) / irate * 100 if irate else 0

        else:
            pnl_m = (fix_m - ex) * amt
            pnl_a = (fix_a - ex) * amt
            pnl_i = (irate - ex) * amt
            pct_m = (fix_m - ex) / ex * 100 if ex else 0
            pct_a = (fix_a - ex) / ex * 100 if ex else 0
            pct_i = (irate - ex) / ex * 100 if ex else 0

        result.append({
            # --- Infos deal -------------------------------------------------
            "transaction_date" : inp.transaction_date.isoformat(),
            "value_date"       : inp.value_date.isoformat(),
            "currency"         : ccy,
            "transaction_type" : inp.transaction_type,
            "amount"           : amt,
            "execution_rate"   : ex,

            # --- Fixings complets ------------------------------------------
            "fix_bid_morning"      : comp_m['bid'],
            "fix_ask_morning"      : comp_m['ask'],
            "fix_mid_morning"      : comp_m['mid'],
            "fix_bid_afternoon"    : comp_a['bid'],
            "fix_ask_afternoon"    : comp_a['ask'],
            "fix_mid_afternoon"    : comp_a['mid'],

            # --- Interbank --------------------------------------------------
            "interbank_rate"       : irate,

            # --- P&L & spreads (basés sur bid/ask) --------------------------
            "pnl_fixing_morning_tnd"      : pnl_m,
            "pnl_fixing_afternoon_tnd"    : pnl_a,
            "pnl_interbank_tnd"           : pnl_i,
            "spread_fixing_morning_pct"   : pct_m,
            "spread_fixing_afternoon_pct" : pct_a,
            "spread_interbank_pct"        : pct_i,
        })

    return jsonify(result), 200


# ------------------------------------------------------------------
# how far we are allowed to roll back (calendar days) for each tenor
# ------------------------------------------------------------------
LOOKBACK = {
    30 : 4,     # 1‑month
    90 : 4,     # 3‑month
    180: 10,    # 6‑month 
    270: 10,    # 9‑month
    360: 10     # 12‑month
}


def get_non_zero_value(model, column, as_of, max_shift=4, **filters):
    """
    Walk backwards day‑by‑day from `as_of` (inclusive) up to `max_shift` days
    and return (value, effective_date) where value != 0.
    Extra keyword arguments are passed to .filter_by() so you can constrain
    on things like currency.
    """
    d = as_of
    for _ in range(max_shift + 1):
        row = (db.session.query(model)
                          .filter_by(date=d, **filters)
                          .first())
        if row:
            v = getattr(row, column)
            if v:                      
                return v, d
        d -= timedelta(days=1)
    return None, None


@tca_bp.route('/spot-forward', methods=['GET'])
@jwt_required()
def tca_spot_forward():
    current_uid = get_jwt_identity()

    #accept either ?client_id=42 or ?client_id=AcmeCorp
    target_id   = request.args.get('client_id')
    client_id   = _effective_client_id(current_uid, target_id)

    ccy_filter = request.args.get('currency', type=str)

    q = TcaSpotInput.query.filter_by(client_id=client_id)
    if ccy_filter:
        q = q.filter(func.upper(TcaSpotInput.currency) == ccy_filter.upper())

    inputs      = q.all()
    MATURITIES  = [30, 90, 180, 270, 360]      

    deals = []
    for inp in inputs:
        base = {
            "transaction_date": inp.transaction_date.isoformat(),
            "value_date"      : inp.value_date.isoformat(),
            "currency"        : inp.currency,
            "transaction_type": inp.transaction_type,
            "amount"          : inp.amount,
            "execution_rate"  : inp.execution_rate,
        }

        ccy    = inp.currency.upper()
        hedges = []

        for Δj in MATURITIES:
            target_day = inp.transaction_date - timedelta(days=Δj)
            max_shift  = LOOKBACK[Δj]                    
   
            spot, d_spot = get_non_zero_value(
                InterbankRate, "rate", target_day, max_shift, currency=ccy
            )

            tenor      = get_yield_period(Δj)             
            y_for_col  = f"{ccy.lower()}_{tenor}"
            y_dom_col  = f"tnd_{tenor}"

            y_for, d_for = get_non_zero_value(
                ExchangeData, y_for_col, target_day, max_shift
            )
            y_dom, d_dom = get_non_zero_value(
                ExchangeData, y_dom_col, target_day, max_shift
            )

            if spot is not None and y_for is not None and y_dom is not None:
                y_dom += 0.0025
                fwd = calculate_forward_rate(spot, y_for, y_dom, Δj)

                if inp.transaction_type.lower() in ("import", "buy"):
                    pnl_tnd = (inp.execution_rate - fwd) * inp.amount
                    pnl_pct = inp.execution_rate / fwd - 1
                else:
                    pnl_tnd = (fwd - inp.execution_rate) * inp.amount
                    pnl_pct = fwd / inp.execution_rate - 1

                assoc_date = max(d_spot, d_for, d_dom)
            else:
                fwd = pnl_tnd = pnl_pct = assoc_date = None

            hedges.append({
                "maturity_days"  : Δj,
                "associated_date": assoc_date.isoformat() if assoc_date else None,
                "associated_spot": spot,
                "forward_rate"   : fwd,
                "pnl_tnd"        : pnl_tnd,
                "pnl_pct"        : pnl_pct,
            })

        # -------- summary -------------------------------------------------
        pnl_vals = [h["pnl_tnd"] for h in hedges if h["pnl_tnd"] is not None]
        summary  = {
            "pnl_total": sum(pnl_vals),
            "pnl_moyen": (sum(pnl_vals) / len(pnl_vals)) if pnl_vals else 0,
            "max_gain" : max(pnl_vals) if pnl_vals else 0,
            "max_perte": min(pnl_vals) if pnl_vals else 0,
        }

        base["hedging_with_forward"] = {"details": hedges, **summary}
        deals.append(base)

    return jsonify(deals), 200

# ---------------------------------------------------------------------------
#  /tca/spot-option   (spot‑plus‑options Transaction‑Cost‑Analysis)
# ---------------------------------------------------------------------------
@tca_bp.route('/spot-option', methods=['GET'])
@jwt_required()
def tca_spot_option():
    current_uid = get_jwt_identity()
    target_id   = request.args.get('client_id')
    client_id   = _effective_client_id(current_uid, target_id)
    ccy_filter = request.args.get('currency', type=str)

    q = TcaSpotInput.query.filter_by(client_id=client_id)
    if ccy_filter:
        q = q.filter(func.upper(TcaSpotInput.currency) == ccy_filter.upper())
    inputs = q.all()

    MATURITIES       = [30, 90, 180, 270, 360]
    LOOKBACK_DEFAULT = 4                    # fallback days if data missing
    deals            = []

    maturity_totals = defaultdict(float)  

    for inp in inputs:
        base = {
            
            "transaction_date": inp.transaction_date.isoformat(),
            "value_date"      : inp.value_date.isoformat(),
            "currency"        : inp.currency,
            "transaction_type": inp.transaction_type,
            "amount"          : inp.amount,
            "execution_rate"  : inp.execution_rate,
        }

        ccy    = inp.currency.upper()
        hedges = []

        for Δj in MATURITIES:
            target_day = inp.transaction_date - timedelta(days=Δj)

            # ---- pull market data (spot & yields) -------------------------
            spot, d_spot = get_non_zero_value(
                InterbankRate, "rate", target_day,
                max_shift=LOOKBACK.get(Δj, LOOKBACK_DEFAULT),
                currency=ccy
            )
            tenor = get_yield_period(Δj)               # '1m' | '3m' | '6m'
            y_for, _ = get_non_zero_value(
                ExchangeData, f"{ccy.lower()}_{tenor}", target_day,
                max_shift=LOOKBACK.get(Δj, LOOKBACK_DEFAULT)
            )
            y_dom, _ = get_non_zero_value(
                ExchangeData, f"tnd_{tenor}", target_day,
                max_shift=LOOKBACK.get(Δj, LOOKBACK_DEFAULT)
            )

            # ---- forward rate --------------------------------------------
            fwd = (
                calculate_forward_rate(spot, y_for, y_dom, Δj)
                if None not in (spot, y_for, y_dom)
                else None
                            )

                    # determine CALL vs PUT from the *deal* side
            deal_side  = inp.transaction_type.lower()        # "import"/"export"
            opt_type   = "CALL" if deal_side == "import" else "PUT"

                # we always buy the option
            option_side = "buy"

                # look up the premium using the option_side, not the deal side
            pct = get_option_pct(
                    ccy,           
                    opt_type,       
                    Δj              
                )

            prime_tnd = pct * inp.amount * (fwd or 0) if pct and fwd else None
            # ---- option P&L ----------------------------------------------
            if fwd is not None:
                if deal_side in ("import", "buy"):          # Achat
                    pnl_tnd = (max(inp.execution_rate - fwd, 0) * inp.amount) - (prime_tnd or 0)
                else:                                  # Vente
                    pnl_tnd = (max(fwd - inp.execution_rate, 0) * inp.amount) - (prime_tnd or 0)
            else:
                pnl_tnd = None

            ratio = (prime_tnd / pnl_tnd * 100) if (prime_tnd and pnl_tnd) else None

            maturity_totals[Δj] += pnl_tnd or 0.0

            hedges.append({
                "maturity_days"     : Δj,
                "associated_date"   : d_spot.isoformat() if d_spot else None,
                "associated_spot"   : spot,
                "forward_rate"      : fwd,
                "option_prime_pct"  : pct,
                "option_prime_tnd"  : prime_tnd,
                "pnl_tnd"           : pnl_tnd,
                "premium_over_pnl%" : ratio,
            })

        pnls = [h["pnl_tnd"] for h in hedges if h["pnl_tnd"] is not None]
        base["hedging_with_options"] = {
            "details"  : hedges,
            "pnl_total": sum(pnls),
            "pnl_moyen": (sum(pnls) / len(pnls)) if pnls else 0,
            "max_gain" : max(pnls) if pnls else 0,
            "max_perte": min(pnls) if pnls else 0,
        }

        deals.append(base)

    summary_by_maturity = [
        {"maturity_days": d, "pnl_total": v}
        for d, v in sorted(maturity_totals.items())    
    ]

    return jsonify({
        "deals": deals,
        "options_totals_by_maturity": summary_by_maturity
    }), 200

