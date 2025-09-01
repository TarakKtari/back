# invoice_helper.py
from datetime import date, timedelta
from collections import defaultdict
from typing import List

from sqlalchemy import func
from num2words import num2words             # pip install num2words

from models import db, User, Order, Invoice, InvoiceLine


# ────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────
def next_invoice_number() -> str:
    """Return a unique number like '004/2025'."""
    year = date.today().year
    last = (Invoice.query
                   .filter(func.extract('year', Invoice.created_at) == year)
                   .order_by(Invoice.id.desc())
                   .first())
    seq = (int(last.number.split('/')[0]) + 1) if last else 1
    return f"{seq:03d}/{year}"


def amount_to_words(amount: float, lang: str = "fr") -> str:
    dinars = int(amount)
    millis = int(round((amount - dinars) * 1000))

    words  = num2words(dinars, lang=lang).replace("-", " ") + " dinars"
    if millis:
        words += " et " + num2words(millis, lang=lang).replace("-", " ") + " millimes"
    return words.capitalize()


def apply_netting(orders: List[Order]) -> None:
    """
    Zero-commission the smaller side (buy vs sell) per (day, currency).
    Works *in-place* on Order.commission_percent.
    """
    buckets = defaultdict(lambda: {"buy": [], "sell": []})

    for o in orders:
        side = "buy" if o.transaction_type.lower() in ("import", "buy") else "sell"
        buckets[(o.transaction_date, o.currency)][side].append(o)

    for grp in buckets.values():
        tot_buy  = sum(o.original_amount for o in grp["buy"])
        tot_sell = sum(o.original_amount for o in grp["sell"])
        if not tot_buy or not tot_sell:
            continue

        weaker = "buy" if tot_buy < tot_sell else "sell"
        remaining = min(tot_buy, tot_sell)

        for o in sorted(grp[weaker], key=lambda x: x.original_amount):
            if remaining <= 0:
                break
            if o.original_amount <= remaining:
                o.commission_percent = 0.0
                remaining -= o.original_amount
            else:
                ratio = remaining / o.original_amount
                o.commission_percent *= (1 - ratio)
                remaining = 0


# ────────────────────────────────────────────────────────────
# main builder
# ────────────────────────────────────────────────────────────
def build_invoice(client: User, period_start: date, period_end: date) -> Invoice:
    """
    Build + commit an Invoice for one client and one calendar month.
    """
    # 1) fetch orders
    orders = (Order.query
                    .filter(Order.user_id == client.id,
                            Order.status.in_(["Executed", "Matched"]),
                            Order.transaction_date.between(period_start, period_end))
                    .all())

    if client.netting_enabled:
        apply_netting(orders)

    variable_lines, total_comm_tnd = [], 0.0
    for o in orders:
        pct  = o.commission_percent or 0.0
        #tnd  = (o.execution_rate or 0) * o.original_amount * pct
        
        # Simple fee calculation: multiply by TND rate if it exists, otherwise by 1
        tnd_rate = o.tnd_rate if o.tnd_rate else 1.0
        tnd  = (o.execution_rate or 0) * o.original_amount * pct * tnd_rate
        total_comm_tnd += tnd

        variable_lines.append(InvoiceLine(
            kind       = "variable",
            trade_date = o.transaction_date,
            tx_type    = o.transaction_type,
            instrument = o.trade_type,
            currency   = o.currency,
            amount_fc  = o.original_amount,
            exec_rate  = o.execution_rate,
            comm_pct   = pct,
            comm_tnd   = tnd
        ))

    # 2) fixed fee row
    fixed_fee  = client.fixed_monthly_fee or 0.0
    fixed_line = InvoiceLine(kind="fixed",
                             description="Frais mensuel fixe",
                             comm_tnd=fixed_fee)

    # 3) totals
    total_ht  = total_comm_tnd + fixed_fee
    vat_rate  = 0.0 if client.tva_exempt else 0.19
    tva       = total_ht * vat_rate
    total_ttc = total_ht + tva + 1.0         # timbre fiscale

    # 4) create & persist invoice
    inv = Invoice(
        number       = next_invoice_number(),
        client       = client,
        period_start = period_start,
        period_end   = period_end,
        due_date     = period_end + timedelta(days=7),
        total_ht     = total_ht,
        vat_rate     = vat_rate,
        tva          = tva,
        total_ttc    = total_ttc,
    )
    inv.lines.extend(variable_lines + [fixed_line])

    # 5) add montant-en-lettres for React / PDF
    inv.total_in_words = amount_to_words(total_ttc)

    db.session.add(inv)
    db.session.commit()
    return inv
