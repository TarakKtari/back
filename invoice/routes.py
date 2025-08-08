# # invoice/routes.py  (drop this next to user.py & admin.py)
from datetime import date, timedelta
from sqlalchemy import case
from flask import Blueprint, request, jsonify, current_app
from . import invoice_bp              
from sqlalchemy import func, case
import flask_jwt_extended    
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)

from models import db, User, Order, Invoice, InvoiceStatus
from user.routes import calculate_benchmark, compute_bank_gains   # reuse your helper!
from user.routes import bank_gains           # we’ll call it internally

def get_invoice_transactions(client_id, year, month, currencies=("USD", "EUR")):
    from datetime import date
    from models import Order
    start = date(year, month, 1)
    # End: first day of next month
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    # Query all transactions for this client, in this period, for selected currencies
    return (
        Order.query
        .filter(
            Order.user_id == client_id,
            Order.status.in_(["Executed", "Matched"]),
            Order.currency.in_(currencies),
            Order.transaction_date >= start,
            Order.transaction_date < end,
        )
        .order_by(Order.transaction_date)
        .all()
    )
def build_invoice_rows(orders, netting_enabled=False):
    from collections import defaultdict
    import math

    # NETTING: group orders by date, zero out commission for lesser side
    if netting_enabled:
        day_orders = defaultdict(list)
        for o in orders:
            day_orders[o.transaction_date].append(o)
        for day, orders_on_day in day_orders.items():
            imports = [o for o in orders_on_day if o.transaction_type.lower() in ("import", "buy")]
            exports = [o for o in orders_on_day if o.transaction_type.lower() in ("export", "sell")]
            sum_import = sum(o.original_amount for o in imports)
            sum_export = sum(o.original_amount for o in exports)
            if sum_import > sum_export:
                for o in exports:
                    o.commission_percent = 0.0
            elif sum_export > sum_import:
                for o in imports:
                    o.commission_percent = 0.0

    rows = []
    for o in orders:
        bench = calculate_benchmark(o)
        tx_type = o.transaction_type.lower()
        is_import = tx_type in ("import", "buy")
        if bench and o.execution_rate:
            if is_import:
                gain_pct = (bench - o.execution_rate) / o.execution_rate
            else:
                gain_pct = (o.execution_rate - bench) / bench
            gain_tnd = gain_pct * o.original_amount * o.execution_rate
        else:
            gain_pct = gain_tnd = 0.0

        comm_pct = o.commission_percent or 0.0
        comm_tnd = o.execution_rate * o.original_amount * comm_pct

        def thousands(x):  return f"{x:,.3f}".replace(",", " ").replace(".", ",")
        fmt_eur  = lambda x: f" {thousands(x)}"
        fmt_tnd  = lambda x: f"{thousands(x)} TND"
        fmt_rate = lambda r: "" if (r is None or math.isnan(r)) else f"{r:,.4f}".replace(",", " ").replace(".", ",")
        fmt_pct  = lambda p: f"{p:.2f}%"

        # Include reference if required
        row = {
            "Date Transaction": o.transaction_date.strftime("%-d/%-m/%Y"),
            "Date Valeur": o.value_date.strftime("%-d/%-m/%Y"),
            "Devise": o.currency,
            "Type": o.transaction_type.title(),
            "Type d’opération": o.trade_type.title(),
            "Montant": fmt_eur(o.original_amount),
            "Taux d’exécution": fmt_rate(o.execution_rate),
            "Banque": o.bank_name or "N/A",
            "Taux de référence *": fmt_rate(bench),
            "% Gain": fmt_pct(gain_pct * 100),
            "Gain**": fmt_tnd(gain_tnd),
            "Commission CC ***": comm_tnd,
            "Commission CC": f"{comm_tnd:,.3f}".replace(",", " ").replace(".", ",") + " TND",
            "Commission Percent": fmt_pct((o.commission_percent or 0.0) * 100),
        }
        if hasattr(o, "reference"):
            row["Référence"] = o.reference or ""
        rows.append(row)
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PUT /clients/<id>/invoice-settings
# ──────────────────────────────────────────────────────────────────────────────
@invoice_bp.route("/clients/<int:cid>/invoice-settings", methods=["PUT"])
@jwt_required()
def update_invoice_settings(cid):
    if not _is_admin():
        return jsonify({"error": "Admin role required"}), 403

    u = User.query.get_or_404(cid)
    data = request.get_json() or {}

    # only allow the invoicing-related fields to be updated
    fields = [
        "address", "matricule_fiscal",
        "fixed_monthly_fee", "tva_exempt",
        "uses_digital_sign", "netting_enabled", "contract_start",  "needs_references"
    ]
    for f in fields:
        if f in data:
            if f == "fixed_monthly_fee":
                # If value is empty string or None, set to 0.0
                value = data[f]
                if value == "" or value is None:
                    setattr(u, f, 0.0)
                else:
                    setattr(u, f, float(value))
            elif f == "contract_start" and data[f]:
                setattr(u, f, date.fromisoformat(data[f]))
            else:
                setattr(u, f, data[f])
    db.session.commit()
    return jsonify({"message": "Settings updated"}), 200

@invoice_bp.route("/clients", methods=["GET"])
@jwt_required()
def list_clients():
    if not _is_admin():
        return jsonify({"error": "Admin role required"}), 403

    clients = User.query.all()
    return jsonify([
        {
            "id": u.id,
            "client_name": u.client_name,
            "address": u.address,
            "matricule_fiscal": u.matricule_fiscal,
            "fixed_monthly_fee": u.fixed_monthly_fee,
            "tva_exempt": u.tva_exempt,
            "uses_digital_sign": u.uses_digital_sign,
            "netting_enabled": u.netting_enabled,
            "needs_references": u.needs_references,
            "contract_start": u.contract_start.isoformat() if u.contract_start else None,
        }
        for u in clients
    ]), 200

# ──────────────────────────────────────────────────────────────────────────────
# helper – return list[dict] for frais variables using existing logic
# ──────────────────────────────────────────────────────────────────────────────
def _variable_fees(client: User, y: int, m: int) -> list[dict]:
    return compute_bank_gains(client.id, currency="USD")



from calendar import monthrange

def _fixed_fee_row(client: User, year: int, month: int) -> dict:
    """
    Calculate the prorated monthly fixed fee for the invoice.
    """
    fee = client.fixed_monthly_fee or 0.0
    nb_mois = 1.0

    # Only prorate for the first month of the contract
    if client.contract_start and client.contract_start.year == year and client.contract_start.month == month:
        # Number of days in the invoice month
        days_in_month = monthrange(year, month)[1]
        # Days from contract start to end of month (including the start day)
        nb_days_billed = (date(year, month, days_in_month) - client.contract_start).days + 1
        # Prorated months
        nb_mois = round(nb_days_billed / days_in_month, 4)  # Rounded for neatness

    return {
        "frais_mensuel": fee,
        "nb_mois": nb_mois,
        "total": round(fee * nb_mois, 3)
    }



def _stamp_duty() -> float:
    return 1.0          # static for now


def _today_year_month() -> tuple[int, int]:
    today = date.today()
    return today.year, today.month


# ──────────────────────────────────────────────────────────────────────────────
# 2.  POST /invoice/draft
# ──────────────────────────────────────────────────────────────────────────────
@invoice_bp.route("/draft", methods=["POST"])
@jwt_required()
def draft_invoice():
    if not _is_admin():
        return jsonify({"error": "Admin role required"}), 403

    data = request.get_json() or {}
    client_name = data.get("client_name")
    year  = int(data.get("year", _today_year_month()[0]))
    month = int(data.get("month", _today_year_month()[1]))

    client = User.query.filter_by(client_name=client_name).first()
    if not client:
        return jsonify({"error": f"Unknown client '{client_name}'"}), 404

    # --- frais variables ----------------------------------------------------
    orders = get_invoice_transactions(client.id, year, month)
    var_rows = build_invoice_rows(orders, netting_enabled=client.netting_enabled)

    total_comm_tnd = sum(
        _parse_tnd(r["Commission CC ***"]) for r in var_rows
    )
    # mixed-currency handling → total traded in TND
    total_traded_tnd = sum(
    _parse_tnd(r["Montant"]) * _parse_tnd(r["Taux d’exécution"])
    for r in var_rows
)

    # --- fixed fee ----------------------------------------------------------
    fixed_row = _fixed_fee_row(client, year, month)
    total_du  = fixed_row["total"]

    total_ht = total_comm_tnd + total_du
    tva_rate = 0.0 if client.tva_exempt else 0.19
    tva_amt  = total_ht * tva_rate
    stamp    = _stamp_duty()
    total_ttc = total_ht + tva_amt + stamp

    payload = {
        "client": {
            "name": client.client_name,
            "address": client.address,
            "matricule_fiscal": client.matricule_fiscal
        },
        "period": {"year": year, "month": month},
        "signature": client.uses_digital_sign,
        "needs_references": client.needs_references,
        "frais_variable": var_rows,
        "frais_fixe": fixed_row,
        "totals": {
            "total_traded_tnd": total_traded_tnd,
            "total_comm_tnd": total_comm_tnd,
            "total_du": total_du,
            "total_ht": total_ht,
            "tva": tva_amt,
            "stamp_duty": stamp,
            "total_ttc": total_ttc
        }
    }

    custom_creation_date = data.get("creation_date")
    if custom_creation_date:
        from datetime import datetime
        creation_date = datetime.strptime(custom_creation_date, "%Y-%m-%d").date()
    else:
        # Default: last day of the month
        last_day = date(year, month, 1).replace(day=28) + timedelta(days=4)
        creation_date = last_day - timedelta(days=last_day.day)

    inv = Invoice(
        client_id=client.id,
        year=year,
        month=month,
        creation_date=creation_date,  
        due_date=date.today() + timedelta(days=7),
        json_payload=payload,
        total_ht=total_ht, tva=tva_amt, stamp_duty=stamp,
        total_ttc=total_ttc
    )
    db.session.add(inv)
    db.session.commit()

    return jsonify({"invoice_id": inv.id}), 201

# ──────────────────────────────────────────────────────────────────────────────
# 3.  POST /invoice/<id>/confirm
# ──────────────────────────────────────────────────────────────────────────────
@invoice_bp.route("/<int:inv_id>/confirm", methods=["POST"])
@jwt_required()
def confirm_invoice(inv_id):
    if not _is_admin():
        return jsonify({"error": "Admin role required"}), 403

    inv = Invoice.query.get_or_404(inv_id)
    if inv.status != InvoiceStatus.draft:
        return jsonify({"error": "Only draft invoices can be confirmed"}), 400

    data = request.get_json() or {}
    pdf_url = data.get("pdf_url")
    if not pdf_url:
        return jsonify({"error": "`pdf_url` required"}), 400

    inv.pdf_url = pdf_url
    inv.status  = InvoiceStatus.sent
    db.session.commit()
    return jsonify({"message": "Invoice confirmed & locked"}), 200


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PATCH /invoice/<id>/status   {"status":"paid"}
# ──────────────────────────────────────────────────────────────────────────────
@invoice_bp.route("/<int:inv_id>/status", methods=["PATCH"])
@jwt_required()
def set_status(inv_id):
    if not _is_admin():
        return jsonify({"error": "Admin role required"}), 403

    inv = Invoice.query.get_or_404(inv_id)
    new_status = request.get_json().get("status", "").lower()
    if new_status not in {"sent", "paid"}:
        return jsonify({"error": "status must be 'sent' or 'paid'"}), 400

    inv.status = InvoiceStatus(new_status)
    db.session.commit()
    return jsonify({"message": "Status updated"}), 200


# ──────────────────────────────────────────────────────────────────────────────
# 5.  GET /invoices  (list & filter)
# ──────────────────────────────────────────────────────────────────────────────
# @invoice_bp.route("/invoices", methods=["GET"])
# @jwt_required()
# def list_invoices():
#     if not _is_admin():
#         return jsonify({"error": "Admin role required"}), 403

#     q = Invoice.query.join(Invoice.client)

#     client_name = request.args.get("client_name")
#     if client_name:
#         q = q.filter(func.lower(User.client_name) == client_name.lower())

#     if y := request.args.get("year"):
#         q = q.filter(Invoice.year == int(y))
#     if m := request.args.get("month"):
#         q = q.filter(Invoice.month == int(m))

#     rows = [{
#         "id": i.id,
#         "client": i.client.client_name,
#         "year": i.year,
#         "month": i.month,
#         "creation_date": i.creation_date.isoformat(),
#         "due_date": i.due_date.isoformat(),
#         "total_ttc": i.total_ttc,
#         "status": i.status.value,
#         "pdf_url": i.pdf_url
#     } for i in q.order_by(Invoice.creation_date.desc()).all()]

#     return jsonify(rows), 200
@invoice_bp.route("/invoices", methods=["GET"])
@jwt_required()
def list_invoices():
    uid = int(get_jwt_identity())
    user = User.query.get(uid)

    q = Invoice.query.join(Invoice.client)

    if _is_admin():
        client_name = request.args.get("client_name")
        if client_name:
            q = q.filter(func.lower(User.client_name) == client_name.lower())
    else:
        q = q.filter(Invoice.client_id == uid)

    if y := request.args.get("year"):
        q = q.filter(Invoice.year == int(y))
    if m := request.args.get("month"):
        q = q.filter(Invoice.month == int(m))

    rows = [{
        "id": i.id,
        "client": i.client.client_name,
        "year": i.year,
        "month": i.month,
        "creation_date": i.creation_date.isoformat(),
        "due_date": i.due_date.isoformat(),
        "total_ttc": i.total_ttc,
        "status": i.status.value,
        "pdf_url": i.pdf_url
    } for i in q.order_by(Invoice.creation_date.desc()).all()]

    return jsonify(rows), 200


# ──────────────────────────────────────────────────────────────────────────────
# 6.  GET /invoice/<id>  (single invoice detail)
# ──────────────────────────────────────────────────────────────────────────────
# @invoice_bp.route("/<int:inv_id>", methods=["GET"])
# @jwt_required()
# def get_invoice(inv_id):
#     if not _is_admin():
#         return jsonify({"error": "Admin role required"}), 403

#     i = Invoice.query.get_or_404(inv_id)
#     return jsonify({
#         "id": i.id,
#         "client": i.client.client_name,
#         "year": i.year, "month": i.month,
#         "creation_date": i.creation_date.isoformat(),
#         "due_date": i.due_date.isoformat(),
#         "status": i.status.value,
#         "pdf_url": i.pdf_url,
#         "payload": i.json_payload
#     }), 200

@invoice_bp.route("/<int:inv_id>", methods=["GET"])
@jwt_required()
def get_invoice(inv_id):
    uid = int(get_jwt_identity())          

    user = User.query.get(uid)

    inv = Invoice.query.get_or_404(inv_id)

    if not _is_admin() and inv.client_id != uid:
        return jsonify({"error": "Access denied"}), 403

    return jsonify({
        "id": inv.id,
        "client": inv.client.client_name,
        "year": inv.year,
        "month": inv.month,
        "creation_date": inv.creation_date.isoformat(),
        "due_date": inv.due_date.isoformat(),
        "status": inv.status.value,
        "pdf_url": inv.pdf_url,
        "payload": inv.json_payload
    }), 200

# ──────────────────────────────────────────────────────────────────────────────
# 7.  GET /invoices/summary     KPI cards for dashboard
# ──────────────────────────────────────────────────────────────────────────────
@invoice_bp.route("/summary", methods=["GET"])
@jwt_required()
def invoices_summary():
    if not _is_admin():
        return jsonify({"error": "Admin role required"}), 403

    y     = int(request.args.get("year", _today_year_month()[0]))
    month = request.args.get("month")          # optional

    q = Invoice.query.filter(Invoice.year == y)
    if month:
        q = q.filter(Invoice.month == int(month))

    total_facture    = func.sum(Invoice.total_ttc)
    total_ht         = func.sum(Invoice.total_ht)
    total_tva        = func.sum(Invoice.tva)

    total_paid = func.sum(
        case(
            (Invoice.status == InvoiceStatus.paid, Invoice.total_ttc),
            else_=0.0
        )
    )
    total_unpaid = func.sum(
        case(
            (Invoice.status != InvoiceStatus.paid, Invoice.total_ttc),
            else_=0.0
        )
    )
    res = db.session.query(
        total_facture, total_ht, total_tva, total_paid, total_unpaid
    ).one()

    tot, ht, tva, paid, unpaid = [float(x or 0) for x in res]
    ratio = (unpaid / tot * 100) if tot else 0.0

    return jsonify({
        "total_facture": tot,
        "total_ht": ht,
        "total_tva": tva,
        "paid": paid,
        "unpaid": unpaid,
        "unpaid_ratio_pct": ratio
    }), 200


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
# def _parse_tnd(txt: str) -> float:
#     """From '12 345,67 TND' ➜ 12345.67"""
#     return float(txt.replace(" ", " ").replace(" TND", "")
#                      .replace(",", ".").replace(" ", "")) if txt else 0.0
def _parse_tnd(txt) -> float:
    if isinstance(txt, (int, float)):
        return float(txt)
    return float(txt.replace(" ", "").replace(" TND", "").replace(",", ".")) if txt else 0.0

def _is_admin() -> bool:
    uid = int(get_jwt_identity())
    u   = User.query.get(uid)
    return any(r.name.lower() == "admin" for r in u.roles)
