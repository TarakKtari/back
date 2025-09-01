import base64
from io import BytesIO
import re
import os
import json
import random
import string
import numpy as np
import pandas as pd
from fpdf import FPDF
from sqlalchemy.orm import joinedload
from collections import defaultdict
from flask import Blueprint, abort, render_template, request, redirect, url_for, send_file, make_response, jsonify, flash, session
from datetime import datetime, timedelta
from sqlalchemy import func

from user.templates import _make_excel
from .services.export_service import export_pdf, download_excel
from models import db, Order, ExchangeData
from flask_login import login_user, logout_user, current_user
from functools import wraps
from models import User, Role, AuditLog, PremiumRate, InternalEmail, BctxFixing, TcaSpotInput
from flask_security import roles_accepted
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, get_jwt_identity, jwt_required, get_jwt, unset_jwt_cookies
from werkzeug.security import generate_password_hash, check_password_hash
from flask import current_app
from user.routes import get_interbank_rate_from_db, require_reference_if_needed, user_bp, fetch_rate_for_date_and_currency
from extentions import limiter, revoke_token          



admin_bp = Blueprint('admin_bp', __name__, template_folder='templates', static_folder='static')
PWD_RE = re.compile(r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).{8,}$")

def _validate_pwd(pwd):
    if not PWD_RE.match(pwd):
        raise ValueError(
            "Mot de passe trop faible : 8 caractères mini + majuscule + minuscule + chiffre"
        )

# Custom roles_required decorator
def roles_required(required_role):
    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            # Get the user identity from the JWT token
            user_id = int(get_jwt_identity())
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({"error": "User not found"}), 404

            # Check if the user has the required role
            user_roles = [role.name for role in user.roles]
            if required_role not in user_roles:
                return jsonify({"error": "Access forbidden, admin role required"}), 403

            # Proceed to the endpoint
            return fn(*args, **kwargs)
        return decorator
    return wrapper

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def generate_unique_key(buyer, seller):
    # Create a unique key based on the first 2 letters of buyer and seller names and 8 random digits
    random_digits = ''.join(random.choices(string.digits, k=8))
    return buyer[:1] + seller[:1] + random_digits

@limiter.limit("5 per minute")
@admin_bp.route('/signup', methods=['GET', 'POST'])
def sign():
    email = request.json.get('email')
    password = request.json.get('password')
    client_name = request.json.get('client_name')  
    role_id = request.json.get('options')
    rating = request.json.get('rating', 0)  

    # ─── 1. contrôle de robustesse du mot de passe ──────────
    try:
        _validate_pwd(password)                     # ← AJOUTE CES 4 LIGNES
    except ValueError as e:
        return jsonify(msg=str(e)), 400
    # ─────────────────────────────────────────────────────────

    if User.query.filter_by(email=email).first():
        return jsonify({"msg": "User already exists"}), 400

    hashed_password = generate_password_hash(password)
    user = User(email=email, active=1, password=hashed_password,client_name=client_name, rating=rating)  # Set the rating when creating the user
              
    role = Role.query.filter_by(id=int(role_id)).first()
    if not role:
        return jsonify({"msg": "Role not found"}), 400

    user.roles.append(role)
    db.session.add(user)
    db.session.commit()

    return jsonify({"msg": "User created successfully!"}), 201
from flask_jwt_extended import (
    create_access_token, create_refresh_token,
    set_access_cookies, set_refresh_cookies,
    unset_jwt_cookies, get_jwt, get_jwt_identity,
    jwt_required
)

@admin_bp.route('/signin', methods=['GET', 'POST'])
def signin():
    debug_msgs = []
    debug_msgs.append(f"BODY: {request.data}")
    try:
        data = request.get_json(force=True)
        debug_msgs.append(f"JSON: {data}")
    except Exception as e:
        debug_msgs.append(f"JSON decode error: {str(e)}")
        return jsonify(msg="Bad JSON", debug=debug_msgs), 400

    email = data.get("email")
    pwd = data.get("password")
    debug_msgs.append(f"Email: {email}")

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, pwd):
        debug_msgs.append("User not found or bad password")
        return jsonify(msg="Invalid email or password", debug=debug_msgs), 401

    access  = create_access_token(identity=str(user.id), fresh=True)
    refresh = create_refresh_token(identity=str(user.id))
    debug_msgs.append(f"Access/refresh token created for user id {user.id}")
    resp = jsonify(roles=[r.name for r in user.roles], debug=debug_msgs)
    set_access_cookies(resp, access)
    set_refresh_cookies(resp, refresh)
    return resp, 200


@admin_bp.post("/token/refresh")
@jwt_required(refresh=True)
def refresh():
    revoke_token(get_jwt())                    # blacklist old refresh
    identity    = get_jwt_identity()
    access      = create_access_token(identity=identity, fresh=False)
    new_refresh = create_refresh_token(identity=identity)

    resp = jsonify(msg="refreshed")
    set_access_cookies(resp,  access)
    set_refresh_cookies(resp, new_refresh)
    return resp, 200


@admin_bp.post("/logout")
@jwt_required()
def logout():
    revoke_token(get_jwt())           # blacklist access jti
    resp = jsonify(msg="logged out")
    unset_jwt_cookies(resp)
    return resp, 200

@admin_bp.get("/me")
@jwt_required()
def me():
    u = User.query.get(int(get_jwt_identity()))

    return jsonify(email=u.email,
                   roles=[r.name for r in u.roles])

# --------------------------------------------------------------------
# helper: ensure current user is an Admin
def _require_admin():
    user = User.query.get(int(get_jwt_identity()))
    if "Admin" not in [r.name for r in user.roles]:
        abort(403, "Admin role required")

@admin_bp.post("/reset-user-password")
@jwt_required()
def reset_user_password():
    _require_admin()               

    data     = request.get_json(force=True)
    target_id = data.get("user_id")
    new_pwd   = data.get("new_password")

    if not target_id or not new_pwd:
        return jsonify(msg="user_id and new_password required"), 400

    # ─── 1. contrôle de robustesse du nouveau mot de passe ──
    try:
        _validate_pwd(new_pwd)                        # ← AJOUTE ICI
    except ValueError as e:
        return jsonify(msg=str(e)), 400
    # ─────────────────────────────────────────────────────────

    user = User.query.get_or_404(target_id)
    user.password = generate_password_hash(new_pwd)
    db.session.commit()

    return jsonify(msg=f"Password reset for user {user.email}"), 200

@admin_bp.route('/api/clients', methods=['GET'])
@jwt_required()
@roles_required('Admin')
def list_clients():
    # only users with the “Client” role
    clients = User.query\
        .join(User.roles)\
        .filter(Role.name == 'Client')\
        .all()

    return jsonify([
        { "client_name": u.client_name, "id": u.id }
        for u in clients
    ]), 200


@admin_bp.route('/api/orders', methods=['GET'])
@jwt_required()
def view_all_orders():
    orders = Order.query.options(joinedload(Order.user)).filter(Order.deleted == False).all()
    if not orders:
        return jsonify([]), 200

    order_list = []
    for order in orders:
        order_list.append({
            "id": order.id,
            "user": order.user.email if order.user else "Unknown",
            "transaction_type": order.transaction_type,
            "trade_type": order.trade_type,  # NEW: Include trade type in response
            "amount": order.amount,
            "currency": order.currency,
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "transaction_date": order.transaction_date.strftime("%Y-%m-%d"),
            "status": order.status,
            "client_name": order.user.client_name if order.user else "Unknown",
            "execution_rate": order.execution_rate,
            "bank_name": order.bank_name,
            "interbank_rate": order.interbank_rate,
            "historical_loss": order.historical_loss,
            "premium": order.premium,
            "is_option": order.is_option,
            "option_type": order.option_type,     
            "strike": order.strike,               
            "moneyness": order.moneyness ,
            "reference": order.reference,

         
        })
    return jsonify(order_list), 200

# @admin_bp.route('api/orders/<int:order_id>', methods=['PUT'])
# @jwt_required()
# def update_order(order_id):
#     """
#     API for admins to update an order's status, execution rate, and bank name.
#     These fields are added by the admin and not by the client.
#     """
#     data = request.get_json()
    
#     # Fetch the order by ID
#     order = Order.query.get(order_id)
#     if not order:
#         return jsonify({"error": "Order not found"}), 404

#     # Admin can update these fields
#     order.status = data.get("status", order.status)  # Update status
#     order.execution_rate = data.get("execution_rate", order.execution_rate)  # Add/update execution rate
#     order.bank_name = data.get("bank_name", order.bank_name)  # Add/update bank name
#     order.historical_loss=data.get("historical_loss", order.historical_loss)
    
#     db.session.commit()  # Save changes
    
#     return jsonify({"message": "Order updated successfully"}), 200


@admin_bp.route('/run_matching', methods=['POST'])
@jwt_required()
@roles_required('Admin')  
def run_matching():
    """
    API to trigger the scheduled matching process manually.
    This allows Admins to run the matching process via a REST API call.
    """
    try:
        debug_messages = []  # List to capture debug messages
        # Call the matching function and pass the debug list
        process_matching_orders(current_app, debug_messages)
        return jsonify({
            'message': 'Matching process executed successfully',
            'debug_messages': debug_messages  # Include debug messages in the response
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

        
def process_matching_orders(app, debug_messages):
    """
    Processes matching of pending orders by grouping them by value_date and currency,
    and updates their statuses based on matching logic. Unmatched orders are marked as 'Market'.
    """
    try:
        # Fetch Pending Orders
        pending_orders = Order.query.filter_by(status='Pending', deleted=False).all()
        if not pending_orders:
            debug_messages.append("No pending orders found.")
            return {"debug_messages": debug_messages, "message": "No pending orders to process."}

        debug_messages.append(f"Fetched Pending Orders: {[order.id for order in pending_orders]}")

        # Convert to DataFrame for easier processing
        try:
            data = [
                {
                    'id': order.id,
                    'type': order.transaction_type,
                    'amount': order.amount,
                    'original_amount': order.original_amount,
                    'currency': order.currency,
                    'value_date': order.value_date,
                    'order_date': order.order_date,
                    'rating': order.rating,
                    'user_id': order.user_id 
                } for order in pending_orders
            ]
            df = pd.DataFrame(data)
            debug_messages.append(f"Converted {len(data)} orders to DataFrame.")
        except Exception as e:
            debug_messages.append(f"Error converting orders to DataFrame: {e}")
            return {"debug_messages": debug_messages, "message": "Failed to process orders."}

        # Group by value_date and currency
        try:
            groups = df.groupby(['value_date', 'currency'])
            debug_messages.append(f"Grouped orders into {len(groups)} groups.")
        except Exception as e:
            debug_messages.append(f"Error grouping orders: {e}")
            return {"debug_messages": debug_messages, "message": "Failed to group orders."}

        for (value_date, currency), group in groups:
            debug_messages.append(f"Processing Group: Value Date={value_date}, Currency={currency}")
            buy_orders = group[group['type'] == 'buy'].sort_values(by=['rating'], ascending=False)
            sell_orders = group[group['type'] == 'sell'].sort_values(by=['rating'], ascending=True)

            for _, buy_order in buy_orders.iterrows():
                for _, sell_order in sell_orders.iterrows():
                    if buy_order['amount'] <= 0:
                        debug_messages.append(f"Buy Order ID={buy_order['id']} has no remaining amount to match.")
                        break
                    if sell_order['amount'] <= 0:
                        debug_messages.append(f"Sell Order ID={sell_order['id']} has no remaining amount to match.")
                        continue
                    if buy_order['user_id'] == sell_order['user_id']:
                        
                        continue
                    # Match orders
                    match_amount = min(buy_order['amount'], sell_order['amount'])
                    buy_order['amount'] -= match_amount
                    sell_order['amount'] -= match_amount

                    # Update database records
                    try:
                        buy = Order.query.get(buy_order['id'])
                        sell = Order.query.get(sell_order['id'])

                        if buy is None or sell is None:
                            debug_messages.append(f"Error: Buy or Sell order not found for IDs: {buy_order['id']}, {sell_order['id']}.")
                            continue

                        buy.amount = buy_order['amount']
                        sell.amount = sell_order['amount']

                        # Update matched amounts for both orders
                        buy.matched_amount = (buy.matched_amount or 0) + match_amount
                        sell.matched_amount = (sell.matched_amount or 0) + match_amount

                        # Set statuses based on remaining amounts
                        if buy.amount == 0:
                            buy.status = 'Matched'
                            debug_messages.append(f"Buy Order ID={buy.id} fully matched and status updated to Matched.")
                        else:
                            buy.status = 'Market'
                            debug_messages.append(f"Buy Order ID={buy.id} partially matched and status updated to Market.")

                        if sell.amount == 0:
                            sell.status = 'Matched'
                            debug_messages.append(f"Sell Order ID={sell.id} fully matched and status updated to Matched.")
                        else:
                            sell.status = 'Market'
                            debug_messages.append(f"Sell Order ID={sell.id} partially matched and status updated to Market.")

                        # Link matched orders
                        buy.matched_order_id = sell.id
                        sell.matched_order_id = buy.id

                        db.session.add(buy)
                        db.session.add(sell)

                        debug_messages.append(
                            f"Matched Buy ID={buy.id} (Remaining Amount={buy.amount}) with "
                            f"Sell ID={sell.id} (Remaining Amount={sell.amount}) for Match Amount={match_amount}"
                        )
                    except Exception as e:
                        debug_messages.append(f"Error updating database records for Buy ID={buy_order['id']} or Sell ID={sell_order['id']}: {e}")

            # Update unmatched orders to 'Market' status
            try:
                for _, unmatched_order in group[group['amount'] > 0].iterrows():
                    unmatched = Order.query.get(unmatched_order['id'])
                    if unmatched and unmatched.status == 'Pending':
                        unmatched.status = 'Market'
                        db.session.add(unmatched)
                        debug_messages.append(f"Unmatched Order ID={unmatched.id} marked as Market.")
            except Exception as e:
                debug_messages.append(f"Error updating unmatched orders in group Value Date={value_date}, Currency={currency}: {e}")

        # Commit all changes
        try:
            db.session.commit()
            debug_messages.append("All changes committed to the database successfully.")
        except Exception as e:
            db.session.rollback()
            debug_messages.append(f"Error committing changes to the database: {e}")
            return {"debug_messages": debug_messages, "message": "Failed to commit changes to the database."}

        return {"debug_messages": debug_messages, "message": "Matching process executed successfully"}

    except Exception as e:
        db.session.rollback()
        debug_messages.append(f"Unexpected error: {str(e)}")
        return {"debug_messages": debug_messages, "message": "Matching process failed due to an unexpected error."}
     

@admin_bp.route('/matched_orders', methods=['GET'])
@jwt_required()
@roles_required('Admin')
def view_matched_orders():
    # Query orders with status 'Matched'
    matched_orders = Order.query.filter(Order.status == 'Matched').all()

    # Prepare the list of matched orders to return
    order_list = []
    for order in matched_orders:
        # Check for matched_order_id to determine the linked order
        matched_order = Order.query.get(order.matched_order_id) if order.matched_order_id else None
        
        # Add order details
        order_list.append({
            "id": order.id,
            "buyer": order.user.email if order.transaction_type == 'buy' else (matched_order.user.email if matched_order else None),
            "seller": order.user.email if order.transaction_type == 'sell' else (matched_order.user.email if matched_order else None),
            "currency": order.currency,
            "matched_amount": order.matched_amount,
            "amount": order.original_amount,
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "status": order.status,
            "execution_rate": order.execution_rate or "",  # Empty string if None
            "bank_name": order.bank_name or "",  # Empty string if None
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "order_date": order.order_date.strftime("%Y-%m-%d"),
            "trade_type": order.trade_type, 

        })

    return jsonify(order_list), 200


@admin_bp.route('/market_orders', methods=['GET'])
@jwt_required()
@roles_required('Admin')
def view_market_orders():
    """
    API to view orders with 'Market' status and their remaining unmatched amount.
    """
    try:
        # Fetch all orders with status 'Market'
        market_orders = Order.query.filter(Order.status.in_(['Market', 'Executed']),Order.deleted == False).all()

        if not market_orders:
            return jsonify([]), 200

        # Prepare the list of market orders to return
        order_list = []
        for order in market_orders:
            order_list.append({
                 "id": order.id,
                "transaction_type": order.transaction_type,
                "currency": order.currency,
                "amount": order.amount,
                "status": order.status,
                "execution_rate": order.execution_rate or "",  # Empty string if None
                "bank_name": order.bank_name or "",  # Empty string if None
                "value_date": order.value_date.strftime("%Y-%m-%d"),
                "order_date": order.order_date.strftime("%Y-%m-%d"),
                "client": order.user.email,
                "client_name": order.user.client_name,  
                "trade_type": order.trade_type, 

            })

        return jsonify(order_list), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# def register_admin_jobs(scheduler, app):
#     """
#     Register background jobs specific to admin functionality.
#     """
#     # Add a job for scheduled matching (run daily at 4:14 PM)
#     scheduler.add_job(
#         func=scheduled_matching,
#         trigger='cron',
#         hour=16,
#         minute=14,
#         args=[app],
#         id="scheduled_matching_job"
#     )

@admin_bp.route('/logs', methods=['GET'])
@jwt_required()
@roles_required('Admin')  # Restrict access to admins only
def get_logs():
    """
    API to retrieve audit logs. Allows filtering by action type, table name, user ID, and date range.
    """
    # Get filters from query parameters
    action_type = request.args.get('action_type')
    table_name = request.args.get('table_name')
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Start query with all logs
    query = AuditLog.query

    # Apply filters if provided
    if action_type:
        query = query.filter(AuditLog.action_type == action_type)
    if table_name:
        query = query.filter(AuditLog.table_name == table_name)
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if start_date:
        query = query.filter(AuditLog.timestamp >= datetime.strptime(start_date, "%Y-%m-%d"))
    if end_date:
        query = query.filter(AuditLog.timestamp <= datetime.strptime(end_date, "%Y-%m-%d"))

    # Execute query and retrieve logs
    logs = query.order_by(AuditLog.timestamp.desc()).all()

    # Format logs for JSON response
    log_list = [{
        "id": log.id,
        "action_type": log.action_type,
        "table_name": log.table_name,
        "record_id": log.record_id,
        "user_id": log.user_id,
        "user_email": log.user.email if log.user else "Unknown",
        "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "details": log.details
    } for log in logs]

    return jsonify(log_list), 200

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xls', 'xlsx'}

# ========== ADMIN: PREMIUM RATE ENDPOINTS ==========

@admin_bp.route('/api/premium-rate', methods=['POST'])
@jwt_required()
@roles_required('Admin')
def create_premium_rate():
    data = request.get_json()
    currency = data.get('currency')
    maturity_days = data.get('maturity_days')
    premium_percentage = data.get('premium_percentage')
    option_type = data.get('option_type')  # "CALL" or "PUT"
    transaction_type = data.get('transaction_type', 'buy')  # "buy" or "sell"
    spot = data.get('spot')
    strike = data.get('strike')

    # spot is required, strike is optional
    if not all([currency, maturity_days, premium_percentage, option_type, transaction_type, spot]):
        return jsonify({'error': 'Missing fields (currency, maturity_days, premium_percentage, option_type, transaction_type, spot)'}), 400

    new_rate = PremiumRate(
        currency=currency.upper(),
        maturity_days=int(maturity_days),
        premium_percentage=float(premium_percentage),
        option_type=option_type.upper(),
        transaction_type=transaction_type.lower(),
        spot=float(spot),
        strike=float(strike) if strike is not None else None
    )
    db.session.add(new_rate)
    db.session.commit()
    return jsonify({'message': 'Premium rate created'}), 201

@admin_bp.route('/api/premium-rate', methods=['GET'])
@jwt_required()
@roles_required('Admin')
def list_premium_rates():
    rates = PremiumRate.query.all()
    result = []
    for r in rates:
        result.append({
            'id': r.id,
            'currency': r.currency,
            'maturity_days': r.maturity_days,
            'premium_percentage': r.premium_percentage,
            'option_type': r.option_type,
            'transaction_type': r.transaction_type,
        })
    return jsonify(result), 200

@admin_bp.route('/api/premium-rate/<int:rate_id>', methods=['PUT'])
@jwt_required()
@roles_required('Admin')
def update_premium_rate(rate_id):
    data = request.get_json()
    rate = PremiumRate.query.get_or_404(rate_id)
    
    currency = data.get('currency', rate.currency)
    maturity_days = data.get('maturity_days', rate.maturity_days)
    premium_percentage = data.get('premium_percentage', rate.premium_percentage)
    option_type = data.get('option_type', rate.option_type)
    transaction_type = data.get('transaction_type', rate.transaction_type)  # NEW

    rate.currency = currency.upper()
    rate.maturity_days = int(maturity_days)
    rate.premium_percentage = float(premium_percentage)
    rate.option_type = option_type.upper()
    rate.transaction_type = transaction_type.lower()
    
    db.session.commit()
    return jsonify({'message': 'Premium rate updated'}), 200


@admin_bp.route('/api/premium-rate/<int:rate_id>', methods=['DELETE'])
@jwt_required()
@roles_required('Admin')
def delete_premium_rate(rate_id):
    rate = PremiumRate.query.get_or_404(rate_id)
    db.session.delete(rate)
    db.session.commit()
    return jsonify({'message': 'Premium rate deleted'}), 200

@admin_bp.route('/api/upsert-exchange-data', methods=['POST'])
def upsert_exchange_data():
    """
    Reads the latest JSON files (midmarket & yield) from /data folder,
    and upserts into the exchange_data table.
    Also back-fills any dates that appear only in the yield file.
    Returns debug logs in the API response.
    """
    debug_messages = []
    data_dir = os.path.join(current_app.root_path, 'data')
    debug_messages.append(f"Data directory: {data_dir}")

    try:
        all_files = os.listdir(data_dir)
        debug_messages.append(f"Files in directory: {all_files}")
    except Exception as e:
        return jsonify({'message': f'Error accessing data directory: {e}', 'debug': debug_messages}), 500

    mid_pattern   = re.compile(r"midmarket_rates_\d{4}-\d{2}-\d{2}_\d{4}\.json")
    yield_pattern = re.compile(r"daily_yield_rates_\d{4}-\d{2}-\d{2}_\d{4}\.json")

    mid_files   = sorted(f for f in all_files if mid_pattern.search(f))
    yield_files = sorted(f for f in all_files if yield_pattern.search(f))
    debug_messages.append(f"Found mid_files: {mid_files}")
    debug_messages.append(f"Found yield_files: {yield_files}")

    if not mid_files or not yield_files:
        return jsonify({'message': 'Required JSON files are missing', 'debug': debug_messages}), 404

    latest_mid   = os.path.join(data_dir, mid_files[-1])
    latest_yield = os.path.join(data_dir, yield_files[-1])
    debug_messages += [f"Latest mid file: {latest_mid}", f"Latest yield file: {latest_yield}"]

    try:
        mid_data   = json.load(open(latest_mid))
        yield_data = json.load(open(latest_yield))
        debug_messages += [
            f"Loaded {len(mid_data)} mid-data records",
            f"Loaded {len(yield_data)} yield-data records"
        ]
    except Exception as e:
        return jsonify({'message': f'Error reading JSON: {e}', 'debug': debug_messages}), 500

    # --- 1) Process all mid-market dates (with matching yields if available) ---
    processed_dates = set()

    for m in mid_data:
        ts = m.get('Timestamp')
        if not ts:
            debug_messages.append("Skipping mid record without Timestamp")
            continue

        try:
            d = datetime.fromisoformat(ts).date()
            debug_messages.append(f"Parsed mid date: {ts} => {d}")
        except Exception as e:
            debug_messages.append(f"Skipping mid parse error [{ts}]: {e}")
            continue

        processed_dates.add(d)

        # look for the matching yield entry
        match = next(
            (y for y in yield_data
             if y.get('Timestamp') 
                and datetime.fromisoformat(y['Timestamp']).date() == d),
            None
        )

        if match:
            debug_messages.append(f"Found matching yield for {d}")
            # TND yields
            tnd_ond = match.get('Mid_TNDOND')
            tnd_1m  = match.get('Mid_TND1M')
            tnd_3m  = match.get('Mid_TND3M')
            tnd_6m  = match.get('Mid_TND6M')
            tnd_1y  = match.get('Mid_TND1Y')
            # USD yields
            usd_ond = match.get('Mid_USDOND')
            usd_1m  = match.get('Mid_USD1M')
            usd_3m  = match.get('Mid_USD3M')
            usd_6m  = match.get('Mid_USD6M')
            usd_1y  = match.get('Mid_USD1Y')
            # EUR yields
            eur_ond = match.get('Mid_EUROND')
            eur_1m  = match.get('Mid_EUR1M')
            eur_3m  = match.get('Mid_EUR3M')
            eur_6m  = match.get('Mid_EUR6M')
            eur_1y  = match.get('Mid_EUR1Y')
        else:
            debug_messages.append(f"No yield match for {d}")
            tnd_ond = tnd_1m = tnd_3m = tnd_6m = tnd_1y = None
            usd_ond = usd_1m = usd_3m = usd_6m = usd_1y = None
            eur_ond = eur_1m = eur_3m = eur_6m = eur_1y = None

        rec = ExchangeData.query.filter_by(date=d).first()
        if rec:
            # update existing record
            rec.spot_usd = m.get('spotUSD') or 0.0
            rec.spot_eur = m.get('spotEUR') or 0.0
            # TND yields
            if tnd_ond is not None: rec.tnd_ond = tnd_ond
            if tnd_1m is not None:  rec.tnd_1m = tnd_1m
            if tnd_3m is not None:  rec.tnd_3m = tnd_3m
            if tnd_6m is not None:  rec.tnd_6m = tnd_6m
            if tnd_1y is not None:  rec.tnd_1y = tnd_1y
            # USD yields
            if usd_ond is not None: rec.usd_ond = usd_ond
            if usd_1m is not None:  rec.usd_1m = usd_1m
            if usd_3m is not None:  rec.usd_3m = usd_3m
            if usd_6m is not None:  rec.usd_6m = usd_6m
            if usd_1y is not None:  rec.usd_1y = usd_1y
            # EUR yields
            if eur_ond is not None: rec.eur_ond = eur_ond
            if eur_1m is not None:  rec.eur_1m = eur_1m
            if eur_3m is not None:  rec.eur_3m = eur_3m
            if eur_6m is not None:  rec.eur_6m = eur_6m
            if eur_1y is not None:  rec.eur_1y = eur_1y
            debug_messages.append(f"Updated record for {d}")
        else:
            # create new record
            new = ExchangeData(
                date=d,
                spot_usd=m.get('spotUSD') or 0.0,
                spot_eur=m.get('spotEUR') or 0.0,
                # TND yields
                tnd_ond=tnd_ond or 0.0,
                tnd_1m=tnd_1m or 0.0,
                tnd_3m=tnd_3m or 0.0,
                tnd_6m=tnd_6m or 0.0,
                tnd_1y=tnd_1y or 0.0,
                # USD yields
                usd_ond=usd_ond or 0.0,
                usd_1m=usd_1m or 0.0,
                usd_3m=usd_3m or 0.0,
                usd_6m=usd_6m or 0.0,
                usd_1y=usd_1y or 0.0,
                # EUR yields
                eur_ond=eur_ond or 0.0,
                eur_1m=eur_1m or 0.0,
                eur_3m=eur_3m or 0.0,
                eur_6m=eur_6m or 0.0,
                eur_1y=eur_1y or 0.0,
            )
            db.session.add(new)
            debug_messages.append(f"Created record for {d}")

    # --- 2) Now back-fill any dates that appear only in the yield file ---
    for y in yield_data:
        y_ts = y.get('Timestamp')
        if not y_ts:
            continue
        try:
            d = datetime.fromisoformat(y_ts).date()
        except:
            continue

        # skip any date we already did in the mid-loop
        if d in processed_dates:
            continue

        # these are yield-only dates - extract all yield curve data
        # TND yields
        tnd_ond = y.get('Mid_TNDOND') or 0.0
        tnd_1m  = y.get('Mid_TND1M')  or 0.0
        tnd_3m  = y.get('Mid_TND3M')  or 0.0
        tnd_6m  = y.get('Mid_TND6M')  or 0.0
        tnd_1y  = y.get('Mid_TND1Y')  or 0.0
        # USD yields
        usd_ond = y.get('Mid_USDOND') or 0.0
        usd_1m  = y.get('Mid_USD1M')  or 0.0
        usd_3m  = y.get('Mid_USD3M')  or 0.0
        usd_6m  = y.get('Mid_USD6M')  or 0.0
        usd_1y  = y.get('Mid_USD1Y')  or 0.0
        # EUR yields
        eur_ond = y.get('Mid_EUROND') or 0.0
        eur_1m  = y.get('Mid_EUR1M')  or 0.0
        eur_3m  = y.get('Mid_EUR3M')  or 0.0
        eur_6m  = y.get('Mid_EUR6M')  or 0.0
        eur_1y  = y.get('Mid_EUR1Y')  or 0.0

        rec = ExchangeData.query.filter_by(date=d).first()
        if rec:
            # update existing record with all yields
            rec.tnd_ond = tnd_ond
            rec.tnd_1m = tnd_1m
            rec.tnd_3m = tnd_3m
            rec.tnd_6m = tnd_6m
            rec.tnd_1y = tnd_1y
            rec.usd_ond = usd_ond
            rec.usd_1m = usd_1m
            rec.usd_3m = usd_3m
            rec.usd_6m = usd_6m
            rec.usd_1y = usd_1y
            rec.eur_ond = eur_ond
            rec.eur_1m = eur_1m
            rec.eur_3m = eur_3m
            rec.eur_6m = eur_6m
            rec.eur_1y = eur_1y
            debug_messages.append(f"Updated yields for existing date {d}")
        else:
            new = ExchangeData(
                date=d,
                spot_usd=0.0,  # no spot data
                spot_eur=0.0,
                # TND yields
                tnd_ond=tnd_ond,
                tnd_1m=tnd_1m,
                tnd_3m=tnd_3m,
                tnd_6m=tnd_6m,
                tnd_1y=tnd_1y,
                # USD yields
                usd_ond=usd_ond,
                usd_1m=usd_1m,
                usd_3m=usd_3m,
                usd_6m=usd_6m,
                usd_1y=usd_1y,
                # EUR yields
                eur_ond=eur_ond,
                eur_1m=eur_1m,
                eur_3m=eur_3m,
                eur_6m=eur_6m,
                eur_1y=eur_1y,
            )
            db.session.add(new)
            debug_messages.append(f"Inserted yield-only record for {d}")

    # finally commit
    try:
        db.session.commit()
        debug_messages.append("All changes committed")
        return jsonify({'message':'Exchange data upserted','debug':debug_messages}), 200
    except Exception as e:
        db.session.rollback()
        debug_messages.append(f"Commit failed: {e}")
        return jsonify({'message':f'Error updating DB: {e}','debug':debug_messages}),500

@admin_bp.route('/api/debug-exchange', methods=['GET'])
def debug_exchange():
    """
    Simple debug endpoint to list all records from the exchange_data table
    (via SQLAlchemy ORM). Note how we use model attributes, not raw DB column names.
    """
    data = ExchangeData.query.all()
    rows = []
    for row in data:
        row_data = {
            'id': row.id,
            'date': row.date.isoformat(),
            'spot_usd': row.spot_usd,
            'spot_eur': row.spot_eur,
            # TND yield curve (overnight, 1M, 3M, 6M, 1Y)
            'tnd_ond': getattr(row, 'tnd_ond', 0),
            'tnd_1m': row.tnd_1m,
            'tnd_3m': row.tnd_3m,
            'tnd_6m': row.tnd_6m,
            'tnd_1y': getattr(row, 'tnd_1y', 0),
            # USD yield curve (overnight, 1M, 3M, 6M, 1Y)
            'usd_ond': getattr(row, 'usd_ond', 0),
            'usd_1m': row.usd_1m,
            'usd_3m': row.usd_3m,
            'usd_6m': row.usd_6m,
            'usd_1y': getattr(row, 'usd_1y', 0),
            # EUR yield curve (overnight, 1M, 3M, 6M, 1Y)
            'eur_ond': getattr(row, 'eur_ond', 0),
            'eur_1m': row.eur_1m,
            'eur_3m': row.eur_3m,
            'eur_6m': row.eur_6m,
            'eur_1y': getattr(row, 'eur_1y', 0),
        }
        rows.append(row_data)
    return jsonify(rows), 200

@admin_bp.route('/api/internal-emails', methods=['GET'])
@jwt_required()
def get_internal_emails():
    user_id =  int(get_jwt_identity())
    user = User.query.get(user_id)
    
    # Optional query parameter "type" to filter emails by type
    email_type = request.args.get('type')
    
    # Admins see all emails; clients see only those sent to their email.
    if "Admin" in [role.name for role in user.roles]:
        query = InternalEmail.query
    else:
        query = InternalEmail.query.filter_by(recipient=user.email)
    
    if email_type:
        query = query.filter_by(email_type=email_type)
    
    emails = query.order_by(InternalEmail.timestamp.desc()).all()
    
    email_list = [{
        "id": email.id,
        "order_id": email.order_id,
        "email_type": email.email_type,
        "subject": email.subject,
        "body": email.body,
        "sender": email.sender,
        "recipient": email.recipient,
        "cc": email.cc,
        "timestamp": email.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "is_read": email.is_read
    } for email in emails]
    
    return jsonify(email_list), 200

@admin_bp.route('api/orders/<int:order_id>', methods=['PUT'])
@jwt_required()
def update_order(order_id):
    data = request.get_json()
    order = Order.query.get(order_id)
    if not order:
        return jsonify({"error": "Order not found"}), 404

    old_status = order.status
    order.status = data.get("status", order.status)
    order.execution_rate = data.get("execution_rate", order.execution_rate)
    order.bank_name = data.get("bank_name", order.bank_name)
    order.historical_loss=data.get("historical_loss", order.historical_loss)
    order.reference = data.get("reference", order.reference)     

    # Update any additional fields here...
    db.session.commit()

    # When the order is updated to Executed (and it was not Executed before), generate emails
    if old_status != "Executed" and order.status == "Executed":
        # Generate confirmation email (choose type based on trade_type; "spot" vs. "terme")
        conf_type = "spot" if order.trade_type == "spot" else "terme"
        subject_conf, body_conf, recipient_conf, cc_conf = generate_confirmation_email(order, conf_type)
        confirmation_email = InternalEmail(
            order_id=order.id,
            email_type="confirmation",
            subject=subject_conf,
            body=body_conf,
            sender="no-reply@yourcompany.com",
            recipient=recipient_conf,
            cc=cc_conf,
            timestamp=datetime.utcnow(),
            is_read=False
        )
        db.session.add(confirmation_email)

        # Generate interbank email using your interbank email generator
        subject_ib, body_ib, sender_ib, recipient_ib, cc_ib = generate_interbank_email(order)
        interbank_email = InternalEmail(
            order_id=order.id,
            email_type="interbank",
            subject=subject_ib,
            body=body_ib,
            sender=sender_ib,
            recipient=recipient_ib,
            cc=cc_ib,
            timestamp=datetime.utcnow(),
            is_read=False
        )
        db.session.add(interbank_email)

        db.session.commit()

    return jsonify({"message": "Order updated successfully"}), 200

def generate_interbank_email(order):
    """
    Generate interbank email content based on order details.
    Uses fetch_rate_for_date_and_currency to obtain the published interbank rate.
    """
    # Fetch the published rate for the order's transaction date and currency
    published_rate = fetch_rate_for_date_and_currency(order.transaction_date, order.currency)
    if not published_rate:
        published_rate = "N/A"

    # Calculate the difference in pips.
    # For 'achat' (buy/import) transactions: difference = (published_rate - execution_rate) * 1000
    # For 'vente' (sell/export) transactions: difference = (execution_rate - published_rate) * 1000
    if order.transaction_type.lower() in ["buy", "import"]:
        difference = (published_rate - order.execution_rate) * 1000 if published_rate != "N/A" else "N/A"
    else:
        difference = (order.execution_rate - published_rate) * 1000 if published_rate != "N/A" else "N/A"
    if difference != "N/A":
        difference = f"{difference:.2f}"
    exec_date_str = order.transaction_date.strftime("%d/%m/%Y")
    exec_date_phrase = order.transaction_date.strftime("%-d %B %Y")

    subject = f"Performance des transactions {order.currency}/TND du {exec_date_str} par rapport au niveau interbancaire"
    body = f"""Bonjour [M./Madame],

<p>Je reviens vers vous concernant les transactions d'achat en {order.currency} effectuées le {exec_date_phrase}.<br>
Suite à la publication du taux interbancaire par la Banque Centrale de Tunisie ({published_rate} selon la devise),<br>
nous avons constaté que la transaction a été exécutée à un taux spot de {order.execution_rate}, soit une performance supérieure de {difference} pips par rapport au marché interbancaire, pour un montant d'achat de {order.amount} {order.currency}.</p>

<p>Il est important de rappeler que le taux interbancaire représente une moyenne des transactions d'achat et de vente entre banques. Ces taux sont réels, fermes et issus d'opérations effectivement conclues sur le marché interbancaire.<br>
Par ailleurs, ces niveaux de cotation restent généralement inaccessibles aux entreprises, étant exclusivement réservés aux institutions bancaires. Cette performance témoigne de notre engagement à optimiser les conditions de change pour nos partenaires.</p>

<p>N'hésitez pas à nous contacter pour toute question ou clarification. Nous restons à votre entière disposition.</p>

<p>Cordialement,<br>
Cordialement,
[Capture d'écran des taux interbancaires]
"""
    sender = "no-reply@yourcompany.com"
    recipient = "bank@example.com"  # Adjust as needed
    cc = None  # Optionally, add CC if required

    return subject, body, sender, recipient, cc

def generate_confirmation_email(order, confirmation_type):
    """
    Generate confirmation email content.
    confirmation_type: "spot" or "terme"
    """
    if confirmation_type == "spot":
        subject = f"Détails de l'achat {order.currency}/TND – {order.user.client_name} – Date valeur: {order.value_date.strftime('%d/%m/%Y')}"
        body = f"""Bonjour ,

Je vous prie de bien vouloir trouver ci-dessous les détails relatifs à la transaction {order.currency}/TND, validée par notre client (en copie): /n
• Taux: {order.execution_rate}
• Montant (en {order.currency}): {order.amount}
• Équivalent en TND: {order.execution_rate * order.amount} (calculé sur la base d’un taux de {order.execution_rate} {order.currency}/TND)
• Date de valeur: {order.value_date.strftime('%d/%m/%Y')}

Cette transaction reflète une opération où {order.user.client_name} procède à l’achat de la devise, soit {"le dollars américain" if order.currency.upper() == "USD" else "l’euro"}, contre le dinar tunisien.

Nous restons à votre disposition pour toute information complémentaire.
Cordialement,
"""
        # In your real system, these addresses come from inputs or config:
        recipient = "bank@example.com"
        cc = order.user.email
    elif confirmation_type == "terme":
        subject = f"Détails des transactions vente à terme {order.currency}/TND – SBF - Domiciliation BIAT"
        body = f"""Bonjour,

Dans le cadre des récentes opérations de change {order.currency}/TND, nous vous transmettons ci-dessous un résumé détaillé de la transaction effectuée, conformément aux instructions reçues et validées par notre client (en copie):

Devise: {order.currency}
Montant encaissé: {order.amount}
Date de valeur: {order.value_date.strftime('%d/%m/%Y')}
Spot: [Spot Rate]
Cours à terme: [Forward Rate]
Montant en TND: [Calculé sur la base du taux d’exécution]

Cette transaction reflète une opération où SBF procède à la vente de la devise, soit l’euro, contre le dinar tunisien.
Je tiens également à vous informer que {order.user.client_name}, en copie, va vous confirmer cette transaction également.

Nous restons à votre disposition pour toute clarification.
Cordialement,
"""
        recipient = "bank@example.com"
        cc = order.user.email
    else:
        raise ValueError("Invalid confirmation type")
    return subject, body, recipient, cc

import io
import pandas as pd
from flask import send_file

@admin_bp.route('/generate-my-excel', methods=['GET'])
@jwt_required()
def generate_my_excel():
    current_user_id = int(get_jwt_identity())
    orders = Order.query.filter_by(user_id=current_user_id).all()

    data_rows = []
    for o in orders:
        row = {
            "Date Transaction" : o.transaction_date.strftime("%d/%m/%Y") if o.transaction_date else "",
            "Date valeur"      : o.value_date.strftime("%d/%m/%Y")       if o.value_date       else "",
            "Devise"           : o.currency,
            "Type"             : o.transaction_type,     # buy / sell …
            "Type d’opération" : o.trade_type,           # spot / forward / option
            "Montant (fc)"     : o.original_amount,
            "Taux d’execution" : o.execution_rate or "",
            "Banque"           : o.bank_name or "",
            "Taux de référence": o.benchmark_rate or "",
        }

        # add “Référence” only when it carries a real value
        if o.reference not in (None, "", 0, "0"):
            row["Référence"] = o.reference

        data_rows.append(row)

    df = pd.DataFrame(data_rows)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='OrdersData')
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="OrdersReport.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def pick(rec: dict, ric: str, side: str):
    """
    Retourne la valeur v telle que la clé k contient à la fois
    le RIC (ex: 'EURTNDX=BCTX') et le side ('BID' ou 'ASK').
    Si rien trouvé -> None
    """
    for k, v in rec.items():
        if ric in k and side in k:
            return v
    return None


@admin_bp.route('/api/upsert-bctx', methods=['POST'])
def upsert_bctx_data():
    """
    Reads the LATEST bctx_data_YYYY-MM-DD_HHMM.json file from /app/data,
    then upserts into bctx_fixings. 
    """

    debug_messages = []
    data_dir = os.path.join(current_app.root_path, 'data')
    
    # 1) Find the newest bctx_data_... file
    pattern = re.compile(r"bctx_data_(\d{4}-\d{2}-\d{2}_\d{4})\.json")
    
    try:
        all_files = os.listdir(data_dir)
        bctx_files = sorted([f for f in all_files if pattern.search(f)])
        if not bctx_files:
            return jsonify({"message": "No bctx_data_*.json files found"}), 404
        
        # The last one is the newest
        latest_bctx_file = bctx_files[-1]
        bctx_file_path = os.path.join(data_dir, latest_bctx_file)

    except Exception as e:
        return jsonify({"message": f"Error accessing data_dir or listing files: {str(e)}"}), 500

    # 2) Read that JSON file
    try:
        with open(bctx_file_path, 'r') as f:
            bctx_data = json.load(f)
    except Exception as e:
        return jsonify({"message": f"Error reading {latest_bctx_file}: {str(e)}"}), 500

    if not isinstance(bctx_data, list):
        return jsonify({"message": "Invalid JSON structure: expected a list"}), 400

    records_processed = 0

    # 3) Upsert logic
    for rec in bctx_data:
        # Try standard key
        ts_str = rec.get("Timestamp")

        # If not found, try fallback pattern like "('Timestamp', '')"
        if not ts_str:
            for key in rec:
                if "Timestamp" in key:
                    ts_str = rec[key]
                    break

        if not ts_str:
            debug_messages.append("Skipping record with missing Timestamp.")
            continue

        try:
            from datetime import datetime
            ts = datetime.fromisoformat(ts_str.replace("Z",""))
        except ValueError as e:
            debug_messages.append(f"Skipping invalid Timestamp '{ts_str}': {e}")
            continue

        if ts.hour < 12:
            session = "morning"
        else:
            session = "afternoon"

        the_date = ts.date()

        # Extract the numeric data
       
        tnd_bid = pick(rec, "TND=BCTX", "BID")
        tnd_ask = pick(rec, "TND=BCTX", "ASK")

        eur_bid = pick(rec, "EURTNDX=BCTX", "BID")
        eur_ask = pick(rec, "EURTNDX=BCTX", "ASK")

        gbp_bid = pick(rec, "GBPTNDX=BCTX", "BID")
        gbp_ask = pick(rec, "GBPTNDX=BCTX", "ASK")

        jpy_bid = pick(rec, "JPYTNDX=BCTX", "BID")
        jpy_ask = pick(rec, "JPYTNDX=BCTX", "ASK")

        # Check if we already have a row for that date+session
        existing = BctxFixing.query.filter_by(date=the_date, session=session).first()
        if existing:
            existing.original_timestamp = ts
            existing.tnd_bid = tnd_bid
            existing.tnd_ask = tnd_ask
            existing.eur_bid = eur_bid
            existing.eur_ask = eur_ask
            existing.gbp_bid = gbp_bid
            existing.gbp_ask = gbp_ask
            existing.jpy_bid = jpy_bid
            existing.jpy_ask = jpy_ask
            debug_messages.append(f"Updated {the_date} {session}.")
        else:
            new_row = BctxFixing(
                date=the_date,
                session=session,
                original_timestamp=ts,
                tnd_bid=tnd_bid,
                tnd_ask=tnd_ask,
                eur_bid=eur_bid,
                eur_ask=eur_ask,
                gbp_bid=gbp_bid,
                gbp_ask=gbp_ask,
                jpy_bid=jpy_bid,
                jpy_ask=jpy_ask
            )
            db.session.add(new_row)
            debug_messages.append(f"Inserted new record for {the_date} {session}.")

        records_processed += 1

    # 4) Commit
    try:
        db.session.commit()
        debug_messages.append(f"Processed {records_processed} records from {latest_bctx_file}.")
        return jsonify({
            "message": "BCTX fixings upsert complete",
            "records_processed": records_processed,
            "debug": debug_messages
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"DB commit error: {str(e)}"}), 500

@admin_bp.route('/api/tca/inputs', methods=['POST'])
@jwt_required()
def upload_tca_for_client():
    # 1) get client_name from query or form
    client_name = request.args.get('client_name') or request.form.get('client_name')
    if not client_name:
        return jsonify({"error": "Must include client_name (e.g. ?client_name=AcmeCorp)"}), 400

    # 2) look up that user
    user = User.query.filter_by(client_name=client_name).first()
    if not user:
        return jsonify({"error": f"No client found with name '{client_name}'"}), 404
    client_id = user.id

    # 3) grab the Excel file
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "Please attach an .xls or .xlsx file under key ‘file’"}), 400
    f = request.files['file']

    # 4) read with pandas
    try:
        df = pd.read_excel(f)
    except Exception as e:
        return jsonify({"error": f"Could not read Excel: {e}"}), 400

    # 5) validate columns
    required = {
      "transaction_date", "value_date",
      "currency", "amount",
      "execution_rate", "transaction_type"
    }
    missing = required - set(df.columns.str.lower())
    if missing:
        return jsonify({"error": f"Missing columns: {sorted(missing)}"}), 400

    # 6) clear old inputs for that client
    TcaSpotInput.query.filter_by(client_id=client_id).delete()

    # 7) bulk‐create new inputs
    objs = []
    for row in df.itertuples(index=False):
        objs.append(TcaSpotInput(
            client_id        = client_id,
            transaction_date = pd.to_datetime(row.transaction_date).date(),
            value_date       = pd.to_datetime(row.value_date).date(),
            currency         = row.currency.upper(),
            amount           = float(row.amount),
            execution_rate   = float(row.execution_rate),
            transaction_type = row.transaction_type.lower(),
        ))
    db.session.bulk_save_objects(objs)
    db.session.commit()

    return jsonify({
        "message":   f"{len(objs)} records uploaded for '{client_name}'",
        "client_id": client_id
    }), 200



# -------------  download template for ADMINS (NEW) --------------------
@admin_bp.get("/download-orders-template")
@jwt_required()
@roles_required("Admin")
def download_orders_template_admin():
    cols = [
        "Transaction date", "Value date", "Currency", "Type",
        "Amount", "Execution rate", "Bank", "Interbancaire",
        "Historical Loss", "Commission %", "Trade Type",  # ← req.
        "TND Rate", "Reference"                                       # ← optional
    ]
    sample = {
        "Transaction date": "2025/07/01",
        "Value date":       "2025/07/04",
        "Currency":         "USD",
        "Type":             "buy",
        "Amount":           1000000,
        "Execution rate":   3.1450,
        "Bank":             "BNP",
        "Interbancaire":    3.1400,
        "Historical Loss":  0.009,
        "Commission %":     0.15,
        "Trade Type":       "spot",
        "TND Rate":         3.0,    
        "Reference":        "azerty",          
    }
    return _make_excel(cols, sample, filename="OrdersTemplate_Admin.xlsx")
@admin_bp.post("/upload-orders")
@jwt_required()
@roles_required("Admin")
def upload_orders():
    """
    Bulk-upload Spot / Forward / Option orders for **one** client.
    Mandatory Excel columns
        Transaction date | Value date | Currency | Type | Amount
        Execution rate | Bank | Historical Loss | Commission % | Trade Type
    Optional:
        Reference | Interbancaire  (leave blank to auto-fill)
    """
    # 0) sanity -------------------------------------------------------
    if "file" not in request.files:
        return {"error": "No file provided"}, 400
    if "client_id" not in request.form:
        return {"error": "client_id is required"}, 400
    try:
        client_id = int(request.form["client_id"])
    except ValueError:
        return {"error": "client_id must be an integer"}, 400

    client = User.query.get(client_id)
    if not client:
        return {"error": f"No user found for id={client_id}"}, 404

    file = request.files["file"]
    if not allowed_file(file.filename):
        return {"error": "Invalid file format – please upload .xls or .xlsx"}, 400

    # 1) read --------------------------------------------------------
    df = pd.read_excel(BytesIO(file.read()))

    required = {
        "Transaction date", "Value date", "Currency", "Type",
        "Amount", "Execution rate", "Bank",
        "Historical Loss", "Commission %", "Trade Type"
    }
    missing = required - {c.strip() for c in df.columns}
    if missing:
        return {"error": f"Missing columns: {sorted(missing)}"}, 400

    # Optional columns
    if "Reference"      not in df.columns: df["Reference"]      = ""
    if "Interbancaire"  not in df.columns: df["Interbancaire"]  = np.nan
    if "TND Rate"       not in df.columns: df["TND Rate"]       = 1.0  # Default to 1.0 if not provided

    # 2) clean / coerce ---------------------------------------------
    df["Transaction date"] = pd.to_datetime(df["Transaction date"])
    df["Value date"]       = pd.to_datetime(df["Value date"])
    df["Amount"]           = df["Amount"].replace(",", "", regex=True).astype(float)
    df["Execution rate"]   = df["Execution rate"].replace(",", ".", regex=True).astype(float)
    df["Historical Loss"]  = df["Historical Loss"].replace(",", ".", regex=True).astype(float)
    df["Commission %"]     = df["Commission %"].replace(",", ".", regex=True).astype(float)
    df["TND Rate"]         = df["TND Rate"].replace(",", ".", regex=True).astype(float)
    df["Trade Type"]       = df["Trade Type"].str.lower().str.strip()
    df["Type"]             = df["Type"].str.lower().str.strip()

    uploaded, updated = 0, 0
    for idx, row in df.iterrows():

        # When admin leaves Interbancaire blank ➜ fetch automatically
        ib_val = row["Interbancaire"]
        if ib_val in (None, "", 0, 0.0, np.nan):
            ib_val = get_interbank_rate_from_db(row["Transaction date"].date(),
                                                row["Currency"].upper())
            if ib_val is None:
                return {"error": f"Row {idx+2}: no interbank rate for "
                                  f"{row['Currency']} on {row['Transaction date'].date()}"}, 400

        # reference check only if the **client** requires it
        try:
            require_reference_if_needed(client, {"reference": row["Reference"]})
        except ValueError as exc:
            return {"error": f"Row {idx+2}: {exc}"}, 400

        #lookup = dict(
        #    transaction_date = row["Transaction date"],
        #    value_date       = row["Value date"],
        #    currency         = row["Currency"].upper(),
        #    transaction_type = row["Type"],
        #    amount           = row["Amount"],
        #    user_id          = client.id,
        #)

        ref = row["Reference"]
        reference = None if not ref or str(ref).strip() == "-" else ref
        common = dict(
            reference          = reference,
            bank_name          = row["Bank"],
            execution_rate     = row["Execution rate"],
            interbank_rate     = ib_val,
            historical_loss    = row["Historical Loss"],
            commission_percent = row["Commission %"],
            trade_type         = row["Trade Type"],
            tnd_rate           = row["TND Rate"],  # Store TND rate for fee calculation
            status             = "Executed",
        )

        lookup = dict(
            transaction_date = row["Transaction date"],
            value_date       = row["Value date"],
            currency         = row["Currency"].upper(),
            transaction_type = row["Type"],
            amount           = row["Amount"],
            user_id          = client.id,
        )

        order = Order.query.filter_by(**lookup).first()
        if order:
            for k, v in common.items():
                setattr(order, k, v)
            updated += 1
        else:
            db.session.add(Order(**lookup,
                                 original_amount=row["Amount"],
                                 order_date=datetime.utcnow(),
                                 **common))
            uploaded += 1

    db.session.commit()
    return {
        "message": "Orders processed",
        "uploaded_count": uploaded,
        "updated_count": updated
    }, 200
