from datetime import datetime, timedelta, date
from scipy.interpolate import interp1d, CubicSpline
from models import db, User, Order, AuditLog, ExchangeData, OpenPosition, PremiumRate, InterbankRate, BctxFixing, TcaSpotInput, ATMVol
from option_pricer.volatility_pricer import get_db_connection, get_latest_exchange_data, reverse_premium_from_vol, strike_from_spot, garman_kohlhagen_greeks

from matplotlib import pyplot as plt
import numpy as np
from sqlalchemy import func
from flask import  request, session, jsonify, Blueprint, flash, send_from_directory
import pandas as pd
import uuid
from flask_socketio import SocketIO
from .utils import convert_to_date, allowed_file
from .services.live_rates_service import update_currency_rates, rates, metric, rates_all, lastUpdated, socketio
from flask_jwt_extended import jwt_required, get_jwt_identity
from io import BytesIO
import json
import requests
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from .templates import _make_excel


# ============ Utility Functions ==================
def load_atm_vols_from_json(json_path):
    """
    Load ATM volatility data from JSON file into the database.
    """
    # Read the JSON file
    df = pd.read_json(json_path)
    
    # Handle the MultiIndex column structure from Eikon data
    # The columns come as tuples like ("('EUR1WO=', 'BID')")
    # We need to flatten and properly parse them
    
    # If we have the timestamp/date column
    if "('Date', '')" in df.columns:
        df = df.rename(columns={"('Date', '')": "Timestamp"})
    
    # Define RICs and tenors
    ric_tenor_map = {
        "EUR1WO=": "1W",
        "EUR1MO=": "1M", 
        "EUR2MO=": "2M",
        "EUR3MO=": "3M",
        "EUR6MO=": "6M",
        "EUR9MO=": "9M",
        "EUR1YO=": "1Y"
    }
    
    # Tenor to days mapping
    tenor_days = {
        "1W": 7,
        "1M": 30,
        "2M": 60,
        "3M": 90,
        "6M": 180,
        "9M": 270,
        "1Y": 365
    }
    
    for _, row in df.iterrows():
        date = pd.to_datetime(row['Timestamp']).date()
        for ric, tenor in ric_tenor_map.items():
            # Handle the tuple-formatted column names
            bid_col = f"('{ric}', 'BID')"
            ask_col = f"('{ric}', 'ASK')"
            mid_col = f"('{ric}', 'CF_LAST')"
            
            bid = row.get(bid_col)
            ask = row.get(ask_col)
            mid = row.get(mid_col)
            
            tau_days = tenor_days.get(tenor)
            tau_years = tau_days / 365.0 if tau_days else None
            
            if bid is not None or ask is not None or mid is not None:
                existing = ATMVol.query.filter_by(date=date, tenor=tenor).first()
                if not existing:
                    db.session.add(ATMVol(
                        date=date, tenor=tenor, bid=bid, ask=ask, mid=mid,
                        tau_days=tau_days, tau_years=tau_years
                    ))
    db.session.commit()


user_bp = Blueprint('user_bp', __name__, static_folder='static', static_url_path='/static/user_bp',
                    template_folder='templates')


# ============ Open positions Routes ==================
@user_bp.route('/upload-open-positions', methods=['POST'])
@jwt_required()
def upload_open_positions():
    """
    API for clients to upload open positions in bulk via an Excel file.
    """
    if 'file' not in request.files or not allowed_file(request.files['file'].filename):
        return jsonify({'error': 'Invalid file format'}), 400

    file = request.files['file']

    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)

        file_stream = file.stream
        file_stream.seek(0)
        bytes_io = BytesIO(file_stream.read())
        uploaded_data = pd.read_excel(bytes_io)

        for index, row in uploaded_data.iterrows():
            value_date = convert_to_date(row['Value Date'])
            currency = row['Currency']
            amount = row['Amount']
            transaction_type = row['Transaction Type']

            new_open_pos = OpenPosition(
                value_date=value_date,
                currency=currency,
                amount=amount,
                transaction_type=transaction_type,
                user=user
            )
            db.session.add(new_open_pos)

        db.session.commit()
        log_action(
            action_type='bulk_upload',
            table_name='open_position',
            record_id=-1,
            user_id=user_id,
            details={"uploaded_open_positions": len(uploaded_data)}
        )

        return jsonify({'message': f'{len(uploaded_data)} open positions uploaded'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
@user_bp.route('/open-positions', methods=['GET'])
@jwt_required()
def get_open_positions():
    """
    Returns the list of open positions for the logged-in user.
    """
    try:
        user_id = get_jwt_identity()
        open_positions = OpenPosition.query.filter_by(user_id=user_id).all()
        data = []
        for pos in open_positions:
            data.append({
                "id": pos.id,
                "value_date": pos.value_date.strftime('%Y-%m-%d'),  
                "currency": pos.currency,
                "amount": pos.amount,
                "transaction_type": pos.transaction_type,
            })

        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@user_bp.route('/convert-open-position/<int:open_position_id>', methods=['POST'])
@jwt_required()
def convert_open_position(open_position_id):
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)

        open_pos = OpenPosition.query.get_or_404(open_position_id)
        new_order = Order(
            value_date=open_pos.value_date,
            currency=open_pos.currency,
            amount=open_pos.amount,
            original_amount=open_pos.amount,
            transaction_type=open_pos.transaction_type,
            user_id=open_pos.user_id, 
            transaction_date=datetime.now(),
            order_date=datetime.now(),
            status='Market',
        )
        db.session.add(new_order)

        db.session.delete(open_pos)
        db.session.commit()
        log_action(
            action_type='convert_open_position',
            table_name='open_position',
            record_id=open_position_id,
            user_id=user_id,
            details={"message": "Converted open_position to Order"}
        )

        return jsonify({'message': 'Open position converted to Order'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def delete_expired_positions(app):
    with app.app_context():
        today = date.today()
        expired_positions = OpenPosition.query.filter(OpenPosition.value_date < today).all()
        for pos in expired_positions:
            db.session.delete(pos)
        db.session.commit()

@user_bp.route('/orders', methods=['POST'])
@jwt_required()
def submit_order_or_option():
    debug_logs = []
    user_id = get_jwt_identity()
    debug_logs.append(f"Fetched user ID from JWT: {user_id}")

    user = User.query.get(user_id)
    if not user:
        debug_logs.append("User not found in the database")
        return jsonify({"message": "Invalid user", "debug": debug_logs}), 400
    debug_logs.append(f"User found: {user.email}")

    data = request.get_json()
    try:
        require_reference_if_needed(user, data)         
    except ValueError as e:
                   return jsonify({"message": "No data provided", "debug": debug_logs}), 400

    debug_logs.append(f"Raw incoming data: {data}")
    transaction_type = data.get('transaction_type')      
    amount = data.get('amount', 0)
    currency = data.get('currency')
    value_date_str = data.get('value_date')                
    bank_account = data.get('bank_account')
    is_option = data.get('is_option', False)
        # For options, also accept call/put and strike
    option_type = data.get('option_type', '').upper()      
    user_strike = data.get('strike', None)                

    if not transaction_type or not currency or not value_date_str:
        debug_logs.append("Missing required fields (transaction_type, currency, value_date).")
        return jsonify({"message": "Missing required fields", "debug": debug_logs}), 400

    try:
        amount = float(amount)
        value_date = datetime.strptime(value_date_str, "%Y-%m-%d")
        debug_logs.append(
            f"Parsed data: transaction_type={transaction_type}, amount={amount}, "
            f"currency={currency}, value_date={value_date}, bank_account={bank_account}, "
            f"is_option={is_option}, option_type={option_type}, strike={user_strike}"
        )
    except Exception as e:
        debug_logs.append(f"Error parsing data: {str(e)}")
        return jsonify({"message": "Invalid data format", "debug": debug_logs}), 400
    # ------------------------------- parse payload ----------------
    data = request.get_json()
    try:
        require_reference_if_needed(user, data)
    except ValueError:
        return jsonify({"message": "No data provided", "debug": debug_logs}), 400

    debug_logs.append(f"Raw incoming data: {data}")

    # ---------------------------------------------------------------
    #   T R A D E - T Y P E   L O G I C   (spot | forward | option)
    # ---------------------------------------------------------------
    is_option = data.get("is_option", False)

    incoming_trade_type = (data.get("trade_type") or "spot").lower()
    allowed_trade_types = ("spot", "forward", "option")

    # 1) basic validation ------------------------------
    if incoming_trade_type not in allowed_trade_types:
        debug_logs.append(f"Invalid trade_type: {incoming_trade_type}")
        return jsonify({
            "message": f"trade_type must be one of {allowed_trade_types}",
            "debug": debug_logs,
        }), 400

    # 2) consistency checks ----------------------------
    if incoming_trade_type == "option" and not is_option:
        return jsonify({
            "error": "If trade_type is 'option' you must also send is_option=true",
            "debug": debug_logs,
        }), 400

    if is_option and incoming_trade_type in ("spot", "forward"):
        return jsonify({
            "error": "Use either is_option=true *or* trade_type='forward/spot', not both",
            "debug": debug_logs,
        }), 400

    # 3) final decision & default status ---------------
    trade_type = "option" if is_option else incoming_trade_type
    status     = "Market" if trade_type in ("option") else "Pending"
        
    computed_premium = None
    moneyness = None
    computed_forward = None
    final_strike = None

    if is_option:
        if option_type not in ["CALL", "PUT"]:
            debug_logs.append(f"Invalid option_type: {option_type}")
            return jsonify({"message": "Option type must be CALL or PUT", "debug": debug_logs}), 400

        # 1) Calculate days until maturity
        today = datetime.now().date()
        days_diff = (value_date.date() - today).days
        debug_logs.append(f"Days until maturity: {days_diff}")

        # 2) Retrieve today's exchange data to compute forward rate
        exchange_data = ExchangeData.query.filter_by(date=today).first()
        if not exchange_data:
            # Fallback: most recent snapshot
            exchange_data = (ExchangeData.query
                            .order_by(ExchangeData.date.desc())
                            .first())
            if not exchange_data:
                debug_logs.append("No exchange data found at all")
                return jsonify({"message": "Exchange data not available", "debug": debug_logs}), 400
            else:
                debug_logs.append(f"Falling back to latest exchange snapshot: {exchange_data.date}")

        # Check if admin provided a spot rate in the request
        admin_spot = data.get("spot")
        if admin_spot is not None:
            try:
                spot_rate = float(admin_spot)
                debug_logs.append(f"Admin spot input used: {spot_rate}")
            except Exception as e:
                debug_logs.append(f"Error parsing admin spot: {e}")
                return jsonify({"message": "Spot must be numeric", "debug": debug_logs}), 400
            if currency.upper() == "USD":
                yield_foreign = getattr(exchange_data, f"usd_{get_yield_period(days_diff)[0]}m")
            elif currency.upper() == "EUR":
                yield_foreign = getattr(exchange_data, f"eur_{get_yield_period(days_diff)[0]}m")
            else:
                debug_logs.append(f"Unsupported currency: {currency}")
                return jsonify({"message": "Unsupported currency", "debug": debug_logs}), 400
            yield_domestic = getattr(exchange_data, f"tnd_{get_yield_period(days_diff)[0]}m")
            computed_forward = calculate_forward_rate(spot_rate, yield_foreign, yield_domestic, days_diff)
            debug_logs.append(f"Computed forward rate (admin spot): {computed_forward}")

        else:
            if currency.upper() == "USD":
                spot_rate = exchange_data.spot_usd
                yield_foreign = getattr(exchange_data, f"usd_{get_yield_period(days_diff)[0]}m")
            elif currency.upper() == "EUR":
                spot_rate = exchange_data.spot_eur
                yield_foreign = getattr(exchange_data, f"eur_{get_yield_period(days_diff)[0]}m")
            else:
                debug_logs.append(f"Unsupported currency: {currency}")
                return jsonify({"message": "Unsupported currency", "debug": debug_logs}), 400
            yield_domestic = getattr(exchange_data, f"tnd_{get_yield_period(days_diff)[0]}m")
            computed_forward = calculate_forward_rate(spot_rate, yield_foreign, yield_domestic, days_diff)
            debug_logs.append(f"Computed forward rate: {computed_forward}")

        # 3) Determine the effective strike (runs for BOTH admin_spot and else branches)
        if user_strike is not None:
            try:
                final_strike = float(user_strike)
                debug_logs.append(f"Using provided strike: {final_strike}")
            except Exception as e:
                debug_logs.append(f"Error parsing strike: {e}")
                return jsonify({"message": "Strike must be numeric", "debug": debug_logs}), 400
        else:
            # Canonical strike-from-spot calculation
            final_strike = strike_from_spot(
                spot_rate,
                yield_domestic,
                yield_foreign,
                T=days_diff / 365.0
            )
            debug_logs.append(f"Derived strike from spot: {final_strike}")



        # 4) Determine moneyness by comparing computed forward and strike
        tol = 0.01 * computed_forward
        if option_type == "CALL":
            if computed_forward > final_strike + tol:
                moneyness = "in the money"
            elif abs(computed_forward - final_strike) <= tol:
                moneyness = "at the money"
            else:
                moneyness = "out of the money"
        else:  # PUT
            if computed_forward < final_strike - tol:
                moneyness = "in the money"
            elif abs(computed_forward - final_strike) <= tol:
                moneyness = "at the money"
            else:
                moneyness = "out of the money"
        debug_logs.append(f"Final strike: {final_strike}, Moneyness: {moneyness}")

        # 5) Canonical premium and Greeks calculation

        T = days_diff / 365.0
        try:
            sigma = float(data.get("volatility_input"))
        except Exception as e:
            debug_logs.append(f"Missing or invalid volatility_input: {e}")
            return jsonify({"message": "Volatility input required for option pricing", "debug": debug_logs}), 400

        rev = reverse_premium_from_vol(
            option_type=option_type.lower(),
            S=spot_rate,
            K=final_strike,
            T=T,
            rd=yield_domestic,
            rf=yield_foreign,
            sigma=sigma
        )

        computed_premium = amount * rev["Non_Annualized_%_Premium"]
        debug_logs.append(f"Computed premium from vol_pricer: {computed_premium}")
        debug_logs.append(f"Annualized % Premium: {rev['Annualized_%_Premium']}")
        debug_logs.append(f"Non-Annualized % Premium: {rev['Non_Annualized_%_Premium']}")
        debug_logs.append(f"Unit Price: {rev['Unit_Price']}")
        debug_logs.append(f"Implied Volatility Input: {sigma}")



    unique_id = str(uuid.uuid4())
    debug_logs.append(f"Generated unique order ID: {unique_id}")

    try:
        new_order = Order(
            id_unique=unique_id,
            user=user,
            transaction_type=transaction_type,
            trade_type=trade_type,      
            amount=amount,
            original_amount=amount,
            currency=currency,
            value_date=value_date,
            transaction_date=datetime.now(),
            order_date=datetime.now(),
            bank_account=bank_account,
            reference=data.get('reference', f'REF-{unique_id}'),
            status=status,
            rating=user.rating,
            premium=computed_premium,
            is_option=is_option,
            option_type=option_type if is_option else None,
            strike=final_strike if is_option else None,
            forward_rate=computed_forward if is_option else None,
            moneyness=moneyness
        )
        debug_logs.append(f"Order/Option object created: {new_order}")
    except Exception as e:
        debug_logs.append(f"Error creating Order object: {str(e)}")
        return jsonify({"message": "Error creating Order object", "debug": debug_logs}), 500

    try:
        db.session.add(new_order)
        db.session.commit()
        debug_logs.append("Order/Option saved to the database")
    except Exception as e:
        db.session.rollback()
        debug_logs.append(f"Database error: {str(e)}")
        return jsonify({"message": "Database error", "debug": debug_logs}), 500

    try:
        log = AuditLog(
            action_type='create',
            table_name='order',
            record_id=new_order.id_unique,
            user_id=user_id,
            details=json.dumps({
                "id": new_order.id_unique,
                "transaction_type": transaction_type,
                "amount": amount,
                "currency": currency,
                "value_date": value_date_str,
                "bank_account": bank_account,
                "is_option": is_option,
                "option_type": option_type,
                "strike": final_strike,
                "premium": computed_premium,
                "moneyness": moneyness,
            })
        )
        db.session.add(log)
        db.session.commit()
        debug_logs.append("Audit log saved to the database")
    except Exception as e:
        db.session.rollback()
        debug_logs.append(f"Error saving audit log: {str(e)}")
        return jsonify({"message": "Error saving audit log", "debug": debug_logs}), 500

    return jsonify({
        "message": "Order/Option submitted successfully",
        "order_id": new_order.id_unique,
        "premium": computed_premium,
        "moneyness": moneyness,
        "forward_rate": computed_forward,
        "strike": final_strike,
        "implied_volatility": sigma,
        "annualized_pct_premium": rev["Annualized_%_Premium"],
        "non_annualized_pct_premium": rev["Non_Annualized_%_Premium"],
        "unit_price": rev["Unit_Price"],
        "debug": debug_logs
    }), 201


def require_reference_if_needed(user: User, payload: dict):
    """
    Raise ValueError when client.needs_references is True
    and no non-empty 'reference' was provided.
    """
    if user.needs_references and not payload.get("reference", "").strip():
        raise ValueError("Référence obligatoire pour ce client")


# ============================
# View Orders Endpoint (for clients)
# =========================
@user_bp.route('/orders', methods=['GET'])
@jwt_required()
def view_orders():
    user_id = get_jwt_identity() 
    orders = Order.query.filter_by(user_id=user_id, deleted=False).all()
    
    if not orders:
        return jsonify([]), 200  
    
    order_list = []
    for order in orders:
        order_list.append({
            "id": order.id,
            "transaction_type": order.transaction_type,
            "trade_type": order.trade_type, 
            "amount": order.original_amount,
            "currency": order.currency,
            "value_date": order.value_date.strftime("%Y-%m-%d"),
            "status": order.status,
            "client_name": order.user.client_name,
            "premium": order.premium,       
            "is_option": order.is_option, 
            "option_type": order.option_type,   
            "strike": order.strike,             
            "moneyness": order.moneyness ,
            "reference": order.reference,
       
        })
    
    return jsonify(order_list), 200


# =========================
# Utility: Log Action
# =========================
def log_action(action_type, table_name, record_id, user_id, details):
    user = User.query.get(user_id)
    details["client_name"] = user.client_name  
    log_entry = AuditLog(
        action_type=action_type,
        table_name=table_name,
        record_id=record_id,
        user_id=user_id,
        timestamp=datetime.now(),
        details=json.dumps(details)
    )
    db.session.add(log_entry)
    db.session.commit()

@user_bp.route('/orders/<int:order_id>', methods=['PUT'])
@jwt_required()
def update_order_user(order_id):
    """
    API for users to update their own orders.
    Logs old and new values of fields that were updated.
    """
    user_id = get_jwt_identity()  
    data = request.get_json()
            
    order = Order.query.filter_by(id=order_id, user_id=user_id, deleted=False).first()
    if not order:
        return jsonify({"error": "Order not found or you don't have permission to update this order"}), 404
    try:
        require_reference_if_needed(order.user, data)   
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    changes = {}

    if 'amount' in data and data['amount'] != order.amount:
        changes['original_amount'] = {"old": order.original_amount, "new": data['amount']}
        changes['amount'] = {"old": order.amount, "new": data['amount']}
        order.amount = data['amount']
        order.original_amount = data['amount']

    if 'currency' in data and data['currency'] != order.currency:
        changes['currency'] = {"old": order.currency, "new": data['currency']}
        order.currency = data['currency']

    if 'value_date' in data:
        new_value_date = datetime.strptime(data['value_date'], "%Y-%m-%d")
        if new_value_date != order.value_date:
            changes['value_date'] = {"old": order.value_date.strftime("%Y-%m-%d"), "new": data['value_date']}
            order.value_date = new_value_date

    if 'bank_account' in data and data['bank_account'] != order.bank_account:
        changes['bank_account'] = {"old": order.bank_account, "new": data['bank_account']}
        order.bank_account = data['bank_account']

    if 'reference' in data and data['reference'] != order.reference:
        changes['reference'] = {"old": order.reference, "new": data['reference']}
        order.reference = data['reference']

    if 'trade_type' in data and data['trade_type'] != order.trade_type:
        changes['trade_type'] = {"old": order.trade_type, "new": data['trade_type']}
        order.trade_type = data['trade_type']

    if order.is_option:
        if 'option_type' in data and data['option_type'] != order.option_type:
            changes['option_type'] = {"old": order.option_type, "new": data['option_type']}
            order.option_type = data['option_type']
        if 'strike' in data:
            try:
                new_strike = float(data['strike']) if data['strike'] is not None else None
            except Exception:
                return jsonify({"error": "Strike must be numeric"}), 400
            if new_strike != order.strike:
                changes['strike'] = {"old": order.strike, "new": new_strike}
                order.strike = new_strike

    log_action(
        action_type='update',
        table_name='order',
        record_id=order.id_unique,
        user_id=user_id,
        details=changes
    )

    db.session.commit()

    return jsonify({"message": "Order updated successfully"}), 200


@user_bp.route('/orders/<int:order_id>', methods=['DELETE'])
@jwt_required()
def delete_order_user(order_id):
    """
    API for users to soft-delete their own orders.
    Logs the delete action in the AuditLog.
    """
    user_id = get_jwt_identity()  
    order = Order.query.filter_by(id=order_id, user_id=user_id, deleted=False).first()
    if not order:
        return jsonify({"error": "Order not found or you don't have permission to delete this order"}), 404
    order.deleted = True
    log_action(
        action_type='delete',
        table_name='order',
        record_id=order.id_unique,
        user_id=user_id,
        details={"status": "deleted"}  
    )
    db.session.commit()

    return jsonify({"message": "Order deleted successfully"}), 200

#=======batch order upload 

#============================live rates and job registration=========
@user_bp.route('/live-rates', methods=['GET'])
def get_live_rates():
    currency = request.args.get('currency', 'USD')  
    if currency not in rates:
        return jsonify({'error': 'Unsupported currency pair'}), 400
    return jsonify({
        'currency': currency,
        'rates': rates[currency],
        'lastUpdated': lastUpdated,
        'metrics': metric[currency],
        'rates_all': rates_all
    })


def init_socketio(app):
    global socketio
    socketio = SocketIO(app, cors_allowed_origins="*")
    return socketio

def calculate_forward_rate(spot_rate, yield_foreign, yield_domestic, days):
    return spot_rate * ((1 + yield_domestic  * days / 360) / (1 + yield_foreign * days / 360))

var_table = {
    'USD': {
        '1m': {'1%': -0.038173, '5%': -0.026578, '10%': -0.020902},
        '3m': {'1%': -0.081835, '5%': -0.062929, '10%': -0.048737},
        '6m': {'1%': -0.200238, '5%': -0.194159, '10%': -0.186580}
    },
    'EUR': {
        '1m': {'1%': -0.188726, '5%': -0.176585, '10%': -0.160856},
        '3m': {'1%': -0.187569, '5%': -0.180371, '10%': -0.174856},
        '6m': {'1%': -0.199737, '5%': -0.192892, '10%': -0.185136}
    }
}

def get_yield_period(days):
    if days <= 60:
        return '1m'
    elif days <= 120:
        return '3m'
    else:
        return '6m'

def calculate_var(currency, days, amount):
    period = get_yield_period(days)
    currency_var = var_table.get(currency.upper(), {}).get(period, {})
    var_1 = currency_var.get('1%', 0.0) * abs(amount)
    var_5 = currency_var.get('5%', 0.0) * abs(amount)
    var_10 = currency_var.get('10%', 0.0) * abs(amount)
    return {'1%': var_1, '5%': var_5, '10%': var_10}

# API to calculate VaR for each order
@user_bp.route('/api/calculate-var', methods=['GET'])
def calculate_var_openpositions_api():
    try:
        open_positions = pd.read_sql('SELECT * FROM open_position', db.engine)
        today = datetime.today().date()
        var_calculations = []
        for _, pos in open_positions.iterrows():
            currency = pos['currency']
            amount = abs(pos['amount'])
            open_pos_date = pd.to_datetime(pos['value_date']).date()
            days_diff = (today - open_pos_date).days
            var_values = calculate_var(currency, days_diff, amount)
            var_calculations.append({
                "Value Date": open_pos_date.isoformat(),
                "Days": days_diff,
                "VaR 1%": var_values['1%'],
                "VaR 5%": var_values['5%'],
                "VaR 10%": var_values['10%']
            })

        return jsonify(var_calculations), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@user_bp.route('/api/calculate-forward-rate', methods=['GET'])
def calculate_forward_rate_api():
    try:
        df = pd.read_sql('SELECT * FROM exchange_data', db.engine)
        open_positions = pd.read_sql('SELECT * FROM open_position', db.engine)
        today = datetime.today().date()
        today_data = df[df['Date'] == today]
        forward_rates = []
        if today_data.empty:
            return jsonify({"error": "No exchange data found for today's date"}), 404

        for _, pos in open_positions.iterrows():
            currency = pos['currency']
            open_pos_date = pd.to_datetime(pos['value_date']).date()
            days_diff = (open_pos_date - today).days 
            try:
                spot_rate = today_data[f'Spot {currency.upper()}'].values[0]
                yield_foreign = today_data[f'{get_yield_period(days_diff).upper()} {currency.upper()}'].values[0]
                yield_domestic = today_data[f'{get_yield_period(days_diff).upper()} TND'].values[0]
            except KeyError as e:
                print(f"Missing required field in exchange data: {str(e)}")
                continue
            forward_rate_value = calculate_forward_rate(spot_rate, yield_foreign, yield_domestic, days_diff)
            forward_rates.append({
                "open_position_id": int(pos["id"]), 
                "Value Date": open_pos_date.isoformat(),
                "Days": days_diff,
                "Forward Rate": forward_rate_value
            })

        return jsonify(forward_rates), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
   
@user_bp.route('/api/dashboard/summary', methods=['GET'])
@jwt_required()
def dashboard_summary():
    from datetime import date
    from collections import defaultdict
    import calendar
    user_id  = get_jwt_identity()
    currency = request.args.get('currency', 'USD').upper()
    orders = Order.query.filter(
    Order.user_id == user_id,
    Order.currency == currency,
    Order.status.in_(['Executed', 'Matched'])
).all()
    has_forward_or_option = any(is_hedged(o) for o in orders)  
    total_traded = sum(o.original_amount for o in orders)
    total_traded_tnd = sum(
        o.original_amount * o.execution_rate for o in orders if o.execution_rate
    )
    total_covered = sum(
    o.original_amount
    for o in orders
    if is_hedged(o)         
)
    coverage_percent = (total_covered / total_traded * 100) if total_traded else 0
    # ---------- Month‑to‑date (MTD) traded ---------------------
    today = date.today()
    total_traded_mtd = sum(
        o.original_amount for o in orders
        if o.transaction_date.year == today.year and o.transaction_date.month == today.month
    )
    total_traded_mtd_tnd = sum(
        o.original_amount * o.execution_rate for o in orders
        if o.execution_rate and o.transaction_date.year == today.year and o.transaction_date.month == today.month
    )
    # ---------- Gains & Commissions ----------------------------------------
    economies_totales = 0.0         # cumulative gain in foreign currency
    economies_totales_tnd = 0.0     # cumulative gain converted to TND
    economies_totales_couv = 0.0    # hedged cumulative gain in foreign currency
    economies_totales_couv_tnd = 0.0  # hedged cumulative gain in TND
    total_commissions_tnd = 0.0     # total commission in TND
    for o in orders:
        if o.execution_rate is None:
            continue

        bench = calculate_benchmark(o)
        if o.transaction_type.lower() in ("import", "buy"):
            gain_pct = (bench - o.execution_rate)/o.execution_rate
        else:
            gain_pct = (o.execution_rate - bench) / bench
        gain_fx  = gain_pct * o.original_amount
        gain_tnd = gain_fx * o.execution_rate
        economies_totales     += gain_fx
        economies_totales_tnd += gain_tnd
        if is_hedged(o):
            economies_totales_couv     += gain_fx
            economies_totales_couv_tnd += gain_tnd
        comm_pct = o.commission_percent or 0.0
        total_commissions_tnd += o.execution_rate * o.original_amount * comm_pct
    # ---------- Net Gain & ROI ---------------------------------------------
    net_gain_tnd = economies_totales_tnd - total_commissions_tnd
    roi_percent  = (net_gain_tnd / total_commissions_tnd * 100) if total_commissions_tnd else None
    # ---------- Super‑performance Rate -------------------------------------
    superformance_rate = calculate_superformance_rate(orders)
    # ---------- Monthly Series (for charts) ----------------------------
    monthly = defaultdict(lambda: {"transacted": 0.0, "gain": 0.0})
    for o in orders:
        # Group by transaction month/year
        key = f"{calendar.month_name[o.transaction_date.month]} {o.transaction_date.year}"
        monthly[key]["transacted"] += o.original_amount * (o.execution_rate or 0)
        if o.execution_rate:
            bench = calculate_benchmark(o)
            if o.transaction_type.lower() in ("import", "buy"):
                order_gain_pct = (bench - o.execution_rate)/o.execution_rate
                
            else:
                order_gain_pct = (o.execution_rate- bench)  / bench
            monthly[key]["gain"] += order_gain_pct * o.original_amount * (o.execution_rate or 0)
    months = list(monthly.keys())
    monthly_total_transacted = [v["transacted"] for v in monthly.values()]
    monthly_total_gain = [v["gain"] for v in monthly.values()]
    # ---------- MTD Gains & Commissions ------------------------------------
    economies_totales_tnd_mtd = 0.0
    for o in orders:
        if (o.transaction_date.year == today.year and o.transaction_date.month == today.month and o.execution_rate):
            bench = calculate_benchmark(o)
            if o.transaction_type.lower() == "import":
                gain_pct = (bench - o.execution_rate) / o.execution_rate
            else:
                gain_pct = (o.execution_rate - bench) / bench
            gain_fx  = gain_pct * o.original_amount
            gain_tnd = gain_fx * o.execution_rate
            economies_totales_tnd_mtd += gain_tnd

    total_commissions_tnd_mtd = sum(
        o.execution_rate * o.original_amount * (o.commission_percent or 0.0)
        for o in orders
        if o.execution_rate and o.transaction_date.year == today.year and o.transaction_date.month == today.month
    )
    net_gain_tnd_mtd = economies_totales_tnd_mtd - total_commissions_tnd_mtd
    roi_percent_mtd = (net_gain_tnd_mtd / total_commissions_tnd_mtd * 100) if total_commissions_tnd_mtd else None

    return jsonify({
        "currency": currency,
        "total_traded_fx": total_traded,             
        "total_traded_tnd": total_traded_tnd,          
        "total_traded_mtd_fx": total_traded_mtd,
        "total_traded_mtd_tnd": total_traded_mtd_tnd,
        "coverage_percent": coverage_percent,
        "economies_totales_fx": economies_totales,   
        "economies_totales_tnd": economies_totales_tnd,  
        "economies_totales_couverture_fx": economies_totales_couv,
        "economies_totales_couverture_tnd": economies_totales_couv_tnd,
        "total_commissions_tnd": total_commissions_tnd,
        "net_gain_tnd": net_gain_tnd,
        "roi_percent": roi_percent,
        "superformance_rate": superformance_rate,       
        "months": months,
        "monthlyTotalTransacted": monthly_total_transacted,
        "monthlyTotalGain": monthly_total_gain,        
        "total_commissions_tnd_mtd": total_commissions_tnd_mtd,
        "net_gain_tnd_mtd": net_gain_tnd_mtd,
        "roi_percent_mtd": roi_percent_mtd,
        "has_forward_or_option": has_forward_or_option,
        "total_covered_mtd_fx": sum(
            o.original_amount for o in orders
            if is_hedged(o) and o.transaction_date.year == today.year and o.transaction_date.month == today.month
        ),
        "total_covered_mtd_tnd": sum(
            o.original_amount * o.execution_rate for o in orders
            if is_hedged(o) and o.transaction_date.year == today.year and o.transaction_date.month == today.month and o.execution_rate
        ),
    })

@user_bp.route('/api/dashboard/secured-vs-market-forward-rate', methods=['GET'])
@jwt_required()
def forward_rate_table():
    user_id = get_jwt_identity()
    currency = request.args.get("currency", "USD").upper()
    orders = Order.query.filter(
    Order.user_id == user_id,
    Order.currency == currency,
    Order.status.in_(['Executed', 'Matched'])  
).all()

    forward_rate_data = []
    for order in orders:
        if not is_hedged(order):
            continue
        secured_forward_rate = order.execution_rate
        benchmark_rate = calculate_benchmark(order)
        forward_rate_data.append({
            "transaction_date": order.transaction_date.strftime('%Y-%m-%d'),
            "value_date": order.value_date.strftime('%Y-%m-%d'),
            "secured_forward_rate_export": secured_forward_rate if order.transaction_type in ["export", "sell"] else None,
            "secured_forward_rate_import": secured_forward_rate if order.transaction_type in ["import", "buy"] else None,
            "market_forward_rate_export": benchmark_rate if order.transaction_type in ["export", "sell"] else None,
            "market_forward_rate_import": benchmark_rate if order.transaction_type in ["import", "buy"] else None,
        })

    return jsonify(forward_rate_data)


@user_bp.route('/api/dashboard/superperformance-trend', methods=['GET'])
@jwt_required() 
def superperformance_trend():
    user_id = get_jwt_identity()
    currency = request.args.get("currency", "USD").upper()

    orders = Order.query.filter(
        Order.user_id == user_id,
        Order.currency == currency,
        Order.status.in_(['Executed', 'Matched'])
    ).order_by(Order.transaction_date).all()

    if not orders:
        return jsonify({"message": "No data available for this user"}), 200
    trend_data = []
    for order in orders:
        trend_data.append({
            "date": order.transaction_date.strftime('%Y-%m-%d'),
            "execution_rate_export": order.execution_rate if order.transaction_type.lower() in ["export", "sell"] else None,
            "execution_rate_import": order.execution_rate if order.transaction_type.lower() in ["import", "buy"] else None,
            "interbank_rate": order.interbank_rate  
        })

    return jsonify(trend_data), 200

from collections import defaultdict
from typing import Iterable, Literal

def calculate_superformance_rate(
    orders: Iterable,
    direction: Literal["buy", "sell"] = "buy",
) -> float:
    """
    Superformance percentage per trading day
    A “superformance day” is one where at least one order satisfies:
      • BUY  (import): execution_rate <= interbank_rate  (lower cost is better)
      • SELL (export): execution_rate >= interbank_rate  (higher price is better)

       """
    if direction not in {"buy", "sell"}:
        raise ValueError("direction must be 'buy' or 'sell'")

    # 1. Filter by transaction type
    if direction == "buy":
        valid_types = {"import", "buy"}
        cmp = lambda exe, inter: exe <= inter      # lower is better
    else:  # direction == "sell"
        valid_types = {"export", "sell"}
        cmp = lambda exe, inter: exe >= inter      # higher is better

    filtered = [o for o in orders if o.transaction_type.lower() in valid_types]

    # 2. Group by calendar day
    groups = defaultdict(list)
    for o in filtered:
        groups[o.transaction_date].append(o)

    # 3. Count “wins” per day
    superformance_days = 0
    for day_orders in groups.values():
        valid = [
            o for o in day_orders
            if o.execution_rate is not None and o.interbank_rate is not None
        ]
        if valid and any(cmp(o.execution_rate, o.interbank_rate) for o in valid):
            superformance_days += 1

    # 4. Percentage
    total_days = len(groups)
    return (superformance_days / total_days * 100.0) if total_days else 0.0

def compute_bank_gains(user_id, currency="USD"):
    from collections import defaultdict
    import math

    orders = (
        Order.query
        .filter(Order.user_id == user_id,
                Order.currency == currency,
                Order.status.in_(["Executed", "Matched"]))
        .order_by(Order.transaction_date)
        .all()
    )

    daily = defaultdict(lambda: {"gain": 0.0, "comm": 0.0})
    cache = []

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

        day = o.transaction_date
        daily[day]["gain"] += gain_tnd
        daily[day]["comm"] += comm_tnd

        cache.append((o, bench, gain_pct, gain_tnd, comm_tnd))

    rows = []
    for o, bench, gain_pct, gain_tnd, comm_tnd in cache:
        stats = daily[o.transaction_date]
        ratio = float(stats["comm"]) / abs(float(stats["gain"])) * 100 if stats["gain"] else 0.0

        o.commission_gain = ratio
        db.session.add(o)

        def thousands(x):  return f"{x:,.2f}".replace(",", " ").replace(".", ",")
        fmt_eur  = lambda x: f" {thousands(x)}"
        fmt_tnd  = lambda x: f"{thousands(x)} TND"
        fmt_rate = lambda r: "" if (r is None or math.isnan(r)) else f"{r:,.4f}".replace(",", " ").replace(".", ",")
        fmt_pct  = lambda p: f"{p:.2f}%"

        rows.append({
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
            "Commission CC ***": fmt_tnd(comm_tnd),
            "Commission Percent": fmt_pct((o.commission_percent or 0.0) * 100),
            "Commission % de gain": fmt_pct(ratio)
        })

    db.session.commit()
    return rows

@user_bp.route('/api/dashboard/bank-gains', methods=['GET'])
@jwt_required()
def bank_gains():
    uid = get_jwt_identity()
    currency = request.args.get("currency", "USD").upper()
    rows = compute_bank_gains(uid, currency)
    return jsonify(rows), 200

def is_hedged(order) -> bool:
    """A deal is hedged whenever it is *not* a spot."""
    return (order.trade_type or "").lower() in ("forward", "option")


def calculate_benchmark(order):
     # ------------------------------------------------------------------
    # 1) Interbank spot on the trade date
    # ------------------------------------------------------------------
    ib_rate = order.interbank_rate or get_interbank_rate_from_db(
        order.transaction_date, order.currency
    )
    if ib_rate is None:
        raise ValueError(
            f"Aucun taux interbancaire pour {order.currency} "
            f"le {order.transaction_date}"
        )
    # ------------------------------------------------------------------
    # 2) Historical‑loss factor
    # ------------------------------------------------------------------
    hl = getattr(order, "historical_loss", None)
    if hl is None:
        raise ValueError("Historical loss manquant : impossible de calculer le benchmark")

    side = order.transaction_type.lower()
    if side in ("import", "buy"):
        base_bmk = ib_rate * (1 + hl)
    elif side in ("export", "sell"):
        base_bmk = ib_rate * (1 - hl)
    else:
        raise ValueError(f"Type de transaction non supporté : {order.transaction_type}")
    # ------------------------------------------------------------------
    # 3) Forward adjustment *only* for forward & option trades
    # ------------------------------------------------------------------
    if order.trade_type in ("forward", "option"):
        # days = (order.value_date - order.transaction_date).days
        days = (order.value_date - order.transaction_date).days - 2
        period = get_yield_period(days)                    # '1m' | '3m' | '6m'
        p_key  = {"1m": "1M", "3m": "3M", "6m": "6M"}[period]

        ed = pd.read_sql(
            'SELECT * FROM exchange_data WHERE "Date" = %(d)s',
            db.engine,
            params={"d": order.transaction_date.strftime("%Y-%m-%d")}
        )
        if ed.empty:
            raise ValueError(f"Aucune donnée de taux pour {order.transaction_date}")

        if order.currency.upper() == "USD":
            y_foreign = ed[f"{p_key} USD"].iloc[0]
        elif order.currency.upper() == "EUR":
            y_foreign = ed[f"{p_key} EUR"].iloc[0]
        else:
            raise ValueError("Devise non supportée pour les yields")

        y_domestic = ed[f"{p_key} TND"].iloc[0]
        return calculate_forward_rate(base_bmk, y_foreign, y_domestic, days)
    return base_bmk


from flask import current_app
def get_interbank_rate_from_db(the_date, currency):
    """
    1. Try exact date.
    2. Try the latest earlier date.
    3. If still None, fetch from BCT website and insert on-the-fly.
    """
    rate_rec = (
        InterbankRate.query
        .filter_by(date=the_date, currency=currency)
        .first()
    )
    if not rate_rec:
        rate_rec = (
            InterbankRate.query
            .filter(
                InterbankRate.date < the_date,
                InterbankRate.currency == currency
            )
            .order_by(InterbankRate.date.desc())
            .first()
        )

    if rate_rec:                     
        return rate_rec.rate

    # ---------- 3 – on-the-fly fetch & insert ----------
    fresh = fetch_rate_for_date_and_currency(the_date, currency)
    if fresh:                         # got a real number from BCT
        rate_rec = InterbankRate(date=the_date, currency=currency, rate=fresh)
        db.session.add(rate_rec)
        db.session.commit()
        return fresh                  # use it immediately
    return None

# Helper: Fetch interbank rate from external source (e.g., BCT website)
def fetch_rate_for_date_and_currency(date, currency):
    formatted_date = date.strftime('%Y-%m-%d')
    response = requests.post(f"https://www.bct.gov.tn/bct/siteprod/cours_archiv.jsp?input={formatted_date}&langue=en")
    soup = BeautifulSoup(response.content, 'html.parser')
    rate = None
    rows = soup.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if cells and cells[1].get_text(strip=True).lower() == currency.lower():
            rate = float(cells[3].get_text(strip=True).replace(',', '.'))
            break
    return rate

def update_interbank_rates_db_logic(start_date_str="2020-01-01"):
    from datetime import datetime, timedelta
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid start_date format, expected YYYY-MM-DD")
        return {"error": "Invalid start_date format, expected YYYY-MM-DD"}

    end_date = datetime.today().date()
    currencies = ["USD", "EUR"]  
    updated_entries = []

    for n in range((end_date - start_date).days + 1):
        current_date = start_date + timedelta(days=n)
        for currency in currencies:
            if not InterbankRate.query.filter_by(date=current_date, currency=currency).first():
                rate = fetch_rate_for_date_and_currency(current_date, currency)
                if rate:
                    new_rate = InterbankRate(date=current_date, currency=currency, rate=rate)
                    db.session.add(new_rate)
                    updated_entries.append({
                        "date": current_date.isoformat(),
                        "currency": currency,
                        "rate": rate
                    })
    try:
        db.session.commit()
        print("Interbank rates DB updated successfully", updated_entries)
        return {"message": "Interbank rates DB updated successfully", "updated": updated_entries}
    except Exception as e:
        db.session.rollback()
        print("Error updating interbank rates DB:", e)
        return {"error": str(e)}

@user_bp.route('/update-interbank-rates-db', methods=['POST'])
def update_interbank_rates_db_endpoint():
    data = request.get_json() or {}
    start_date_str = data.get("start_date", "2020-08-01")
    result = update_interbank_rates_db_logic(start_date_str)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result), 200

@user_bp.route('/update-interbank-rates', methods=['POST'])
def update_interbank_rates():
    try:
        update_order_interbank_and_benchmark_rates(current_app)
        return jsonify({'message': 'Orders updated with interbank & benchmark rates successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def update_order_interbank_and_benchmark_rates(app):
    with app.app_context():
        try:
            orders = Order.query.all()
            for order in orders:
                if order.interbank_rate is None:
                    rate = get_interbank_rate_from_db(order.transaction_date, order.currency)
                    if rate:
                        order.interbank_rate = float(rate)
                try:
                    benchmark = calculate_benchmark(order)
                    order.benchmark_rate = float(benchmark)
                except Exception as ex:
                    print(f"Error calculating benchmark for Order ID {order.id}: {ex}")

                db.session.add(order)

            db.session.commit()
            print("Orders successfully updated")

        except Exception as e:
            db.session.rollback()
            print(f"Error updating orders: {e}")

def calculate_time_to_maturity(trade_date, echeance):
    """
    Calculates time to maturity (in years) given the trade date and the value (or maturity) date.
    Both dates should be strings in the format '%d/%m/%Y'.
    """
    trade_date = pd.to_datetime(trade_date, format='%d/%m/%Y')
    echeance = pd.to_datetime(echeance, format='%d/%m/%Y')
    return (echeance - trade_date).days / 365

def interpolate_prime(time_to_maturity, known_times, known_primes):
    """
    Interpolates the premium percentage based on the target time_to_maturity.
    
    Parameters:
      - time_to_maturity: The calculated time (in years) for the option.
      - known_times: A NumPy array of times (in years) from your PremiumRate model.
      - known_primes: A NumPy array of premium percentages corresponding to the known times.
    
    Returns:
      A dictionary with values computed using different interpolation methods.
    """
    sorted_indices = np.argsort(known_times)
    known_times = known_times[sorted_indices]
    known_primes = known_primes[sorted_indices]

    linear_interp = interp1d(known_times, known_primes, kind='linear', fill_value='extrapolate')
    quadratic_interp = interp1d(known_times, known_primes, kind='quadratic', fill_value='extrapolate')
    cubic_spline = CubicSpline(known_times, known_primes, extrapolate=True)
    
    return {
        'Linear': linear_interp(time_to_maturity),
        'Quadratic': quadratic_interp(time_to_maturity),
        'CubicSpline': cubic_spline(time_to_maturity),
    }


@user_bp.route('/orders/preview', methods=['POST'])
@jwt_required()
def preview_option():
    """
    Preview an option order by computing the forward rate, premium, moneyness, and default strike.
    This endpoint does NOT create an order.
    """
    data = request.get_json()
    if not data:
        return jsonify({"message": "No data provided"}), 400
    try:
        amount = float(data.get("amount", 0))
        transaction_type = data.get("transaction_type")
        value_date_str = data.get("value_date")
        currency = data.get("currency")
        bank_account = data.get("bank_account")
        is_option = data.get("is_option", False)
        option_type = data.get("option_type", "").upper()
        strike = data.get("strike")
    except Exception as e:
        return jsonify({"message": f"Invalid data format: {e}"}), 400

    if not (transaction_type and value_date_str and currency and bank_account):
        return jsonify({"message": "Missing required fields"}), 400

    try:
        value_date = datetime.strptime(value_date_str, "%Y-%m-%d")
    except Exception as e:
        return jsonify({"message": "Invalid date format"}), 400

    if not is_option:
        return jsonify({"message": "Preview is only available for options"}), 400

    today = datetime.today().date()
    exchange_data = ExchangeData.query.filter_by(date=today).first()
    if not exchange_data:
        return jsonify({"message": "Exchange data for today not available"}), 400

    days_diff = (value_date.date() - today).days
    period = get_yield_period(days_diff)

    admin_spot = data.get("spot")
    if admin_spot is not None:
        try:
            spot_rate = float(admin_spot)
        except Exception:
            return jsonify({"message": "Spot must be numeric"}), 400
        if currency.upper() == 'USD':
            yield_foreign = getattr(exchange_data, f"usd_{period[0]}m")
        elif currency.upper() == 'EUR':
            yield_foreign = getattr(exchange_data, f"eur_{period[0]}m")
        else:
            return jsonify({"message": "Unsupported currency"}), 400
        yield_domestic = getattr(exchange_data, f"tnd_{period[0]}m")
        computed_forward = calculate_forward_rate(spot_rate, yield_foreign, yield_domestic, days_diff)
    else:
        if currency.upper() == 'USD':
            spot_rate = exchange_data.spot_usd
            yield_foreign = getattr(exchange_data, f"usd_{period[0]}m")
        elif currency.upper() == 'EUR':
            spot_rate = exchange_data.spot_eur
            yield_foreign = getattr(exchange_data, f"eur_{period[0]}m")
        else:
            return jsonify({"message": "Unsupported currency"}), 400
        yield_domestic = getattr(exchange_data, f"tnd_{period[0]}m")
        computed_forward = calculate_forward_rate(spot_rate, yield_foreign, yield_domestic, days_diff)

    # Strike-from-spot rule
    if strike is not None:
        try:
            strike_value = float(strike)
        except Exception:
            return jsonify({"message": "Strike must be numeric"}), 400
    else:
        strike_value = strike_from_spot(
            spot_rate,
            yield_domestic,
            yield_foreign,
            T=days_diff / 365.0
        )

    # Forward = strike for ATM forward pricing
    computed_forward = strike_value

    # Recalculate tolerance now that forward is final
    tol = 0.01 * computed_forward

    # Determine moneyness
    if option_type == "CALL":
        if computed_forward > strike_value + tol:
            moneyness = "in the money"
        elif abs(computed_forward - strike_value) <= tol:
            moneyness = "at the money"
        else:
            moneyness = "out of the money"
    elif option_type == "PUT":
        if computed_forward < strike_value - tol:
            moneyness = "in the money"
        elif abs(computed_forward - strike_value) <= tol:
            moneyness = "at the money"
        else:
            moneyness = "out of the money"
    else:
        return jsonify({"message": "Option type must be CALL or PUT"}), 400

    # Time to maturity
    T = days_diff / 365.0

    try:
        sigma = float(data.get("volatility_input"))
    except Exception as e:
        return jsonify({"message": f"Missing or invalid volatility_input: {e}"}), 400

    # Premium calculation
    rev = reverse_premium_from_vol(
        option_type=option_type.lower(),
        S=spot_rate,
        K=strike_value,
        T=T,
        rd=yield_domestic,
        rf=yield_foreign,
        sigma=sigma
    )

    computed_premium = amount * rev["Non_Annualized_%_Premium"]

    # Greeks calculation
    delta, gamma, theta = garman_kohlhagen_greeks(
        option_type.lower(),
        spot_rate,
        strike_value,
        T,
        yield_domestic,
        yield_foreign,
        sigma
    )

    return jsonify({
        "forward_rate": computed_forward,
        "strike": strike_value,
        "moneyness": moneyness,
        "premium": computed_premium,
        "annualized_pct_premium": rev["Annualized_%_Premium"],
        "non_annualized_pct_premium": rev["Non_Annualized_%_Premium"],
        "unit_price": rev["Unit_Price"],
        "implied_volatility": sigma,
        "greeks": {
            "delta": delta,
            "gamma": gamma,
            "theta": theta
        }
    }), 200

@user_bp.route('/get-interbank-rates', methods=['GET'])
def get_interbank_rates():
    try:
        rates = InterbankRate.query.all()
        result = [{
            "date": rate.date.strftime("%Y-%m-%d"),
            "currency": rate.currency,
            "rate": rate.rate
        } for rate in rates]
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ─── convenience flag ─────────────────────────────────────────
@property
def is_admin(self) -> bool:
    """True if the user owns a role named “admin” (case-insensitive)."""
    return any(r.name and r.name.lower() == "admin" for r in self.roles)


# ------------------  Downloads A)  open positions  ---------------------------
@user_bp.get("/download-open-positions-template")
@jwt_required(optional=True)         # template is public-ish; no hard auth
def dl_open_positions_tpl():
    cols = ["Value Date", "Currency", "Amount", "Transaction Type"]
    sample = {
        "Value Date": "2024/09/30",
        "Currency": "USD",
        "Amount": 500000,
        "Transaction Type": "buy",      # or "sell"
    }
    return _make_excel(cols, sample)

# ------------------  B)  spot / forward / option orders ------------
@user_bp.get("/download-orders-template")
@jwt_required(optional=True)
def dl_orders_tpl():
    cols = [
        "Transaction Date", "Value Date", "Currency", "Transaction Type",
        "Amount", "Bank Account",            # ← your extra client fields
        "Trade Type",                        # spot / forward / option
        "Option Type", "Strike"              # keep blank for spots/forwards
    ]
    sample = {
        "Transaction Date": "2024/09/25",
        "Value Date":       "2024/10/02",
        "Currency":         "EUR",
        "Transaction Type": "buy",
        "Amount":           250000,
        "Bank Account":     "BE12 3456 7890 1234",
        "Trade Type":       "option",
        "Option Type":      "CALL",
        "Strike":           3.2150,
    }
    return _make_excel(cols, sample)

# ------------------  C)  TCA spot-input upload ---------------------
@user_bp.get("/download-tca-inputs-template")
@jwt_required(optional=True)
def dl_tca_spot_tpl():
    cols = [
        "transaction_date", "value_date", "currency",
        "amount", "execution_rate", "transaction_type"
    ]
    sample = {
        "transaction_date": "2024/09/20",
        "value_date":       "2024/09/22",
        "currency":         "USD",
        "amount":           1000000,
        "execution_rate":   3.1450,
        "transaction_type": "import",      # import / export
    }
    return _make_excel(cols, sample)
@user_bp.route('/upload-orders', methods=['POST'])
@jwt_required()
def upload_orders():
    """
    Bulk-upload Spot / Forward / Option orders for the **logged-in client**.
    Excel columns (case-insensitive):
        Transaction Date | Value Date | Currency | Transaction Type
        Amount | Bank Account | Trade Type | Option Type | Strike | Reference
    Columns that NO LONGER appear in the sheet (auto-filled server-side):
        Execution rate, Interbancaire, Historical Loss, Commission %
    """
    # 0) sanity check -------------------------------------------------
    if 'file' not in request.files or not allowed_file(request.files['file'].filename):
        return jsonify({'error': 'Invalid file format (.xls / .xlsx expected)'}), 400

    file = request.files['file']
    user_id = get_jwt_identity()
    user    = User.query.get(user_id)

    try:
        # 1) read Excel ------------------------------------------------
        df = pd.read_excel(BytesIO(file.read()))

        required = {
            "Transaction Date", "Value Date", "Currency",
            "Transaction Type", "Amount", "Bank Account", "Trade Type"
        }
        missing = required - {c.strip() for c in df.columns}
        if missing:
            return jsonify({'error': f'Missing columns: {sorted(missing)}'}), 400

        # optional columns
        if "Reference" not in df.columns:   df["Reference"]   = ""
        if "Option Type" not in df.columns: df["Option Type"] = ""
        if "Strike"      not in df.columns: df["Strike"]      = np.nan

        # 2) clean / coerce ------------------------------------------
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
        df["Value Date"]       = pd.to_datetime(df["Value Date"])
        df["Amount"]           = df["Amount"].replace(",", "", regex=True).astype(float)
        df["Trade Type"]       = df["Trade Type"].str.lower().str.strip()
        df["Transaction Type"] = df["Transaction Type"].str.lower().str.strip()

        # 3) iterate rows --------------------------------------------
        uploaded = 0
        for idx, row in df.iterrows():
            tx_date = row["Transaction Date"].date()
            val_date= row["Value Date"].date()
            currency= row["Currency"].upper()

            ib_rate = get_interbank_rate_from_db(tx_date, currency)
            if ib_rate is None:
                raise Exception(f"No interbank rate for {currency} on {tx_date}")

            new_order = Order(
                user                = user,
                transaction_type    = row["Transaction Type"],
                trade_type          = row["Trade Type"],
                amount              = row["Amount"],
                original_amount     = row["Amount"],
                currency            = currency,
                value_date          = val_date,
                transaction_date    = tx_date,
                order_date          = datetime.utcnow(),
                bank_account        = row["Bank Account"],
                reference           = row["Reference"] or None,
                option_type         = (row["Option Type"] or None).upper() if pd.notna(row["Option Type"]) else None,
                strike              = float(row["Strike"]) if pd.notna(row["Strike"]) else None,
                status              = "Pending" if row["Trade Type"] == "spot" else "Market",
                interbank_rate      = ib_rate
            )
            db.session.add(new_order)
            uploaded += 1

        db.session.commit()
        log_action('bulk_upload', 'order', -1, user_id,
                   {"uploaded_orders": uploaded})
        return jsonify({'message': f'{uploaded} orders uploaded'}), 200

    except Exception as exc:
        db.session.rollback()
        return jsonify({'error': str(exc)}), 500


@user_bp.route('/load-atm-vols', methods=['POST'])
@jwt_required()
def load_atm_vols_endpoint():
    """
    Load ATM volatility data from JSON file into the database.
    Expects a JSON payload with 'json_filename' parameter.
    """
    data = request.get_json()
    if not data or 'json_filename' not in data:
        return jsonify({'error': 'json_filename is required'}), 400
    
    json_filename = data['json_filename']
    json_path = f"data/{json_filename}"
    
    try:
        load_atm_vols_from_json(json_path)
        return jsonify({'message': f'ATM volatility data loaded successfully from {json_filename}'}), 200
    except Exception as e:
        return jsonify({'error': f'Error loading ATM vols: {str(e)}'}), 500


@user_bp.route('/api/volatility-from-premium', methods=['POST'])
@jwt_required()
def volatility_from_premium():
    from option_pricer.volatility_pricer import (
        get_db_connection,
        get_latest_exchange_data,
        strike_from_spot,
        implied_volatility,
        garman_kohlhagen_price,
        garman_kohlhagen_greeks,
    )

    data = request.get_json()
    required_fields = ["currency_pair", "option_type", "spot", "value_date", "premium_percentage"]
    for field in required_fields:
        if field not in data:
            return jsonify({"message": f"Missing required field: {field}"}), 400

    try:
        currency_pair = data["currency_pair"].upper().strip()
        option_type = data["option_type"].lower().strip()
        if option_type not in ("call", "put"):
            return jsonify({"message": "option_type must be 'CALL' or 'PUT'"}), 400
        if currency_pair not in ("EUR/TND", "USD/TND"):
            return jsonify({"message": "currency_pair must be 'EUR/TND' or 'USD/TND'"}), 400

        spot = float(str(data["spot"]).strip())
        strike = float(str(data.get("strike")).strip()) if data.get("strike") not in (None, "") else None

        val_date = datetime.today()
        maturity_date = datetime.strptime(str(data["value_date"]).strip(), "%Y-%m-%d")
        T = (maturity_date - val_date).days / 365.0
        if T <= 0:
            return jsonify({"message": "Maturity must be in the future"}), 400

        # Annualized % premium (e.g., 0.0315 for 3.15%)
        premium_input = float(str(data["premium_percentage"]).strip())

        conn = get_db_connection()
        rates = get_latest_exchange_data(conn, currency_pair)
        rd = float(np.interp(T, rates['known_tenors'], rates['domestic_yields']))
        rf = float(np.interp(T, rates['known_tenors'], rates['foreign_yields']))

        # Strike rule: use provided, else derive from spot
        if strike is None:
            strike = strike_from_spot(spot, rd, rf, T)

        # Solve for implied vol
        sigma = implied_volatility(
            option_type, S=spot, K=strike, T=T,
            rd=rd, rf=rf, annualized_premium=premium_input
        )
        if sigma is None or (isinstance(sigma, float) and not np.isfinite(sigma)):
            return jsonify({
                "message": "Could not compute implied volatility for given inputs",
                "details": {
                    "spot": spot, "strike": strike, "T_years": round(T, 6),
                    "rd": rd, "rf": rf, "premium_percentage": premium_input
                }
            }), 422

        theo_price = garman_kohlhagen_price(option_type, spot, strike, T, rd, rf, sigma)
        delta, gamma, theta = garman_kohlhagen_greeks(option_type, spot, strike, T, rd, rf, sigma)
        unit_price = theo_price
        annualized_pct_premium = unit_price / strike
        non_annualized_pct_premium = annualized_pct_premium * T

        # ATM-forward reporting convention (diagnostic only)
        forward_rate = strike

        return jsonify({
            "T_years": round(T, 6),
            "currency_pair": currency_pair,
            "spot": spot,
            "strike": strike,
            "forward_rate": forward_rate,
            "implied_volatility": sigma,
            "unit_price": unit_price,
            "theoretical_price": theo_price,
            "annualized_pct_premium": annualized_pct_premium,
            "non_annualized_pct_premium": non_annualized_pct_premium,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "rd": rd,
            "rf": rf
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

    finally:
        if 'conn' in locals():
            conn.close()


@user_bp.route('/api/implied-vol-from-db', methods=['POST'])
@jwt_required()
def implied_vol_from_db():
    from option_pricer.volatility_pricer import (
        get_db_connection,
        get_latest_exchange_data,
        strike_from_spot,
        implied_volatility,
        garman_kohlhagen_price,
        garman_kohlhagen_greeks,
    )
    from models import PremiumRate
    import numpy as np

    data = request.get_json()
    required = ["currency", "option_type", "transaction_type", "spot", "value_date"]
    for field in required:
        if field not in data:
            return jsonify({"message": f"Missing required field: {field}"}), 400

    try:
        currency = str(data["currency"]).upper().strip()            # "EUR" or "USD"
        option_type = str(data["option_type"]).upper().strip()      # "CALL"/"PUT"
        transaction_type = str(data["transaction_type"]).lower().strip()  # "buy"/"sell"
        spot = float(str(data["spot"]).strip())
        strike_in = data.get("strike")
        strike = float(strike_in) if (strike_in not in (None, "")) else None

        # Target maturity
        value_date = datetime.strptime(str(data["value_date"]).strip(), "%Y-%m-%d")
        today = datetime.today()
        T = (value_date - today).days / 365.0
        if T <= 0:
            return jsonify({"message": "Maturity must be in the future"}), 400

        # 1) Fetch calibration rows
        premium_rates = PremiumRate.query.filter_by(
            currency=currency,
            option_type=option_type,
            transaction_type=transaction_type
        ).all()
        if not premium_rates:
            return jsonify({"message": "No premium entries found for this config"}), 404

        # 2) Load yields once
        conn = get_db_connection()
        rates = get_latest_exchange_data(conn, f"{currency}/TND")
        known_tenors = rates['known_tenors']

        # 3) Compute sigma_i for each tenor from its premium %
        T_list, sigma_list = [], []
        for r in premium_rates:
            T_i = float(r.maturity_days) / 365.0
            if T_i <= 0:
                continue

            rd_i = float(np.interp(T_i, known_tenors, rates['domestic_yields']))
            rf_i = float(np.interp(T_i, known_tenors, rates['foreign_yields']))

            K_i = strike if strike is not None else strike_from_spot(spot, rd_i, rf_i, T_i)
            premium_pct_i = float(r.premium_percentage)

            sigma_i = implied_volatility(
                option_type.lower(), S=spot, K=K_i, T=T_i, rd=rd_i, rf=rf_i,
                annualized_premium=premium_pct_i
            )
            if sigma_i is not None and np.isfinite(sigma_i):
                T_list.append(T_i)
                sigma_list.append(float(sigma_i))

        if len(T_list) < 2:
            return jsonify({
                "message": "Insufficient valid points to interpolate volatility",
                "details": {"points_found": len(T_list)}
            }), 400

        # 4) Interpolate sigma at target T
        sigma = float(np.interp(T, T_list, sigma_list))

        # 5) Rates & strike at target T
        rd = float(np.interp(T, known_tenors, rates['domestic_yields']))
        rf = float(np.interp(T, known_tenors, rates['foreign_yields']))
        if strike is None:
            strike = strike_from_spot(spot, rd, rf, T)

        # 6) Price & greeks at target with interpolated vol
        theo_price = garman_kohlhagen_price(option_type.lower(), spot, strike, T, rd, rf, sigma)
        delta, gamma, theta = garman_kohlhagen_greeks(option_type.lower(), spot, strike, T, rd, rf, sigma)
        annualized_pct_premium = theo_price / strike
        non_annualized_pct_premium = annualized_pct_premium * T

        forward_rate = strike  # ATM-forward reporting
        pairs = sorted(zip(T_list, sigma_list), key=lambda x: x[0])

        return jsonify({
            "T_years": round(T, 6),
            "currency": currency,
            "option_type": option_type,
            "transaction_type": transaction_type,
            "spot": spot,
            "strike": strike,
            "forward_rate": forward_rate,
            "implied_volatility_interpolated": sigma,
            "unit_price": theo_price,
            "annualized_pct_premium": annualized_pct_premium,
            "non_annualized_pct_premium": non_annualized_pct_premium,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "rd": rd,
            "rf": rf,
            "vol_points_used": [{"T": t, "sigma": s} for t, s in pairs]
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

    finally:
        if 'conn' in locals():
            conn.close()
