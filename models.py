from flask import url_for
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Mapped
from sqlalchemy import String, Enum, CheckConstraint
import uuid
from flask_security import UserMixin, RoleMixin
from datetime import datetime
import enum                     
from sqlalchemy import Enum as SAEnum

db = SQLAlchemy()

roles_users = db.Table('roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    email = db.Column(db.String, nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)
    active = db.Column(db.Boolean(), nullable=True)
    client_name = db.Column(db.String(255), nullable=False)  
    rating = db.Column(db.Integer, nullable=False, default=0)  
    fixed_monthly_fee   = db.Column(db.Float,   default=0.0)     
    tva_exempt          = db.Column(db.Boolean, default=False)   
    uses_digital_sign   = db.Column(db.Boolean, default=True)    
    netting_enabled     = db.Column(db.Boolean, default=False)   # règle BK Food
    needs_references    = db.Column(db.Boolean, default=False)   # ajoute colonne Réf.
    matricule_fiscal    = db.Column(db.String(32))               # M.F. dans le PDF
    address             = db.Column(db.String(255))            
    contract_start = db.Column(db.Date, nullable=True)  # Date de début du contrat
    contract_end = db.Column(db.Date, nullable=True)    # Date de fin du contrat
    phone_number   = db.Column(db.String(32))         
    avatar_filename = db.Column(db.String(255))      
    def avatar_url(self):
            if not self.avatar_filename:
                return None
            return url_for(
                "accounts_bp.avatar_raw",      
                filename=self.avatar_filename,
                _external=True,                # returns full http://…/profile/avatar/<file>
            )
          
    roles = db.relationship('Role', secondary=roles_users, backref='roled')
    fs_uniquifier: Mapped[str] = db.Column(String(64), unique=True, nullable=True, default=lambda: str(uuid.uuid4()))
    @property
    def is_admin(self):
        return any(r.name and r.name.lower() == "admin" for r in self.roles)
    
class Role(db.Model, RoleMixin):
    __tablename__ = 'role'
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)

class BankAccount(db.Model):
    id = db.Column(db.String, primary_key=True)
    bank_name = db.Column(db.String(80), nullable=False)
    currency = db.Column(db.String(3), nullable=False)
    owner = db.Column(db.String(120), nullable=False)
    balance = db.Column(db.Float, nullable=False)
    account_number = db.Column(db.String(20), unique=True, nullable=False)
    branch = db.Column(db.String(100))
    category = db.Column(db.String(50))
    date = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    user = db.relationship('User', backref='bank_accounts', lazy=True)

class OpenPosition(db.Model):
    __tablename__ = 'open_position'
    id = db.Column(db.Integer, primary_key=True)
    id_unique = db.Column(db.String(80), nullable=False, default=lambda: str(uuid.uuid4()))
    value_date = db.Column(db.Date, nullable=False)
    currency = db.Column(db.String(3), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    transaction_type = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    user = db.relationship('User', backref='open_positions', lazy=True)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    id_unique = db.Column(db.String(80), nullable=False, default=lambda: str(uuid.uuid4()))
    transaction_type = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)  # Remaining amount for unmatched or partially matched orders
    original_amount = db.Column(db.Float, nullable=False)  # Initial order amount
    currency = db.Column(db.String(3), nullable=False, index=True)
    value_date = db.Column(db.Date, nullable=False, index=True)
    transaction_date = db.Column(db.Date, nullable=False)
    order_date = db.Column(db.Date, nullable=False)
    bank_account = db.Column(db.String(100))
    reference = db.Column(db.String(100))
    signing_key = db.Column(db.String(255))
    status = db.Column(Enum('Pending', 'Executed', 'Market','Cancelled', 'Matched', name='order_status'), nullable=False, default='Pending')
    rating = db.Column(db.Integer)
    deleted = db.Column(db.Boolean, default=False, nullable=False)
    historical_loss = db.Column(db.Float, nullable=True)
    interbank_rate = db.Column(db.Float, nullable=True, default=None)
    execution_rate = db.Column(db.Float, nullable=True, default=None)
    bank_name = db.Column(db.String(100), nullable=True, default=None)
    matched_order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=True)  
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    matched_amount = db.Column(db.Float, nullable=True, default=0.0)  
    option_type = db.Column(db.String(10), nullable=True)  # "CALL"/"PUT"
    strike = db.Column(db.Float, nullable=True)            # user input or None
    forward_rate = db.Column(db.Float, nullable=True)      # store the computed forward
    moneyness = db.Column(db.String(20), nullable=True) 
    premium = db.Column(db.Float, nullable=True)  
    is_option = db.Column(db.Boolean, default=False)
    trade_type = db.Column(db.String(10), nullable=False, default="spot")  #  "spot", "forward", or "option"
    benchmark_rate = db.Column(db.Float, nullable=True, default=None)  
    gain = db.Column(db.Float, nullable=True, default=0)
    gain_percentage = db.Column(db.Float, nullable=True, default=0)
    commission_percent = db.Column(db.Float)  
    commission_gain = db.Column(db.Float, nullable=True, default=0.0) 
    # Relationships
    user = db.relationship('User', backref='orders', lazy=True)
    matched_order = db.relationship('Order', remote_side=[id], backref='related_orders')
    # Adding a check constraint for the 'amount'
    __table_args__ = (
        CheckConstraint('amount >= 0', name='check_amount_positive'),
    )

class Meeting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(100), nullable=False)
    representative_name = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    notes = db.Column(db.Text)

class ExchangeData(db.Model):
    __tablename__ = 'exchange_data'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, name='Date')
    spot_usd = db.Column(db.Float, nullable=False, name='Spot USD')
    spot_eur = db.Column(db.Float, nullable=False, name='Spot EUR')
    
    # TND yield curve
    tnd_ond = db.Column(db.Float, nullable=False, name='TND Overnight')
    tnd_1m = db.Column(db.Float, nullable=False, name='1M TND')
    tnd_3m = db.Column(db.Float, nullable=False, name='3M TND')
    tnd_6m = db.Column(db.Float, nullable=False, name='6M TND')
    tnd_1y = db.Column(db.Float, nullable=False, name='1Y TND')
    
    # EUR yield curve
    eur_ond = db.Column(db.Float, nullable=False, name='EUR Overnight')
    eur_1m = db.Column(db.Float, nullable=False, name='1M EUR')
    eur_3m = db.Column(db.Float, nullable=False, name='3M EUR')
    eur_6m = db.Column(db.Float, nullable=False, name='6M EUR')
    eur_1y = db.Column(db.Float, nullable=False, name='1Y EUR')
    
    # USD yield curve
    usd_ond = db.Column(db.Float, nullable=False, name='USD Overnight')
    usd_1m = db.Column(db.Float, nullable=False, name='1M USD')
    usd_3m = db.Column(db.Float, nullable=False, name='3M USD')
    usd_6m = db.Column(db.Float, nullable=False, name='6M USD')
    usd_1y = db.Column(db.Float, nullable=False, name='1Y USD')


class PremiumRate(db.Model):
    """
    Calibration input for FX option pricing.

    Stores historical market premium percentages for given:
    - Maturity (in days)
    - Currency (e.g., EUR, USD)
    - Option type ("CALL" or "PUT")
    - Transaction type ("buy" or "sell")

    These values are used to interpolate percentage premiums for different maturities
    and solve for implied volatilities via the Garman-Kohlhagen model.
    """
    __tablename__ = 'premium_rate'
    id = db.Column(db.Integer, primary_key=True)
    currency = db.Column(db.String(3), nullable=False)    
    maturity_days = db.Column(db.Integer, nullable=False) 
    premium_percentage = db.Column(db.Float, nullable=False)
    option_type = db.Column(db.String(4), nullable=False)  # "CALL" or "PUT"
    transaction_type = db.Column(db.String(4), nullable=False, default="buy")
    spot = db.Column(db.Float, nullable=False)  # required
    strike = db.Column(db.Float, nullable=True) # optional

class AuditLog(db.Model):
    __tablename__ = 'audit_log'
    id = db.Column(db.Integer, primary_key=True)
    action_type = db.Column(db.String(50), nullable=False) 
    table_name = db.Column(db.String(50), nullable=False)  
    record_id = db.Column(db.String(80), nullable=False)  
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) 
    timestamp = db.Column(db.DateTime, default=datetime.now, nullable=False)
    details = db.Column(db.Text)  # JSON string 
    user = db.relationship('User', backref='audit_logs')

class InterbankRate(db.Model):
    __tablename__ = 'interbank_rate'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    currency = db.Column(db.String(3), nullable=False)
    rate = db.Column(db.Float, nullable=False)
    __table_args__ = (db.UniqueConstraint('date', 'currency', name='uix_date_currency'),)

class InternalEmail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=True)
    email_type = db.Column(db.String(50), nullable=False)  # e.g., "confirmation" or "interbank"
    subject = db.Column(db.String(255), nullable=False)
    body = db.Column(db.Text, nullable=False)
    sender = db.Column(db.String(255), nullable=False)
    recipient = db.Column(db.String(255), nullable=False)
    cc = db.Column(db.String(255))  
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
    order = db.relationship('Order', backref='internal_emails', lazy=True)

class BctxFixing(db.Model):
        __tablename__ = 'bctx_fixings'

        id = db.Column(db.Integer, primary_key=True)
        # The 'date' of the fixing
        date = db.Column(db.Date, nullable=False)
        # "morning" or "afternoon"
        session = db.Column(db.String(10), nullable=False)
        original_timestamp = db.Column(db.DateTime, nullable=True)
        # The BID/ASK columns
        tnd_bid = db.Column(db.Float)
        tnd_ask = db.Column(db.Float)
        eur_bid = db.Column(db.Float)
        eur_ask = db.Column(db.Float)
        gbp_bid = db.Column(db.Float)
        gbp_ask = db.Column(db.Float)
        jpy_bid = db.Column(db.Float)
        jpy_ask = db.Column(db.Float)

        # __table_args__ = (
        #     db.UniqueConstraint('date', 'session', name='unique_date_session'),
        # )
        def __repr__(self):
            return f"<BctxFixing {self.date} {self.session} TND_BID={self.tnd_bid}>"
        
class TcaSpotInput(db.Model):
    __tablename__ = 'tca_spot_input'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    client_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    transaction_date = db.Column(db.Date,   nullable=False)
    value_date       = db.Column(db.Date,   nullable=False)
    currency         = db.Column(db.String(3), nullable=False)
    amount           = db.Column(db.Float,  nullable=False)
    execution_rate   = db.Column(db.Float,  nullable=False)
    transaction_type = db.Column(
        Enum('import', 'export', name='tca_transaction_type'),
        nullable=False
    )
    client = db.relationship('User', backref='tca_spot_inputs', lazy=True)        

class InvoiceStatus(enum.Enum):
    draft   = "draft"      # JSON generated, waiting for admin to review
    sent    = "sent"       # PDF confirmed & sent to the client
    paid    = "paid"       # client paid (or manually marked as such)

class Invoice(db.Model):
    __tablename__ = "invoice"
    id            = db.Column(db.Integer, primary_key=True)
    client_id     = db.Column(db.Integer,
                              db.ForeignKey("user.id", ondelete="CASCADE"),
                              nullable=False, index=True)
    year          = db.Column(db.Integer, nullable=False)
    month         = db.Column(db.Integer, nullable=False)          
    creation_date = db.Column(db.Date,    nullable=False)          
    due_date      = db.Column(db.Date,    nullable=False)           # +7 days
    status        = db.Column(SAEnum(InvoiceStatus), default=InvoiceStatus.draft)
    json_payload  = db.Column(db.JSON)     
    pdf_url       = db.Column(db.String)    
    total_ht      = db.Column(db.Float)   
    tva           = db.Column(db.Float)
    stamp_duty    = db.Column(db.Float)
    total_ttc     = db.Column(db.Float)
    client = db.relationship("User", backref="invoices", lazy=True)
    __table_args__ = (
        db.UniqueConstraint("client_id", "year", "month",
                            name="uix_invoice_client_period"),
    )
    
class RevokedToken(db.Model):
    jti     = db.Column(db.String, primary_key=True)  
    expires = db.Column(db.DateTime)
class ATMVol(db.Model):
    __tablename__ = 'atm_vol'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    tenor = db.Column(db.String(5), nullable=False)  # e.g., "1W", "1M"
    bid = db.Column(db.Float, nullable=True)
    ask = db.Column(db.Float, nullable=True)
    mid = db.Column(db.Float, nullable=True)
    tau_days = db.Column(db.Integer, nullable=True)
    tau_years = db.Column(db.Float, nullable=True)
    __table_args__ = (db.UniqueConstraint('date', 'tenor', name='uix_date_tenor'),)
