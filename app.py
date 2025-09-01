from flask import Flask
from flask_cors import CORS
from datetime import timedelta  , datetime       
import os
from dotenv import load_dotenv
from extentions import limiter
from models import db, User, Role, RevokedToken, ExchangeData
from scheduler import scheduler, start_scheduler
from scheduler_jobs import check_for_new_files
from flask_security import Security, SQLAlchemySessionUserDatastore
from flask_jwt_extended import JWTManager
from flask_mail import Mail
from admin.routes import admin_bp
from user.routes import user_bp, init_socketio, delete_expired_positions, \
                         update_order_interbank_and_benchmark_rates, \
                         update_interbank_rates_db_logic
from invoice import invoice_bp
from accounts import accounts_bp
from tca.routes import tca_bp

load_dotenv()

# ─── app & core config ───────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/.*": {"origins": [
    "http://localhost:3000",
    "http://localhost:5173"
]}},  supports_credentials=True,)

limiter.init_app(app)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pass123@localhost:5432/postgres'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:pass123@db:5432/postgres'
# app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("SQLALCHEMY_URL")

app.config['SECRET_KEY']     = os.getenv("FLASK_SECRET")
app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET")
app.config['FRONTEND_RESET_URL'] = os.getenv(
    "FRONTEND_RESET_URL",
)

# ---- Flask-Security ------------------------------------------------------------
app.config['SECURITY_REGISTERABLE']       = True
app.config['SECURITY_SEND_REGISTER_EMAIL'] = False

# --- JWT -----------------------------------------------------------------
app.config.update(
    JWT_TOKEN_LOCATION      = ["cookies"],
    JWT_COOKIE_SECURE       = False,         # HTTPS only in prod
    JWT_COOKIE_SAMESITE     = "Lax",          # CSRF protection
    JWT_COOKIE_CSRF_PROTECT = True,         # double-submit protection
    JWT_ACCESS_TOKEN_EXPIRES  = timedelta(minutes=15),
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=7),
)


# ─── init extensions ────────────────────────────────────────────────────────────
mail = Mail()     # put near your other extensions

# … after you load env vars:
app.config.update(
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 465)),
    MAIL_USE_SSL=os.getenv("MAIL_USE_SSL", "false").lower() == "true",
    MAIL_USE_TLS=os.getenv("MAIL_USE_TLS", "false").lower() == "true",
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_DEFAULT_SENDER=os.getenv("MAIL_DEFAULT_SENDER"),
    FRONTEND_RESET_URL= os.getenv("FRONTEND_RESET_URL"),   

)

mail.init_app(app)
jwt = JWTManager(app)
db.init_app(app)


@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    # si on trouve le jti dans la table -> token refusé
    return RevokedToken.query.get(jwt_payload["jti"]) is not None

# ─── blueprints ──────────────────────────────────────────────────────────────────
app.register_blueprint(admin_bp,   url_prefix='/admin')
app.register_blueprint(user_bp,    url_prefix='/')
app.register_blueprint(invoice_bp, url_prefix='/invoice')
app.register_blueprint(accounts_bp, url_prefix='/profile')
app.register_blueprint(tca_bp, url_prefix='/tca')

# ─── Flask-Security setup ───────────────────────────────────────────────────────
user_datastore = SQLAlchemySessionUserDatastore(db.session, User, Role)
security = Security(app, user_datastore)

with app.app_context():
    db.create_all()

# (the rest of your scheduler + socket.io code stays exactly as-is)

# --------------------------------------------------
# 1) Add the check_for_new_files job (5 min interval)
# --------------------------------------------------
scheduler.add_job(
    func=check_for_new_files, 
    trigger='interval', 
    minutes=5,
    id="check_for_new_files_job"
)

# ------------------------------------------------------------
# 2) Add daily job to delete expired positions (24h interval)
# ------------------------------------------------------------
scheduler.add_job(
    func=delete_expired_positions,
    trigger='interval',
    hours=24,
    kwargs={'app': app},
    id="delete_expired_positions_job"
)

# ------------------------------------------------------------
# 3) Add daily job to update interbank rates (24h interval)
# ------------------------------------------------------------
scheduler.add_job(
    func=update_order_interbank_and_benchmark_rates,
    trigger='interval',
    minutes=5,
    kwargs={'app': app},
    id="update_order_interbank_rates_job"
)
# 1) In app.py, define a small wrapper function
def update_interbank_rates_db_wrapper(app, start_date_str):
    with app.app_context():
        update_interbank_rates_db_logic(start_date_str)

# 2) Change your scheduled job to call that wrapper
scheduler.add_job(
    func=lambda: update_interbank_rates_db_wrapper(app, "2024-08-01"),
    trigger='interval',
    minutes=30,
    id="update_interbank_rates_db_job"
)
# Start the scheduler
start_scheduler()

socketio = init_socketio(app)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
    socketio.run(app, debug=True)
