from flask import Blueprint
invoice_bp = Blueprint('invoice_bp', __name__)   # single source of truth

from . import routes      # routes.py will import this object
