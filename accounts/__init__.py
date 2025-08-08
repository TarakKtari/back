from flask import Blueprint
accounts_bp = Blueprint('accounts_bp', __name__)   

from . import routes     