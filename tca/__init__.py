from flask import Blueprint
tca_bp = Blueprint('tca_bp', __name__)   

from . import routes     