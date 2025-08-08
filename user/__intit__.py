# user/__init__.py
from flask import Blueprint


user = Blueprint('user', __name__, template_folder='templates')

from user import tca_routes 