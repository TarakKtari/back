# extensions.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime, timedelta
from models import db, RevokedToken

# 1. Limiter instance
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[]
)

# 2. Token revoke helper
def revoke_token(decoded_token):
    db.session.add(
        RevokedToken(
            jti     = decoded_token["jti"],
            expires = datetime.utcnow() + timedelta(days=30)
        )
    )
    db.session.commit()
