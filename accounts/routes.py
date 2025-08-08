
import os, uuid
from pathlib import Path
from flask import (
    request, jsonify, current_app, url_for, send_from_directory
)
from werkzeug.security import generate_password_hash   
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from flask_mail import Message
from admin.routes import _validate_pwd
from models import db, User 
from extentions import limiter, revoke_token          
from . import accounts_bp

ALLOWED_EXTS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ────────────────────────────────────────────────────────────────────────────
@accounts_bp.route("/me", methods=["GET"])
@jwt_required()
def get_profile():
    u: User = User.query.get_or_404(get_jwt_identity())
    return jsonify({
        "id":            u.id,
        "email":         u.email,
        "client_name":   u.client_name,
        "rating":        u.rating,
        "phone_number":  u.phone_number,
        "avatar_url":    u.avatar_url(),
        "address":       u.address,
    }), 200


# ────────────────────────────────────────────────────────────────────────────
@accounts_bp.route("/me", methods=["PUT"])
@jwt_required()
def update_profile():
    u: User = User.query.get_or_404(get_jwt_identity())
    payload = request.get_json(force=True)

    # whitelist of editable fields
    for fld in ("client_name", "phone_number", "address"):
        if fld in payload:
            setattr(u, fld, payload[fld] or None)

    db.session.commit()
    return jsonify(msg="Profile updated"), 200


# ────────────────────────────────────────────────────────────────────────────
@accounts_bp.route("/avatar", methods=["POST"])
@jwt_required()
def upload_avatar():
    if "file" not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify(error="Invalid filename/extension"), 400

    u: User = User.query.get_or_404(get_jwt_identity())
    ext = file.filename.rsplit(".", 1)[1].lower()
    fn  = f"user_{u.id}_{uuid.uuid4().hex[:8]}.{ext}"
    dest_folder = Path(current_app.root_path) / "accounts" / "static" / "avatars"
    dest_folder.mkdir(parents=True, exist_ok=True)
    file.save(dest_folder / fn)

    # if user had a previous avatar you might delete it here
    u.avatar_filename = fn
    db.session.commit()

    return jsonify(avatar_url=u.avatar_url), 201


# ────────────────────────────────────────────────────────────────────────────
# OPTIONAL direct file serving (only if you don’t use web server for static)
# ────────────────────────────────────────────────────────────────────────────
@accounts_bp.route("/avatar/<filename>")
def avatar_raw(filename):
    return send_from_directory(
        os.path.join(current_app.root_path, "accounts", "static", "avatars"),
        filename
    )

def _ts():
    secret = current_app.config["SECRET_KEY"]
    return URLSafeTimedSerializer(secret, salt="pw-reset")
@accounts_bp.post("/forgot-password")
@limiter.limit("3 per minute")
def forgot_password():
    email = request.json.get("email", "").strip().lower()
    user  = User.query.filter_by(email=email).first()
    if not user:
        # never reveal that the address is unknown
        return jsonify(msg="If this email exists you’ll get a reset message"), 200

    token = _ts().dumps({"uid": user.id})
    reset_url = f"{current_app.config['FRONTEND_RESET_URL']}?token={token}"

    # ---- send the email ----------------------------------------------------
    msg = Message(
        subject="Reset your Colombus FX password",
        recipients=[email],
        body=f"Hello,\n\nClick the link below to reset your password:\n{reset_url}\n\n"
             "This link is valid for 30 minutes."
    )
    current_app.extensions["mail"].send(msg)
    # -----------------------------------------------------------------------

    return jsonify(msg="If this email exists you’ll get a reset message"), 200
@accounts_bp.post("/reset-password")
def reset_password():
    token      = request.json.get("token", "")
    new_pwd    = request.json.get("password", "")

    # 1/ validate new pwd strength
    try:
        _validate_pwd(new_pwd)
    except ValueError as e:
        return jsonify(msg=str(e)), 400

    # 2/ decode token
    try:
        data = _ts().loads(token, max_age=60*30)      # 30 min validity
    except SignatureExpired:
        return jsonify(msg="Token expired"), 400
    except BadSignature:
        return jsonify(msg="Invalid token"), 400

    user = User.query.get_or_404(data["uid"])
    user.password = generate_password_hash(new_pwd)
    db.session.commit()
    return jsonify(msg="Password updated"), 200
