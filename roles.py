# create_roles.py
from app import app, db  # Import the Flask app and database instance from app.py
from models import Role , Order , ExchangeData # Import the Role model (make sure it's defined in models.py)
from datetime import datetime

def create_roles():
    # Check if roles already exist
    if not Role.query.filter_by(name='Admin').first():
        admin = Role(id=1, name='Admin')
        db.session.add(admin)
    if not Role.query.filter_by(name='Client').first():
        client = Role(id=2, name='Client')
        db.session.add(client)

    db.session.commit()
    print("Roles created or already exist!")


if __name__ == "__main__":
    # Ensure the Flask application context is pushed
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
        create_roles()  # Call the function to create roles






