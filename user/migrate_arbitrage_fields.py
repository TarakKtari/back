#!/usr/bin/env python3
"""
Migration script to add TND rate field to the Order table.
Run this script to update the database schema.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from models import db, Order
from sqlalchemy import text

def migrate_tnd_rate_field():
    """Add TND rate field to Order table."""
    
    with app.app_context():
        try:
            # Add TND rate column to the Order table
            print("Adding TND rate field to Order table...")
            
            # Check if column already exists
            inspector = db.inspect(db.engine)
            existing_columns = [col['name'] for col in inspector.get_columns('order')]
            
            if 'tnd_rate' not in existing_columns:
                db.engine.execute(text("ALTER TABLE `order` ADD COLUMN tnd_rate FLOAT NULL"))
                print("✓ Added tnd_rate column")
            else:
                print("✓ tnd_rate column already exists")
            
            # Set default value of 1.0 for existing records
            print("\nSetting default TND rate for existing orders...")
            db.engine.execute(text("UPDATE `order` SET tnd_rate = 1.0 WHERE tnd_rate IS NULL"))
            print("✓ Updated existing orders with default TND rate")
            
            print("\nMigration completed successfully!")
            
        except Exception as e:
            print(f"Error during migration: {e}")
            db.session.rollback()
            raise

if __name__ == "__main__":
    migrate_tnd_rate_field()
