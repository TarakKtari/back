from datetime import datetime
import pandas as pd

def convert_to_date(value):
    if pd.isnull(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    try:
        return pd.to_datetime(value).date()
    except ValueError:
        return None

def allowed_file(filename, allowed_extensions={'xlsx'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
