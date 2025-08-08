# scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone
import atexit

# Initialize the scheduler with a specific time zone
scheduler = BackgroundScheduler(timezone=timezone('Africa/Tunis'))

def start_scheduler():
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
