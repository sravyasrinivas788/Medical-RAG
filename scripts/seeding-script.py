import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import setup_tables, seed_dummy_data

setup_tables()
seed_dummy_data()
