# api/index.py
from app import app  # Import the Flask instance from app.py

# Vercel handler for the Flask app
# Vercel automatically exposes this as an endpoint
app = app
