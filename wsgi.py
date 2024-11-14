# wsgi.py
from app import app  # Import the Flask instance from app.py

if __name__ == "__main__":
    app.run()  # For running the app locally (optional, as WSGI server will handle this)
