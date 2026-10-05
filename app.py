"""
Entrypoint forwarder for Streamlit Community Cloud.
Dispatches directly to streamlit_app.py.
"""
import runpy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(os.path.dirname(__file__), "streamlit_app.py"), run_name="__main__")
