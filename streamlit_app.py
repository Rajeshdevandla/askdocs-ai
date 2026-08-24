"""Streamlit Community Cloud entry point.

The deploy configuration targets this root file. Importing the application module
keeps the hosted app and the tested frontend implementation on the same code path.
"""

from frontend.app import *  # noqa: F401,F403
