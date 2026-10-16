"""Streamlit Community Cloud entry point.

Call the shared UI on every rerun; importing it alone only executes it once
per Python process and leaves later interactions and sessions blank.
"""

from frontend.app import main

main()
