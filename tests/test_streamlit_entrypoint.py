"""Regression test for the Streamlit Community Cloud entry point."""

import ast
from pathlib import Path


def test_streamlit_cloud_entrypoint_loads_tested_frontend():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "frontend.app" in imported_modules
