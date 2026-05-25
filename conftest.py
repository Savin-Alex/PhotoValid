"""Make the repository root importable.

The test suite imports ``backend.*``. Without this, a bare ``pytest`` invocation
(as opposed to ``python -m pytest``) fails collection with
``ModuleNotFoundError: No module named 'backend'`` because the repo root is not
on ``sys.path``. Placing this conftest at the root puts it there for both.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
