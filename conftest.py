"""
Repo-root conftest. Ensures the project root (this directory) is the FIRST entry
on sys.path for every pytest run, no matter what working directory pytest is
launched from. This makes `from core.x import ...` resolve as a package import and
keeps `core/types.py` from ever shadowing the stdlib `types` module (which happens
only when core/ itself lands on sys.path[0]).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
