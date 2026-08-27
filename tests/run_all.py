"""Run all smoke tests.

Usage (any directory):
    python tests/run_all.py
    or from the project root: python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

# Make sure the project root is on sys.path (so util/models/trainers import)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(start_dir=os.path.join(ROOT, 'tests'))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)

