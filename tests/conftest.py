"""Make experiments/ importable and expose the fixture path."""
import os
import sys

HERE = os.path.dirname(__file__)
EXPERIMENTS = os.path.abspath(os.path.join(HERE, "..", "experiments"))
if EXPERIMENTS not in sys.path:
    sys.path.insert(0, EXPERIMENTS)
