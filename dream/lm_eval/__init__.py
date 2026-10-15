import logging
import os

try:
    from .evaluator import evaluate, simple_evaluate
except ImportError:
    evaluate = None
    simple_evaluate = None


__version__ = "0.4.8"
