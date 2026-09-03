"""MuJoCo adapter for the Unitree R1 foothold policy."""

from .foothold import FixedFootholdGenerator
from .policy import NumpyMLPPolicy

__all__ = ["FixedFootholdGenerator", "NumpyMLPPolicy"]
