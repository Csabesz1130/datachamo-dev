"""
Jel feldolgozó modul - Az alkalmazás jel feldolgozó funkcióit tartalmazza.
"""

from .action_potential import ActionPotentialProcessor
from .purple_integration_control import PurpleIntegrationController
from .curve_processor import CurveProcessor
from .derivative_analyzer import DerivativeAnalyzer

__all__ = [
    'ActionPotentialProcessor',
    'PurpleIntegrationController',
    'CurveProcessor',
    'DerivativeAnalyzer'
]