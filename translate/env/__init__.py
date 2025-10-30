"""
Translation environment module
"""
from .translate import TranslateEnv

# Export the environment as the default
Env = TranslateEnv

__all__ = ['TranslateEnv', 'Env']
