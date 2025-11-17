"""
Configuration module: parameter management and run configuration.

Provides YAML/JSON configuration loading and validation for TDE-SPH simulations.
"""

from tde_sph.config.loaders import (
    load_config,
    save_config,
    config_from_dict,
    flatten_config,
)

__all__ = [
    'load_config',
    'save_config',
    'config_from_dict',
    'flatten_config',
]
