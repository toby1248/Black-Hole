"""
Configuration loaders for YAML and JSON files.

This module provides functions to load and validate simulation configurations
from YAML/JSON files with support for nested dictionaries and unit conversions.

Implements TASK-015-config-b: YAML/JSON loaders
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import yaml
import json
import warnings

from tde_sph.core.simulation import SimulationConfig


def load_config(filename: Union[str, Path], **overrides) -> SimulationConfig:
    """
    Load simulation configuration from YAML or JSON file.

    Supports nested dictionaries and flattens them to match SimulationConfig fields.
    Also supports command-line style overrides.

    Parameters
    ----------
    filename : str or Path
        Path to configuration file (.yaml, .yml, or .json)
    **overrides : keyword arguments
        Override specific config values (e.g., mode="GR", bh_spin=0.9)

    Returns
    -------
    config : SimulationConfig
        Validated simulation configuration

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist
    ValueError
        If file format is unsupported or config is invalid

    Examples
    --------
    >>> config = load_config("schwarzschild_tde.yaml")
    >>> config = load_config("config.yaml", mode="GR", bh_spin=0.7)
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    # Determine file type and load
    suffix = filepath.suffix.lower()
    if suffix in ['.yaml', '.yml']:
        config_dict = load_yaml(filepath)
    elif suffix == '.json':
        config_dict = load_json(filepath)
    else:
        raise ValueError(
            f"Unsupported config file format: {suffix}. "
            "Use .yaml, .yml, or .json"
        )

    # Flatten nested dictionaries
    flat_config = flatten_config(config_dict)

    # Apply overrides
    flat_config.update(overrides)

    # Create and validate config
    try:
        config = SimulationConfig(**flat_config)
    except Exception as e:
        raise ValueError(
            f"Configuration validation failed for {filepath}: {e}"
        ) from e

    return config


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    filepath : Path
        Path to YAML file

    Returns
    -------
    config_dict : Dict[str, Any]
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    return config_dict


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load JSON configuration file.

    Parameters
    ----------
    filepath : Path
        Path to JSON file

    Returns
    -------
    config_dict : Dict[str, Any]
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    return config_dict


def flatten_config(config_dict: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
    """
    Flatten nested configuration dictionary.

    Converts nested structures like:
        {'black_hole': {'mass': 1.0, 'spin': 0.7}}
    to:
        {'bh_mass': 1.0, 'bh_spin': 0.7}

    Also handles common aliases and performs unit conversions.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Nested configuration dictionary
    parent_key : str
        Parent key for recursion

    Returns
    -------
    flat_dict : Dict[str, Any]
        Flattened configuration dictionary
    """
    flat = {}

    # Field mapping for nested structures
    FIELD_MAPPINGS = {
        'black_hole': {
            'mass': 'bh_mass',
            'spin': 'bh_spin',
        },
        'simulation': {
            'mode': 'mode',
            't_start': 't_start',
            't_end': 't_end',
            'dt_initial': 'dt_initial',
            'snapshot_interval': 'snapshot_interval',
            'log_interval': 'log_interval',
            'output_dir': 'output_dir',
        },
        'metric': {
            'type': 'metric_type',
        },
        'integration': {
            'use_hamiltonian_integrator': 'use_hamiltonian_integrator',
            'isco_radius_threshold': 'isco_radius_threshold',
            'cfl_factor': 'cfl_factor',
            'cfl_factor_gr': 'cfl_factor_gr',
            'orbital_timestep_factor': 'orbital_timestep_factor',
        },
        'sph': {
            'neighbour_search_method': 'neighbour_search_method',
            'smoothing_length_eta': 'smoothing_length_eta',
            'artificial_viscosity_alpha': 'artificial_viscosity_alpha',
            'artificial_viscosity_beta': 'artificial_viscosity_beta',
        },
    }

    for key, value in config_dict.items():
        # Check if this is a nested section with mappings
        if key in FIELD_MAPPINGS and isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey in FIELD_MAPPINGS[key]:
                    mapped_key = FIELD_MAPPINGS[key][subkey]
                    flat[mapped_key] = subvalue
                else:
                    # Pass through unmapped keys
                    flat[subkey] = subvalue
        elif isinstance(value, dict):
            # Recursively flatten nested dicts without explicit mappings
            nested = flatten_config(value, parent_key=key)
            flat.update(nested)
        else:
            # Direct assignment
            flat[key] = value

    return flat


def save_config(config: SimulationConfig, filename: Union[str, Path]) -> None:
    """
    Save SimulationConfig to YAML file.

    Parameters
    ----------
    config : SimulationConfig
        Configuration to save
    filename : str or Path
        Output file path (.yaml or .yml)
    """
    filepath = Path(filename)

    # Convert config to dictionary
    config_dict = config.model_dump()

    # Organize into nested structure for readability
    organized = {
        'simulation': {
            'mode': config_dict['mode'],
            't_start': config_dict['t_start'],
            't_end': config_dict['t_end'],
            'dt_initial': config_dict['dt_initial'],
            'snapshot_interval': config_dict['snapshot_interval'],
            'log_interval': config_dict['log_interval'],
            'output_dir': config_dict['output_dir'],
        },
        'black_hole': {
            'mass': config_dict['bh_mass'],
            'spin': config_dict['bh_spin'],
        },
        'metric': {
            'type': config_dict['metric_type'],
            'coordinate_system': config_dict['coordinate_system'],
        },
        'integration': {
            'use_hamiltonian_integrator': config_dict['use_hamiltonian_integrator'],
            'isco_radius_threshold': config_dict['isco_radius_threshold'],
            'cfl_factor': config_dict['cfl_factor'],
            'cfl_factor_gr': config_dict['cfl_factor_gr'],
            'orbital_timestep_factor': config_dict['orbital_timestep_factor'],
        },
        'sph': {
            'neighbour_search_method': config_dict['neighbour_search_method'],
            'smoothing_length_eta': config_dict['smoothing_length_eta'],
            'artificial_viscosity_alpha': config_dict['artificial_viscosity_alpha'],
            'artificial_viscosity_beta': config_dict['artificial_viscosity_beta'],
        },
        'precision': {
            'use_fp64_for_metric': config_dict['use_fp64_for_metric'],
        },
        'misc': {
            'energy_tolerance': config_dict['energy_tolerance'],
            'random_seed': config_dict['random_seed'],
            'verbose': config_dict['verbose'],
        },
    }

    # Write YAML
    suffix = filepath.suffix.lower()
    if suffix in ['.yaml', '.yml']:
        with open(filepath, 'w') as f:
            yaml.dump(organized, f, default_flow_style=False, sort_keys=False)
    elif suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(organized, f, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {suffix}. Use .yaml or .json")


def config_from_dict(config_dict: Dict[str, Any]) -> SimulationConfig:
    """
    Create SimulationConfig from dictionary (helper for programmatic use).

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    config : SimulationConfig
        Validated configuration
    """
    flat = flatten_config(config_dict)
    return SimulationConfig(**flat)
