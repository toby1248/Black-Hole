#!/usr/bin/env python3
"""
Configuration System Demo for Phase 2

Demonstrates:
1. Loading configs from YAML files
2. Creating configs programmatically
3. Config validation and error handling
4. GR vs Newtonian mode setup
5. Config overrides

Run from project root:
    python examples/config_demo.py
"""

from pathlib import Path
import warnings

from tde_sph.core.simulation import SimulationConfig
from tde_sph.config import load_config, save_config, config_from_dict


def demo_default_config():
    """Demo 1: Default configuration (Newtonian)."""
    print("=" * 70)
    print("DEMO 1: Default Configuration (Backward Compatible)")
    print("=" * 70)

    config = SimulationConfig()

    print(f"Mode: {config.mode}")
    print(f"Metric type: {config.metric_type}")
    print(f"BH mass: {config.bh_mass}")
    print(f"BH spin: {config.bh_spin}")
    print(f"t_end: {config.t_end}")
    print(f"CFL factor (Newtonian): {config.cfl_factor}")
    print(f"CFL factor (GR): {config.cfl_factor_gr}")
    print()


def demo_load_yaml_configs():
    """Demo 2: Load example YAML configurations."""
    print("=" * 70)
    print("DEMO 2: Loading YAML Configurations")
    print("=" * 70)

    config_dir = Path(__file__).parent.parent / "configs"

    configs_to_load = [
        "newtonian_tde.yaml",
        "schwarzschild_tde.yaml",
        "kerr_retrograde_tde.yaml",
    ]

    for config_file in configs_to_load:
        config_path = config_dir / config_file
        if not config_path.exists():
            print(f"⚠️  Config not found: {config_file}")
            continue

        print(f"\n📄 Loading: {config_file}")
        config = load_config(config_path)

        print(f"   Mode: {config.mode}")
        print(f"   Metric: {config.metric_type}")
        print(f"   BH spin: {config.bh_spin}")
        print(f"   Hamiltonian integrator: {config.use_hamiltonian_integrator}")
        print(f"   Output dir: {config.output_dir}")

    print()


def demo_programmatic_config():
    """Demo 3: Create configs programmatically."""
    print("=" * 70)
    print("DEMO 3: Programmatic Configuration Creation")
    print("=" * 70)

    # GR Kerr simulation
    config = SimulationConfig(
        mode="GR",
        metric_type="kerr",
        bh_mass=1.0,
        bh_spin=0.7,
        t_end=100.0,
        use_hamiltonian_integrator=True,
        isco_radius_threshold=8.0,
        output_dir="output/kerr_spin_0.7",
    )

    print("Created GR Kerr config:")
    print(f"  Mode: {config.mode}")
    print(f"  Metric: {config.metric_type}")
    print(f"  Spin: {config.bh_spin}")
    print(f"  Hamiltonian integrator: {config.use_hamiltonian_integrator}")
    print()


def demo_nested_dict_config():
    """Demo 4: Create config from nested dictionary."""
    print("=" * 70)
    print("DEMO 4: Config from Nested Dictionary")
    print("=" * 70)

    config_dict = {
        'simulation': {
            'mode': 'GR',
            't_end': 50.0,
        },
        'black_hole': {
            'mass': 1.0,
            'spin': 0.5,
        },
        'metric': {
            'type': 'kerr',
        },
        'integration': {
            'use_hamiltonian_integrator': False,
        },
    }

    config = config_from_dict(config_dict)

    print("Config created from nested dict:")
    print(f"  Mode: {config.mode}")
    print(f"  Metric: {config.metric_type}")
    print(f"  BH spin: {config.bh_spin}")
    print(f"  t_end: {config.t_end}")
    print()


def demo_config_overrides():
    """Demo 5: Config overrides (simulating CLI usage)."""
    print("=" * 70)
    print("DEMO 5: Configuration Overrides")
    print("=" * 70)

    config_path = Path(__file__).parent.parent / "configs" / "newtonian_tde.yaml"

    if not config_path.exists():
        print(f"⚠️  Base config not found: {config_path}")
        return

    # Load base config and override to GR mode
    print(f"Base config: {config_path.name}")
    base_config = load_config(config_path)
    print(f"  Original mode: {base_config.mode}")

    # Override to GR
    print("\nOverriding: mode='GR', bh_spin=0.8, metric_type='kerr'")
    overridden = load_config(
        config_path,
        mode="GR",
        bh_spin=0.8,
        metric_type="kerr",
    )
    print(f"  New mode: {overridden.mode}")
    print(f"  New metric: {overridden.metric_type}")
    print(f"  New spin: {overridden.bh_spin}")
    print()


def demo_validation_warnings():
    """Demo 6: Validation warnings and errors."""
    print("=" * 70)
    print("DEMO 6: Configuration Validation")
    print("=" * 70)

    # Warning: Kerr with zero spin
    print("Creating Kerr config with a=0 (should warn):")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = SimulationConfig(metric_type="kerr", bh_spin=0.0)
        if w:
            print(f"  ⚠️  {w[0].message}")

    # Warning: Schwarzschild with non-zero spin
    print("\nCreating Schwarzschild config with a=0.5 (should warn):")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = SimulationConfig(metric_type="schwarzschild", bh_spin=0.5)
        if w:
            print(f"  ⚠️  {w[0].message}")

    # Error: Hamiltonian in Newtonian mode
    print("\nTrying Hamiltonian integrator in Newtonian mode (should error):")
    try:
        config = SimulationConfig(
            mode="Newtonian",
            use_hamiltonian_integrator=True
        )
    except ValueError as e:
        print(f"  ❌ ValidationError: {e}")

    # Error: Invalid mode
    print("\nTrying invalid mode (should error):")
    try:
        config = SimulationConfig(mode="InvalidMode")
    except ValueError as e:
        print(f"  ❌ ValidationError: {e}")

    print()


def demo_save_config():
    """Demo 7: Save config to file."""
    print("=" * 70)
    print("DEMO 7: Saving Configuration to File")
    print("=" * 70)

    # Create a custom config
    config = SimulationConfig(
        mode="GR",
        metric_type="kerr",
        bh_spin=0.95,
        t_end=200.0,
        use_hamiltonian_integrator=True,
        isco_radius_threshold=7.0,
        output_dir="output/custom_kerr",
    )

    # Save to temporary file
    output_path = Path("temp_config_demo.yaml")
    save_config(config, output_path)

    print(f"Config saved to: {output_path}")

    # Read it back to verify
    loaded = load_config(output_path)
    print(f"Verified roundtrip: mode={loaded.mode}, spin={loaded.bh_spin}")

    # Clean up
    output_path.unlink()
    print("Temp file removed")
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TDE-SPH Phase 2 Configuration System Demo" + " " * 11 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    demo_default_config()
    demo_load_yaml_configs()
    demo_programmatic_config()
    demo_nested_dict_config()
    demo_config_overrides()
    demo_validation_warnings()
    demo_save_config()

    print("=" * 70)
    print("✅ All demos complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Explore example configs in configs/")
    print("  2. Create your own YAML config file")
    print("  3. Load with: load_config('your_config.yaml')")
    print("  4. Run tests: pytest tests/test_config.py -v")
    print()


if __name__ == "__main__":
    main()
