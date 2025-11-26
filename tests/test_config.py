"""
Tests for configuration system (TASK-015).

Validates:
- SimulationConfig Pydantic validation
- YAML/JSON loading
- Configuration consistency checks
- Backward compatibility with Phase 1 Newtonian defaults
"""

import pytest
import warnings
from pathlib import Path
import tempfile
import yaml

from tde_sph.core.simulation import SimulationConfig
from tde_sph.config import load_config, save_config, config_from_dict


class TestSimulationConfigValidation:
    """Test Pydantic validation rules for SimulationConfig."""

    def test_default_config_is_newtonian(self):
        """Default config should be Newtonian mode (backward compatibility)."""
        config = SimulationConfig()
        assert config.mode == "Newtonian"
        assert config.bh_mass == 1.0
        assert config.bh_spin == 0.0

    def test_mode_validation(self):
        """Test mode field validation."""
        # Valid modes
        config_newton = SimulationConfig(mode="Newtonian")
        assert config_newton.mode == "Newtonian"

        config_gr = SimulationConfig(mode="GR")
        assert config_gr.mode == "GR"

        # Invalid mode
        with pytest.raises(ValueError, match="mode must be one of"):
            SimulationConfig(mode="InvalidMode")

    def test_metric_type_validation(self):
        """Test metric_type field validation."""
        # Valid metrics
        for metric in ["minkowski", "schwarzschild", "kerr"]:
            config = SimulationConfig(metric_type=metric)
            assert config.metric_type == metric

        # Invalid metric
        with pytest.raises(ValueError, match="metric_type must be one of"):
            SimulationConfig(metric_type="invalid_metric")

    def test_spin_bounds(self):
        """Test black hole spin bounds [0, 1]."""
        # Valid spins
        SimulationConfig(bh_spin=0.0)
        SimulationConfig(bh_spin=0.5)
        SimulationConfig(bh_spin=1.0)

        # Invalid spins
        with pytest.raises(ValueError):
            SimulationConfig(bh_spin=-0.1)

        with pytest.raises(ValueError):
            SimulationConfig(bh_spin=1.1)

    def test_cfl_factor_bounds(self):
        """Test CFL factor bounds (0, 1]."""
        # Valid CFL
        SimulationConfig(cfl_factor=0.3)
        SimulationConfig(cfl_factor_gr=0.1)

        # Invalid CFL
        with pytest.raises(ValueError):
            SimulationConfig(cfl_factor=0.0)

        with pytest.raises(ValueError):
            SimulationConfig(cfl_factor=1.5)

    def test_time_ordering(self):
        """Test that t_end > t_start."""
        # Valid
        SimulationConfig(t_start=0.0, t_end=10.0)

        # Invalid
        with pytest.raises(ValueError, match="t_end.*must be greater than t_start"):
            SimulationConfig(t_start=10.0, t_end=5.0)

    def test_hamiltonian_integrator_requires_gr(self):
        """Test that Hamiltonian integrator requires GR mode."""
        # Valid: Hamiltonian + GR
        SimulationConfig(mode="GR", use_hamiltonian_integrator=True)

        # Valid: No Hamiltonian in Newtonian
        SimulationConfig(mode="Newtonian", use_hamiltonian_integrator=False)

        # Invalid: Hamiltonian in Newtonian mode
        with pytest.raises(ValueError, match="Hamiltonian integrator.*only valid in GR mode"):
            SimulationConfig(mode="Newtonian", use_hamiltonian_integrator=True)

    def test_gr_with_minkowski_warning(self):
        """Test warning for GR mode with Minkowski metric."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SimulationConfig(mode="GR", metric_type="minkowski")
            assert len(w) == 1
            assert "Minkowski" in str(w[0].message)

    def test_kerr_with_zero_spin_warning(self):
        """Test warning for Kerr metric with zero spin."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SimulationConfig(metric_type="kerr", bh_spin=0.0)
            assert len(w) == 1
            assert "Kerr" in str(w[0].message)

    def test_schwarzschild_with_nonzero_spin_warning(self):
        """Test warning for Schwarzschild with non-zero spin."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SimulationConfig(metric_type="schwarzschild", bh_spin=0.7)
            assert len(w) == 1
            assert "Schwarzschild" in str(w[0].message)

    def test_isco_threshold_warning(self):
        """Test warning for ISCO threshold below photon orbit."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SimulationConfig(isco_radius_threshold=2.5)
            assert len(w) == 1
            assert "photon orbit" in str(w[0].message)


class TestConfigLoaders:
    """Test YAML/JSON configuration loading."""

    def test_load_newtonian_config(self):
        """Test loading Newtonian config from YAML."""
        config_path = Path(__file__).parent.parent / "configs" / "newtonian_tde.yaml"
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_config(config_path)
        assert config.mode == "Newtonian"
        assert config.output_dir == "output/newtonian_tde"

    def test_load_schwarzschild_config(self):
        """Test loading Schwarzschild config from YAML."""
        config_path = Path(__file__).parent.parent / "configs" / "schwarzschild_tde.yaml"
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_config(config_path)
        assert config.mode == "GR"
        assert config.metric_type == "schwarzschild"
        assert config.bh_spin == 0.0

    def test_load_kerr_config(self):
        """Test loading Kerr config from YAML."""
        config_path = Path(__file__).parent.parent / "configs" / "kerr_retrograde_tde.yaml"
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_config(config_path)
        assert config.mode == "GR"
        assert config.metric_type == "kerr"
        assert config.bh_spin == 0.9
        assert config.use_hamiltonian_integrator is True

    def test_config_overrides(self):
        """Test command-line style overrides."""
        config_path = Path(__file__).parent.parent / "configs" / "newtonian_tde.yaml"
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        # Override mode and spin
        config = load_config(config_path, mode="GR", bh_spin=0.5)
        assert config.mode == "GR"
        assert config.bh_spin == 0.5

    def test_load_config_ignores_unknown_keys(self, tmp_path):
        """Unknown keys should be ignored with a warning rather than raising."""
        cfg_path = tmp_path / "extra.yaml"
        cfg_path.write_text(
            """
            simulation:
              mode: "Newtonian"
              t_end: 2.0
            custom_section:
              foo: 1
            """
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = load_config(cfg_path)

        assert config.t_end == 2.0
        assert any("unsupported configuration keys" in str(msg.message) for msg in w)

    def test_save_and_load_roundtrip(self):
        """Test saving and loading config."""
        # Create config
        original = SimulationConfig(
            mode="GR",
            metric_type="kerr",
            bh_spin=0.8,
            t_end=50.0,
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_config(original, temp_path)

            # Load back
            loaded = load_config(temp_path)

            # Compare key fields
            assert loaded.mode == original.mode
            assert loaded.metric_type == original.metric_type
            assert loaded.bh_spin == original.bh_spin
            assert loaded.t_end == original.t_end

        finally:
            temp_path.unlink()

    def test_nested_dict_flattening(self):
        """Test flattening of nested configuration dictionaries."""
        nested_dict = {
            'simulation': {
                'mode': 'GR',
                't_end': 100.0,
            },
            'black_hole': {
                'mass': 1.0,
                'spin': 0.7,
            },
            'metric': {
                'type': 'kerr',
            },
        }

        config = config_from_dict(nested_dict)
        assert config.mode == "GR"
        assert config.t_end == 100.0
        assert config.bh_mass == 1.0
        assert config.bh_spin == 0.7
        assert config.metric_type == "kerr"

    def test_invalid_config_file(self):
        """Test error handling for invalid config files."""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

        # Unsupported format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
            f.write("invalid config")

        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                load_config(temp_path)
        finally:
            temp_path.unlink()


class TestBackwardCompatibility:
    """Ensure Phase 1 Newtonian functionality is preserved."""

    def test_phase1_defaults_unchanged(self):
        """Phase 1 default config should work unchanged."""
        config = SimulationConfig()

        # Core Phase 1 parameters
        assert config.mode == "Newtonian"
        assert config.t_start == 0.0
        assert config.t_end == 10.0
        assert config.dt_initial == 0.001
        assert config.cfl_factor == 0.7
        assert config.output_dir == "outputs/default_run"
        assert config.snapshot_interval == 0.01
        assert config.neighbour_search_method == "bruteforce"
        assert config.smoothing_length_eta == 1.2
        assert config.artificial_viscosity_alpha == 1.0
        assert config.artificial_viscosity_beta == 2.0
        assert config.energy_tolerance == 0.01
        assert config.random_seed == 42
        assert config.verbose is True

    def test_phase1_config_dict_style(self):
        """Phase 1 dictionary-style config creation should work."""
        # Simulate Phase 1 usage pattern
        config = SimulationConfig(
            t_end=5.0,
            dt_initial=0.0005,
            output_dir="test_output",
            cfl_factor=0.25,
        )

        assert config.mode == "Newtonian"
        assert config.t_end == 5.0
        assert config.dt_initial == 0.0005
        assert config.output_dir == "test_output"
        assert config.cfl_factor == 0.25


class TestGRModeConfigurations:
    """Test GR-specific configuration scenarios."""

    def test_minimal_gr_schwarzschild(self):
        """Test minimal GR config for Schwarzschild."""
        config = SimulationConfig(
            mode="GR",
            metric_type="schwarzschild",
        )
        assert config.mode == "GR"
        assert config.metric_type == "schwarzschild"
        assert config.bh_spin == 0.0
        assert config.use_hamiltonian_integrator is False

    def test_full_gr_kerr_config(self):
        """Test full GR Kerr configuration."""
        config = SimulationConfig(
            mode="GR",
            metric_type="kerr",
            bh_mass=1.0,
            bh_spin=0.9,
            use_hamiltonian_integrator=True,
            isco_radius_threshold=8.0,
            cfl_factor_gr=0.08,
            orbital_timestep_factor=0.03,
            coordinate_system="cartesian",
            use_fp64_for_metric=True,
        )

        assert config.mode == "GR"
        assert config.metric_type == "kerr"
        assert config.bh_spin == 0.9
        assert config.use_hamiltonian_integrator is True
        assert config.isco_radius_threshold == 8.0
        assert config.cfl_factor_gr == 0.08
        assert config.orbital_timestep_factor == 0.03
        assert config.use_fp64_for_metric is True

    def test_gr_mode_with_all_metric_types(self):
        """Test GR mode with all supported metric types."""
        for metric in ["minkowski", "schwarzschild", "kerr"]:
            config = SimulationConfig(mode="GR", metric_type=metric)
            assert config.metric_type == metric
