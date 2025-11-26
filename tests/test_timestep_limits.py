import types

import numpy as np
import pytest

from tde_sph.core.simulation import Simulation, SimulationConfig, SimulationState


class DummyParticles:
    def __init__(self, n: int):
        self.positions = np.zeros((n, 3), dtype=np.float32)
        self.velocities = np.zeros((n, 3), dtype=np.float32)
        self.masses = np.ones(n, dtype=np.float32)
        self.smoothing_lengths = np.ones(n, dtype=np.float32) * 0.1
        self.density = np.ones(n, dtype=np.float32)
        self.pressure = np.ones(n, dtype=np.float32)
        self.sound_speed = np.ones(n, dtype=np.float32)
        self.internal_energy = np.ones(n, dtype=np.float32)


class DummyIntegrator:
    def __init__(self, proposed_dt: float):
        self.proposed_dt = proposed_dt
        self.last_step_dt = None

    def step(self, particles, dt, forces):
        self.last_step_dt = dt

    def estimate_timestep(self, *_args, **_kwargs) -> float:
        return self.proposed_dt


def _make_simulation(dt_initial=1e-2, dt_min=1e-5, dt_max=1e-1, dt_change_limit=2.0):
    config = SimulationConfig(
        dt_initial=dt_initial,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_change_limit=dt_change_limit,
        verbose=False
    )
    sim = Simulation.__new__(Simulation)
    sim.config = config
    sim.state = SimulationState(dt=dt_initial)
    sim._log = lambda *_, **__: None
    return sim


def test_enforce_timestep_limits_respects_bounds_and_change():
    sim_dt_min = _make_simulation(dt_change_limit=1e6)
    sim_dt_min.state.dt = sim_dt_min.config.dt_initial
    clamped = sim_dt_min._enforce_timestep_limits(sim_dt_min.config.dt_min * 1e-2)
    assert clamped == pytest.approx(sim_dt_min.config.dt_min)
    assert sim_dt_min.state.last_dt_limiter == 'dt_min'

    sim_increase = _make_simulation()
    sim_increase.state.dt = sim_increase.config.dt_initial
    clamped = sim_increase._enforce_timestep_limits(sim_increase.config.dt_max * 10)
    assert clamped == pytest.approx(sim_increase.config.dt_initial * sim_increase.config.dt_change_limit)
    assert sim_increase.state.last_dt_limiter == 'dt_max,increase_limit'

    sim_decrease = _make_simulation()
    sim_decrease.state.dt = sim_decrease.config.dt_initial
    candidate = sim_decrease.config.dt_initial / (sim_decrease.config.dt_change_limit * 10)
    clamped = sim_decrease._enforce_timestep_limits(candidate)
    expected = sim_decrease.config.dt_initial / sim_decrease.config.dt_change_limit
    assert clamped == pytest.approx(expected)
    assert sim_decrease.state.last_dt_limiter == 'decrease_limit'


def test_simulation_step_applies_timestep_limits():
    sim = _make_simulation(dt_initial=1e-3, dt_min=5e-4, dt_max=1e-1, dt_change_limit=2.0)
    sim.metric = None
    sim.particles = DummyParticles(n=4)
    sim.integrator = DummyIntegrator(proposed_dt=1.0)

    def fake_compute_forces(_self):
        zeros = np.zeros_like(sim.particles.positions)
        du_dt = np.zeros(sim.particles.positions.shape[0], dtype=np.float32)
        return {'gravity': zeros, 'hydro': zeros, 'total': zeros, 'du_dt': du_dt}

    def fake_compute_energies(_self):
        return {'kinetic': 0.0, 'potential': 0.0, 'internal': 0.0, 'total': 0.0}

    sim.compute_forces = types.MethodType(fake_compute_forces, sim)
    sim.compute_energies = types.MethodType(fake_compute_energies, sim)

    sim.step()

    assert sim.state.dt == pytest.approx(sim.config.dt_initial * sim.config.dt_change_limit)
    assert sim.state.last_dt_limiter == 'dt_max,increase_limit'
    assert sim.integrator.last_step_dt == pytest.approx(sim.config.dt_initial)
    assert sim.state.step == 1
    assert sim.state.time == pytest.approx(sim.config.dt_initial)
