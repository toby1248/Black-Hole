"""
Phase 3 Demo: Radiation EOS, Energy Diagnostics, Cooling, and Diagnostic Outputs

This demo showcases all Phase 3 features:
- RadiationGas EOS with gas + radiation pressure
- EnergyDiagnostics for comprehensive energy accounting
- SimpleCooling for radiative cooling
- DiagnosticsWriter for time-series outputs

Usage:
    python examples/phase3_demo.py
"""

import numpy as np
from pathlib import Path

from tde_sph.eos import IdealGas, RadiationGas
from tde_sph.radiation import SimpleCooling
from tde_sph.core import EnergyDiagnostics
from tde_sph.io import DiagnosticsWriter, compute_fallback_rate
from tde_sph.gravity.newtonian import NewtonianGravity


def create_test_particles(N=1000, spread=10.0):
    """Create a simple test particle system."""
    np.random.seed(42)

    # Random positions in a sphere
    r = np.random.uniform(0, spread, N)
    theta = np.random.uniform(0, np.pi, N)
    phi = np.random.uniform(0, 2*np.pi, N)

    positions = np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ]).astype(np.float32)

    # Circular velocities (simplified)
    v_mag = np.sqrt(1.0 / np.maximum(r, 1.0))  # Keplerian
    velocities = np.column_stack([
        -v_mag * np.sin(phi),
        v_mag * np.cos(phi),
        np.zeros(N)
    ]).astype(np.float32) * 0.1

    particles = {
        'positions': positions,
        'velocities': velocities,
        'masses': np.ones(N, dtype=np.float32) * (1.0 / N),
        'internal_energy': np.ones(N, dtype=np.float32) * 1e15,  # High energy for radiation
        'density': np.ones(N, dtype=np.float32) * 1e-8,
        'smoothing_length': np.ones(N, dtype=np.float32) * 1.0
    }

    return particles


def demo_radiation_eos():
    """Demonstrate RadiationGas EOS."""
    print("=" * 70)
    print("PHASE 3 DEMO: RadiationGas EOS")
    print("=" * 70)

    ideal_eos = IdealGas(gamma=5/3)
    rad_eos = RadiationGas(gamma=5/3)

    # Test particles with varying temperatures
    rho = np.array([1e-10, 1e-8, 1e-6], dtype=np.float32)
    u = np.array([1e12, 1e15, 1e18], dtype=np.float32)

    print("\nComparing IdealGas vs RadiationGas:")
    print(f"{'Density':>12} {'Int. Energy':>14} {'T (Ideal)':>12} {'T (Rad+Gas)':>14} {'Beta':>8}")
    print("-" * 70)

    for i in range(len(rho)):
        T_ideal = ideal_eos.temperature(np.array([rho[i]]), np.array([u[i]]))
        T_rad = rad_eos.temperature(np.array([rho[i]]), np.array([u[i]]))
        beta = rad_eos.beta_parameter(np.array([rho[i]]), np.array([u[i]]))

        print(f"{rho[i]:12.3e} {u[i]:14.3e} {T_ideal[0]:12.3e} {T_rad[0]:14.3e} {beta[0]:8.4f}")

    print("\nBeta parameter interpretation:")
    print("  β ≈ 1.0: Gas pressure dominated")
    print("  β ≈ 0.5: Transition regime")
    print("  β ≈ 0.0: Radiation pressure dominated")


def demo_energy_diagnostics():
    """Demonstrate EnergyDiagnostics."""
    print("\n" + "=" * 70)
    print("PHASE 3 DEMO: Energy Diagnostics")
    print("=" * 70)

    particles = create_test_particles(N=500)
    eos = RadiationGas()
    gravity_solver = NewtonianGravity()
    diagnostics = EnergyDiagnostics()

    # Compute diagnostics
    result = diagnostics.compute(particles, gravity_solver, eos, bh_mass=1.0)

    print("\nEnergy Components:")
    print(f"  E_kinetic:              {result['E_kinetic']:14.6e}")
    print(f"  E_potential:            {result['E_potential']:14.6e}")
    print(f"  E_internal_total:       {result['E_internal_total']:14.6e}")
    print(f"    ├─ E_internal_gas:    {result['E_internal_gas']:14.6e}")
    print(f"    └─ E_internal_rad:    {result['E_internal_radiation']:14.6e}")
    print(f"  E_total:                {result['E_total']:14.6e}")
    print(f"  Energy conservation:    {result['energy_conservation']:14.6e}")

    print("\nGlobal Quantities:")
    print(f"  Total mass:             {result['total_mass']:14.6e}")
    print(f"  Linear momentum:        {np.linalg.norm(result['linear_momentum']):14.6e}")
    print(f"  Angular momentum:       {np.linalg.norm(result['angular_momentum']):14.6e}")

    print("\nThermodynamic Statistics:")
    print(f"  T_min:                  {result['T_min']:14.3e} K")
    print(f"  T_max:                  {result['T_max']:14.3e} K")
    print(f"  T_mean:                 {result['T_mean']:14.3e} K")
    print(f"  P_mean:                 {result['P_mean']:14.3e}")

    if 'beta_mean' in result:
        print(f"  Beta (P_gas/P_tot):     {result['beta_mean']:14.6f}")


def demo_cooling():
    """Demonstrate SimpleCooling."""
    print("\n" + "=" * 70)
    print("PHASE 3 DEMO: Radiative Cooling")
    print("=" * 70)

    particles = create_test_particles(N=200)
    eos = RadiationGas()
    cooling_models = ['free_free', 'blackbody', 'power_law']

    rho = particles['density']
    u = particles['internal_energy']
    masses = particles['masses']

    # Compute temperature
    T = eos.temperature(rho, u)

    print("\nCooling Model Comparison:")
    print(f"{'Model':>15} {'Luminosity (erg/s)':>20} {'Min t_cool (s)':>25}")
    print("-" * 70)

    for model in cooling_models:
        cooling = SimpleCooling(cooling_model=model)

        # Compute luminosity
        L = cooling.luminosity(rho, T, u, masses, smoothing_length=particles['smoothing_length'])

        # Compute cooling timescale
        t_cool = cooling.cooling_timescale(rho, T, u, smoothing_length=particles['smoothing_length'])
        finite_tcool = t_cool[np.isfinite(t_cool)]
        if len(finite_tcool) > 0:
            t_cool_min = np.min(finite_tcool)
            t_cool_str = f"{t_cool_min:18.6e}"
        else:
            t_cool_str = "      infinite/negligible"

        print(f"{model:>15} {L:20.6e} {t_cool_str:>25}")

    # Demonstrate cooling application
    print("\nApplying free-free cooling over dt = 1e6 s:")
    cooling = SimpleCooling(cooling_model='free_free')

    u_initial = u.copy()
    u_final = cooling.apply_cooling(rho, T, u_initial, eos, dt=1e6)

    dE = np.sum(masses * (u_final - u_initial))
    L_avg = -dE / 1e6

    print(f"  Initial internal energy: {np.sum(masses * u_initial):14.6e}")
    print(f"  Final internal energy:   {np.sum(masses * u_final):14.6e}")
    print(f"  Energy lost:             {-dE:14.6e}")
    print(f"  Average luminosity:      {L_avg:14.6e} erg/s")


def demo_diagnostics_writer():
    """Demonstrate DiagnosticsWriter."""
    print("\n" + "=" * 70)
    print("PHASE 3 DEMO: Diagnostics Writer")
    print("=" * 70)

    # Create output directory
    output_dir = Path("output/phase3_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting diagnostics to: {output_dir}")

    particles = create_test_particles(N=300)
    eos = RadiationGas()
    gravity_solver = NewtonianGravity()
    energy_diag = EnergyDiagnostics()
    cooling = SimpleCooling(cooling_model='free_free')

    # Initialize diagnostics writer
    with DiagnosticsWriter(output_dir, format='csv') as writer:
        writer.add_metadata('demo', 'Phase 3 Features')
        writer.add_metadata('N_particles', len(particles['masses']))

        # Simulate a few timesteps
        times = [0.0, 1e6, 2e6, 3e6, 4e6, 5e6]

        for t in times:
            # Compute energy diagnostics
            energy_dict = energy_diag.compute(particles, gravity_solver, eos, bh_mass=1.0)

            # Compute luminosity
            rho = particles['density']
            u = particles['internal_energy']
            T = eos.temperature(rho, u)
            masses = particles['masses']
            L = cooling.luminosity(rho, T, u, masses)

            # Compute fallback rate
            fb_info = compute_fallback_rate(particles, bh_mass=1.0, r_capture=10.0)

            # Write diagnostics
            writer.write_energy_diagnostic(t, energy_dict)
            writer.write_luminosity(t, L)
            writer.write_fallback_rate(
                t,
                fallback_rate=fb_info['bound_mass'] / (t + 1.0),  # Approximate
                bound_mass=fb_info['bound_mass']
            )

            # Apply some cooling for next step
            if t < times[-1]:
                u_new = cooling.apply_cooling(rho, T, u, eos, dt=1e6)
                particles['internal_energy'] = u_new

    print("  ✓ energy.csv written")
    print("  ✓ luminosity.csv written")
    print("  ✓ fallback.csv written")
    print("  ✓ metadata.json written")

    print("\nOutput files:")
    for file in sorted(output_dir.glob('*.csv')) + sorted(output_dir.glob('*.json')):
        size = file.stat().st_size
        print(f"  {file.name:20s}  ({size:5d} bytes)")


def main():
    """Run all Phase 3 demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "BLACK HOLE TDE-SPH: PHASE 3 DEMO" + " " * 20 + "║")
    print("║" + " " * 10 + "Radiation EOS • Energy Diagnostics • Cooling" + " " * 12 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        demo_radiation_eos()
        demo_energy_diagnostics()
        demo_cooling()
        demo_diagnostics_writer()

        print("\n" + "=" * 70)
        print("PHASE 3 DEMO COMPLETE")
        print("=" * 70)
        print("\nAll Phase 3 features demonstrated successfully!")
        print("\nNext steps:")
        print("  1. Review diagnostic outputs in output/phase3_demo/")
        print("  2. Run full test suite: pytest tests/test_phase3_energy.py")
        print("  3. Integrate Phase 3 components into Simulation orchestrator")

    except Exception as e:
        print(f"\nERROR during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
