#!/usr/bin/env python
"""
Demo script for TDE-SPH I/O and visualization modules.

This script demonstrates:
1. Creating particle data
2. Saving snapshots to HDF5
3. Reading snapshots back
4. Creating 3D visualizations
5. Creating animations

Run with: python examples/demo_io_visualization.py
"""

import numpy as np
from pathlib import Path

from tde_sph.io import HDF5Writer, write_snapshot, read_snapshot
from tde_sph.visualization import Plotly3DVisualizer, quick_plot

 
def create_expanding_sphere(n_particles, time, expansion_rate=0.5):
    """
    Create a simple expanding sphere of particles.

    Parameters
    ----------
    n_particles : int
        Number of particles
    time : float
        Current time (controls expansion)
    expansion_rate : float
        Rate of expansion

    Returns
    -------
    particles : dict
        Dictionary of particle arrays
    """
    # Random positions on sphere
    theta = np.random.uniform(0, np.pi, n_particles)
    phi = np.random.uniform(0, 2*np.pi, n_particles)
    radius = 1.0 + time * expansion_rate

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    positions = np.stack([x, y, z], axis=1).astype(np.float32)

    # Radial velocities
    velocities = positions * expansion_rate

    # Density falls off with expansion
    density = np.ones(n_particles, dtype=np.float32) * 1000.0 / (1.0 + time)**2

    # Temperature increases toward center
    r = np.sqrt(x**2 + y**2 + z**2)
    temperature = 10000.0 * np.exp(-r / radius).astype(np.float32)

    particles = {
        'positions': positions,
        'velocities': velocities,
        'masses': np.ones(n_particles, dtype=np.float32) * 1e-5,
        'density': density,
        'internal_energy': temperature * 100.0,  # Simplified
        'smoothing_length': np.ones(n_particles, dtype=np.float32) * 0.1,
        'temperature': temperature,
    }

    return particles


def main():
    """Run the demo."""
    print("="*60)
    print("TDE-SPH I/O and Visualization Demo")
    print("="*60)

    # Setup
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    n_particles = 5000
    n_snapshots = 5

    # Create HDF5 writer
    writer = HDF5Writer(compression='gzip', compression_level=4)

    # Metadata for the simulation
    metadata = {
        'simulation_name': 'expanding_sphere_demo',
        'n_particles': n_particles,
        'expansion_rate': 0.5,
    }

    print(f"\nCreating {n_snapshots} snapshots with {n_particles:,} particles each...")

    # Create and save snapshots
    snapshot_files = []
    snapshots_for_animation = []

    for i in range(n_snapshots):
        time = float(i) * 0.5  # Time step of 0.5

        # Create particle data
        particles = create_expanding_sphere(n_particles, time)

        # Save snapshot
        filename = output_dir / f"snapshot_{i:04d}.h5"
        writer.write_snapshot(str(filename), particles, time=time, metadata=metadata)
        snapshot_files.append(filename)

        print(f"  Snapshot {i}: t={time:.2f}, saved to {filename.name}")

        # Store for animation
        snapshots_for_animation.append({
            'positions': particles['positions'],
            'quantities': {
                'density': particles['density'],
                'temperature': particles['temperature'],
            },
            'time': time
        })

    # Read back one snapshot to verify
    print("\nReading snapshot 0 to verify I/O...")
    data = read_snapshot(str(snapshot_files[0]))
    print(f"  Time: {data['time']}")
    print(f"  Particles: {data['n_particles']:,}")
    print(f"  Fields: {list(data['particles'].keys())}")

    # Create visualization
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)

    viz = Plotly3DVisualizer(max_particles_plot=5000)

    # 1. Single snapshot visualization colored by density
    print("\n1. Creating density visualization...")
    fig_density = viz.plot_particles(
        data['particles']['positions'],
        quantities={
            'density': data['particles']['density'],
            'temperature': data['particles']['temperature'],
        },
        color_by='density',
        log_scale=True,
        title=f"Expanding Sphere - Density (t={data['time']:.2f})"
    )

    html_file = output_dir / "density_visualization.html"
    fig_density.write_html(str(html_file))
    print(f"   Saved to: {html_file}")

    # 2. Temperature visualization
    print("\n2. Creating temperature visualization...")
    fig_temp = viz.plot_particles(
        data['particles']['positions'],
        quantities={'temperature': data['particles']['temperature']},
        color_by='temperature',
        log_scale=False,
        colorscale='Hot',
        title=f"Expanding Sphere - Temperature (t={data['time']:.2f})"
    )

    html_file = output_dir / "temperature_visualization.html"
    fig_temp.write_html(str(html_file))
    print(f"   Saved to: {html_file}")

    # 3. Quick plot example
    print("\n3. Creating quick plot...")
    fig_quick = quick_plot(
        data['particles']['positions'],
        color_by=data['particles']['density'],
        title="Quick Plot Example"
    )

    html_file = output_dir / "quick_plot.html"
    fig_quick.write_html(str(html_file))
    print(f"   Saved to: {html_file}")

    # 4. Create animation
    print("\n4. Creating animation (this may take a moment)...")
    fig_anim = viz.animate(
        snapshots_for_animation,
        color_by='density',
        log_scale=True,
        title="Expanding Sphere Evolution",
        frame_duration=500,  # 500ms per frame
    )

    html_file = output_dir / "evolution_animation.html"
    fig_anim.write_html(str(html_file))
    print(f"   Saved to: {html_file}")
    print(f"   Open in browser to view interactive animation with time slider")

    # Summary
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nFiles created:")
    print("  Snapshots:")
    for f in snapshot_files:
        size_kb = f.stat().st_size / 1024
        print(f"    - {f.name} ({size_kb:.1f} KB)")
    print("  Visualizations:")
    print("    - density_visualization.html")
    print("    - temperature_visualization.html")
    print("    - quick_plot.html")
    print("    - evolution_animation.html")

    print("\nTo view visualizations:")
    print(f"  Open {output_dir.absolute()}/*.html in your web browser")

    print("\nNext steps:")
    print("  - Modify create_expanding_sphere() to test different scenarios")
    print("  - Try different colorscales (Viridis, Plasma, Jet, etc.)")
    print("  - Experiment with marker sizes and downsampling")
    print("  - Export to PNG/PDF using fig.write_image() (requires kaleido)")


if __name__ == '__main__':
    main()
