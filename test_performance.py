"""
Test performance of adaptive smoothing length updates.
"""
import yaml
import time
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gui.simulation_thread import SimulationThread

def test_performance(n_particles):
    # Load config
    with open('configs/autogen2.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure for test
    config['particles']['count'] = n_particles
    config['integration']['t_end'] = 0.01
    
    # Create simulation thread
    thread = SimulationThread(config)
    
    # Initialize simulation
    print(f'\nInitializing simulation with N={n_particles}...')
    sim = thread._initialize_simulation(config)
    print(f'Gravity solver: {type(sim.gravity_solver).__name__}')
    
    # Time steps
    times = []
    for i in range(5):
        start = time.time()
        sim.step()
        elapsed = time.time() - start
        times.append(elapsed * 1000)
        if i < 2:
            print(f'Step {i+1}: {elapsed*1000:.2f} ms')
    
    avg_time = np.mean(times[1:])  # Skip first for warmup
    print(f'\nAverage step time (N={n_particles}): {avg_time:.2f} ms')
    print('Timing breakdown (last step):')
    print(f'  Gravity: {sim.state.timing_gravity*1000:.2f} ms')
    print(f'  Density: {sim.state.timing_sph_density*1000:.2f} ms')
    print(f'  Smoothing: {sim.state.timing_smoothing_lengths*1000:.2f} ms')
    print(f'  Pressure: {sim.state.timing_sph_pressure*1000:.2f} ms')
    print(f'  Total: {sim.state.timing_total*1000:.2f} ms')
    
    return avg_time

if __name__ == '__main__':
    # Test with increasing particle counts
    for n in [10000, 20000, 50000]:
        avg = test_performance(n)
        print('='*60)
        
        if n == 100000:
            if avg < 10:
                print(f'\n✓ TARGET MET: {avg:.2f}ms < 10ms per step!')
            else:
                print(f'\n✗ Target not met ({avg:.2f} ms > 10ms)')
