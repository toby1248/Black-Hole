"""
Test GUI simulation thread HAS_CUDA fix.
"""
import sys
from pathlib import Path

# Add gui and src to path
gui_dir = Path(__file__).parent / "gui"
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(gui_dir))
sys.path.insert(0, str(src_dir))

print("=" * 60)
print("Testing GUI SimulationThread HAS_CUDA Fix")
print("=" * 60)

# Test 1: Check imports in simulation_thread
print("\n1. Checking simulation_thread imports...")
try:
    from simulation_thread import (
        HAS_CUDA, 
        HAS_GPU_GRAVITY,
        SimulationThread
    )
    print(f"   ✓ HAS_CUDA imported from simulation_thread: {HAS_CUDA}")
    print(f"   ✓ HAS_GPU_GRAVITY imported from simulation_thread: {HAS_GPU_GRAVITY}")
except ImportError as e:
    print(f"   ✗ Failed to import from simulation_thread: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create a minimal config and verify thread can initialize
print("\n2. Testing SimulationThread initialization...")
try:
    minimal_config = {
        'simulation': {
            'mode': 'Newtonian',
            'output_dir': 'outputs/test',
            'use_gpu': True
        },
        'black_hole': {
            'mass': 1e6,
            'spin': 0.0
        },
        'star': {
            'mass': 1.0,
            'radius': 1.0,
            'polytropic_index': 3
        },
        'orbit': {
            'pericentre': 10.0,
            'eccentricity': 0.95,
            'inclination': 0.0,
            'starting_distance': 3.0
        },
        'particles': {
            'count': 100  # Small for quick test
        },
        'integration': {
            'timestep': 0.01,
            't_end': 0.1,  # Very short
            'output_interval': 0.05
        },
        'physics': {
            'gravity_type': 'barnes_hut',
            'theta': 0.5
        },
        'seed': 42
    }
    
    thread = SimulationThread(minimal_config)
    print(f"   ✓ SimulationThread created successfully")
    print(f"   - HAS_CUDA available: {HAS_CUDA}")
    print(f"   - HAS_GPU_GRAVITY available: {HAS_GPU_GRAVITY}")
    
    # Test the _log_acceleration_capabilities method
    print("\n3. Testing _log_acceleration_capabilities...")
    messages = []
    thread.log_message.connect(lambda msg: messages.append(msg))
    thread._log_acceleration_capabilities()
    
    print("   Logged capabilities:")
    for msg in messages:
        print(f"     {msg}")
    
    print(f"\n   ✓ _log_acceleration_capabilities executed successfully")
    
except Exception as e:
    print(f"   ✗ Failed to test SimulationThread: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All GUI SimulationThread tests passed!")
print("=" * 60)
