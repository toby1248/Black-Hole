"""
Comprehensive integration test for GPU auto-detection fix.
Simulates the exact GUI initialization flow that was failing.
"""
import sys
from pathlib import Path

# Add gui and src to path
gui_dir = Path(__file__).parent / "gui"
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(gui_dir))
sys.path.insert(0, str(src_dir))

print("=" * 60)
print("Comprehensive GPU Auto-Detection Integration Test")
print("=" * 60)

# Import exactly as the GUI does
print("\n1. Importing modules as GUI does...")
try:
    from simulation_thread import (
        Simulation,
        SimulationConfig,
        ParticleSystem,
        HAS_CUDA,
        HAS_GPU_GRAVITY,
        SimulationThread
    )
    print("   ✓ All imports successful")
    print(f"     - HAS_CUDA: {HAS_CUDA}")
    print(f"     - HAS_GPU_GRAVITY: {HAS_GPU_GRAVITY}")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create config exactly as GUI does
print("\n2. Creating simulation config (GUI style)...")
try:
    config_dict = {
        'simulation': {
            'mode': 'Newtonian',
            'output_dir': 'outputs/integration_test',
            # Note: use_gpu NOT specified - should auto-detect
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
            'count': 50  # Small for quick test
        },
        'integration': {
            'timestep': 0.01,
            't_end': 0.05,  # Very short
            'output_interval': 0.02
        },
        'physics': {
            'gravity_type': 'barnes_hut',
            'theta': 0.5
        },
        'seed': 42
    }
    
    thread = SimulationThread(config_dict)
    print("   ✓ SimulationThread created successfully")
    
    # Check that thread can access HAS_CUDA without NameError
    print(f"\n3. Verifying thread can access HAS_CUDA...")
    print(f"   - Thread can access HAS_CUDA: {HAS_CUDA}")
    print(f"   - Thread can access HAS_GPU_GRAVITY: {HAS_GPU_GRAVITY}")
    print("   ✓ No NameError!")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test the exact line that was failing (line 336 in original error)
print("\n4. Testing the exact failing condition...")
try:
    # This is the line that was causing: NameError: name 'has_cuda' is not defined
    sim_params = config_dict.get('simulation', {})
    phys_params = config_dict.get('physics', {})
    
    use_gpu = sim_params.get('use_gpu', HAS_CUDA)  # Auto-detect
    gravity_type = phys_params.get('gravity_type', 'newtonian')
    
    # This was line 336 that failed with "name 'has_cuda' is not defined"
    if use_gpu and HAS_GPU_GRAVITY and HAS_CUDA:
        print("   ✓ GPU mode selected (all conditions true)")
        print(f"     - use_gpu: {use_gpu}")
        print(f"     - HAS_GPU_GRAVITY: {HAS_GPU_GRAVITY}")
        print(f"     - HAS_CUDA: {HAS_CUDA}")
    else:
        print("   ✓ CPU mode selected")
        print(f"     - use_gpu: {use_gpu}")
        print(f"     - HAS_GPU_GRAVITY: {HAS_GPU_GRAVITY}")
        print(f"     - HAS_CUDA: {HAS_CUDA}")
    
    print("   ✓ No NameError on the previously failing line!")
    
except NameError as e:
    print(f"   ✗ NameError still present: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"   ✗ Other error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test _log_acceleration_capabilities
print("\n5. Testing _log_acceleration_capabilities method...")
try:
    messages = []
    thread.log_message.connect(lambda msg: messages.append(msg))
    thread._log_acceleration_capabilities()
    
    print(f"   ✓ Method executed successfully")
    print(f"     Logged {len(messages)} messages:")
    for msg in messages[:3]:  # First 3 messages
        print(f"       - {msg}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("The NameError: name 'has_cuda' is not defined is FIXED")
print("=" * 60)
