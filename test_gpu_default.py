"""
Test GPU default auto-detection and has_cuda fix.
"""
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

print("=" * 60)
print("Testing GPU Default Auto-Detection")
print("=" * 60)

# Test 1: Check HAS_CUDA import
print("\n1. Checking HAS_CUDA import...")
try:
    from tde_sph.gpu import HAS_CUDA
    print(f"   ✓ HAS_CUDA imported successfully: {HAS_CUDA}")
except ImportError as e:
    print(f"   ✗ Failed to import HAS_CUDA: {e}")
    HAS_CUDA = False

# Test 2: Check GPU gravity import
print("\n2. Checking GPU gravity import...")
try:
    from tde_sph.gravity.barnes_hut_gpu import BarnesHutGravityGPU
    print(f"   ✓ BarnesHutGravityGPU imported successfully")
    HAS_GPU_GRAVITY = True
except ImportError as e:
    print(f"   ✗ Failed to import BarnesHutGravityGPU: {e}")
    HAS_GPU_GRAVITY = False

# Test 3: Check simulation default behavior
print("\n3. Checking Simulation GPU default behavior...")
try:
    from tde_sph.core import Simulation, SimulationConfig
    from tde_sph.sph import ParticleSystem
    from tde_sph.gravity import NewtonianGravity
    from tde_sph.eos import IdealGas
    from tde_sph.integration import LeapfrogIntegrator
    import numpy as np
    
    # Create minimal particle system
    n_particles = 10
    particles = ParticleSystem(
        n_particles=n_particles,
        positions=np.random.randn(n_particles, 3).astype(np.float32),
        velocities=np.random.randn(n_particles, 3).astype(np.float32),
        masses=np.ones(n_particles, dtype=np.float32),
        internal_energy=np.ones(n_particles, dtype=np.float32),
        smoothing_length=np.ones(n_particles, dtype=np.float32) * 0.1
    )
    
    # Initialize simulation WITHOUT specifying use_gpu (should auto-detect)
    config = SimulationConfig(verbose=False)
    sim = Simulation(
        particles=particles,
        gravity_solver=NewtonianGravity(),
        eos=IdealGas(gamma=5/3),
        integrator=LeapfrogIntegrator(),
        config=config
        # Note: use_gpu NOT specified - should default to HAS_CUDA
    )
    
    print(f"   ✓ Simulation initialized successfully")
    print(f"   - HAS_CUDA detected: {HAS_CUDA}")
    print(f"   - sim.use_gpu: {sim.use_gpu}")
    print(f"   - Expected: {HAS_CUDA} (should match)")
    
    if sim.use_gpu == HAS_CUDA:
        print(f"   ✓ GPU auto-detection working correctly!")
    else:
        print(f"   ✗ GPU auto-detection mismatch!")
        
    # Test explicit disable
    sim2 = Simulation(
        particles=particles,
        gravity_solver=NewtonianGravity(),
        eos=IdealGas(gamma=5/3),
        integrator=LeapfrogIntegrator(),
        config=config,
        use_gpu=False  # Explicitly disable
    )
    print(f"\n   - sim2.use_gpu (explicit False): {sim2.use_gpu}")
    if sim2.use_gpu == False:
        print(f"   ✓ Explicit GPU disable working correctly!")
    else:
        print(f"   ✗ Explicit GPU disable failed!")
        
except Exception as e:
    print(f"   ✗ Failed to test Simulation: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("GPU Default Auto-Detection Test Complete")
print("=" * 60)
