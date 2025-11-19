import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

class GPUManager:
    """
    Manages data transfer and storage on the GPU.
    """
    def __init__(self, particles):
        if cp is None:
            raise RuntimeError("CuPy is not installed.")
        
        self.n_particles = particles.n_particles
        
        # Allocate GPU arrays
        self.pos = cp.asarray(particles.positions, dtype=cp.float32)
        self.vel = cp.asarray(particles.velocities, dtype=cp.float32)
        self.mass = cp.asarray(particles.masses, dtype=cp.float32)
        self.h = cp.asarray(particles.smoothing_lengths, dtype=cp.float32)
        self.u = cp.asarray(particles.internal_energy, dtype=cp.float32)
        self.rho = cp.asarray(particles.density, dtype=cp.float32)
        self.pressure = cp.asarray(particles.pressure, dtype=cp.float32)
        self.cs = cp.asarray(particles.sound_speed, dtype=cp.float32)
        
        # Force arrays
        self.acc_grav = cp.zeros_like(self.pos)
        self.acc_hydro = cp.zeros_like(self.pos)
        self.du_dt = cp.zeros(self.n_particles, dtype=cp.float32)
        
        # Neighbour data (for brute force, we might not store lists, just compute on fly)
        # But for density and hydro, we need neighbours.
        # For GPU, we often recompute neighbours or use a grid.
        # We'll implement a simple neighbour list or just direct summation first.
        
    def sync_to_host(self, particles):
        """Copy data back to CPU particles object."""
        particles.positions = cp.asnumpy(self.pos)
        particles.velocities = cp.asnumpy(self.vel)
        particles.masses = cp.asnumpy(self.mass)
        particles.smoothing_lengths = cp.asnumpy(self.h)
        particles.internal_energy = cp.asnumpy(self.u)
        particles.density = cp.asnumpy(self.rho)
        particles.pressure = cp.asnumpy(self.pressure)
        particles.sound_speed = cp.asnumpy(self.cs)
        
    def sync_to_device(self, particles):
        """Copy data from CPU particles object to GPU."""
        self.pos = cp.asarray(particles.positions, dtype=cp.float32)
        self.vel = cp.asarray(particles.velocities, dtype=cp.float32)
        self.mass = cp.asarray(particles.masses, dtype=cp.float32)
        self.h = cp.asarray(particles.smoothing_lengths, dtype=cp.float32)
        self.u = cp.asarray(particles.internal_energy, dtype=cp.float32)
        # Derived quantities might not need sync if computed on GPU
