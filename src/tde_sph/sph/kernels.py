"""
SPH kernel functions for density estimation and gradient computation.

This module implements the M4 cubic spline kernel (Monaghan & Lattanzio 1985)
widely used in modern SPH codes. The kernel has compact support (radius 2h)
and C¹ continuity.

Design follows REQ-001 (SPH core) with vectorized numpy implementation
suitable for future GPU acceleration.
"""

import numpy as np
import numpy.typing as npt

# Type alias for clarity
NDArrayFloat = npt.NDArray[np.float32]


class CubicSplineKernel:
    """
    M4 cubic spline SPH kernel.

    The standard cubic spline kernel with compact support radius 2h,
    following Monaghan & Lattanzio (1985) [1]_ and widely used in
    modern SPH codes (Price 2012 [2]_, PHANTOM [3]_).

    The kernel is defined as:
        W(r, h) = σ_d / h^d × w(q)
    where q = r/h, d is dimension, and:
        w(q) = { 1 - (3/2)q² + (3/4)q³,     0 ≤ q < 1
               { (1/4)(2 - q)³,              1 ≤ q < 2
               { 0,                          q ≥ 2

    Normalization constants in 3D:
        σ_3 = 1/π

    References
    ----------
    .. [1] Monaghan, J. J., & Lattanzio, J. C. (1985),
           "A refined particle method for astrophysical problems",
           Astronomy and Astrophysics, 149, 135.
    .. [2] Price, D. J. (2012), "Smoothed particle hydrodynamics and
           magnetohydrodynamics", Journal of Computational Physics, 231, 759.
    .. [3] Price, D. J. et al. (2018), "PHANTOM: A Smoothed Particle
           Hydrodynamics and Magnetohydrodynamics Code for Astrophysics",
           PASA, 35, e031.
    """

    def __init__(self, dim: int = 3):
        """
        Initialize cubic spline kernel.

        Parameters
        ----------
        dim : int, optional
            Spatial dimension (1, 2, or 3). Default is 3.
        """
        self.dim = dim
        self.support_radius = 2.0  # Kernel has compact support at 2h

        # Normalization constants (Price 2012, Table 1)
        if dim == 1:
            self.sigma = 2.0 / 3.0
        elif dim == 2:
            self.sigma = 10.0 / (7.0 * np.pi)
        elif dim == 3:
            self.sigma = 1.0 / np.pi
        else:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 1, 2, or 3.")

    def w(self, q: NDArrayFloat) -> NDArrayFloat:
        """
        Compute dimensionless kernel function w(q).

        Parameters
        ----------
        q : NDArrayFloat
            Dimensionless distance r/h.

        Returns
        -------
        w : NDArrayFloat
            Kernel value (unnormalized).
        """
        w_vals = np.zeros_like(q, dtype=np.float32)

        # Region 1: 0 ≤ q < 1
        mask1 = (q >= 0.0) & (q < 1.0)
        q1 = q[mask1]
        w_vals[mask1] = 1.0 - 1.5 * q1**2 + 0.75 * q1**3

        # Region 2: 1 ≤ q < 2
        mask2 = (q >= 1.0) & (q < 2.0)
        q2 = q[mask2]
        w_vals[mask2] = 0.25 * (2.0 - q2)**3

        # Region 3: q ≥ 2 (zero, already set)

        return w_vals

    def dw_dq(self, q: NDArrayFloat) -> NDArrayFloat:
        """
        Compute derivative of dimensionless kernel dw/dq.

        Parameters
        ----------
        q : NDArrayFloat
            Dimensionless distance r/h.

        Returns
        -------
        dw : NDArrayFloat
            Derivative dw/dq.
        """
        dw = np.zeros_like(q, dtype=np.float32)

        # Region 1: 0 ≤ q < 1
        mask1 = (q >= 0.0) & (q < 1.0)
        q1 = q[mask1]
        dw[mask1] = -3.0 * q1 + 2.25 * q1**2

        # Region 2: 1 ≤ q < 2
        mask2 = (q >= 1.0) & (q < 2.0)
        q2 = q[mask2]
        dw[mask2] = -0.75 * (2.0 - q2)**2

        # Region 3: q ≥ 2 (zero, already set)

        return dw

    def kernel(self, r: NDArrayFloat, h: NDArrayFloat) -> NDArrayFloat:
        """
        Compute SPH kernel W(r, h).

        Parameters
        ----------
        r : NDArrayFloat
            Distance(s) between particles (scalar or array).
        h : NDArrayFloat
            Smoothing length(s) (scalar or array, must broadcast with r).

        Returns
        -------
        W : NDArrayFloat
            Kernel value W(r, h) = σ_d / h^d × w(r/h).

        Notes
        -----
        This function is vectorized and broadcasts over input arrays.
        For element-wise operations with different h per particle,
        ensure shapes are compatible.
        """
        # Ensure float32 for GPU compatibility
        r = np.asarray(r, dtype=np.float32)
        h = np.asarray(h, dtype=np.float32)

        # Avoid division by zero
        h = np.maximum(h, 1e-10)

        q = r / h
        w_val = self.w(q)

        # W(r, h) = σ / h^d × w(q)
        normalization = self.sigma / h**self.dim
        return normalization * w_val

    def kernel_gradient(
        self,
        r_vec: NDArrayFloat,
        h: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute gradient of kernel ∇W(r, h).

        Parameters
        ----------
        r_vec : NDArrayFloat, shape (..., 3)
            Position vector(s) between particles (r_ij = r_i - r_j).
        h : NDArrayFloat, shape (...)
            Smoothing length(s), must broadcast with r_vec.

        Returns
        -------
        grad_W : NDArrayFloat, shape (..., 3)
            Gradient vector ∇W = (dW/dr) × (r_vec / r).

        Notes
        -----
        The gradient is computed as:
            ∇W(r, h) = (dW/dr) × r̂
        where:
            dW/dr = (σ / h^(d+1)) × (dw/dq)
            r̂ = r_vec / r
        """
        # Ensure float32
        r_vec = np.asarray(r_vec, dtype=np.float32)
        h = np.asarray(h, dtype=np.float32)

        # Avoid division by zero
        h = np.maximum(h, 1e-10)

        # Compute distance r = |r_vec|
        r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        r = np.maximum(r, 1e-10)  # Avoid division by zero

        # Unit vector r_hat = r_vec / r
        r_hat = r_vec / r

        # Dimensionless distance q = r / h
        q = (r / h[..., np.newaxis]).squeeze(-1)

        # Compute dw/dq
        dw = self.dw_dq(q)

        # dW/dr = (σ / h^(d+1)) × (dw/dq) × (1/h)
        # = (σ / h^(d+1)) × (dw/dq) / h
        # Simplified: (σ / h^(d+1)) × (dw/dq)
        dW_dr = (self.sigma / h**(self.dim + 1)) * dw

        # Gradient: ∇W = (dW/dr) × r_hat
        grad_W = dW_dr[..., np.newaxis] * r_hat

        return grad_W


# Convenience function for common operations
def compute_kernel_summation(
    positions_i: NDArrayFloat,
    positions_j: NDArrayFloat,
    h_i: NDArrayFloat,
    masses_j: NDArrayFloat,
    kernel: CubicSplineKernel
) -> NDArrayFloat:
    """
    Compute SPH density summation: ρ_i = ∑_j m_j W(|r_i - r_j|, h_i).

    This is a convenience function for density estimation using the
    standard SPH summation (Price 2012, Eq. 13).

    Parameters
    ----------
    positions_i : NDArrayFloat, shape (N_i, 3)
        Positions of particles where density is computed.
    positions_j : NDArrayFloat, shape (N_j, 3)
        Positions of neighbour particles.
    h_i : NDArrayFloat, shape (N_i,)
        Smoothing lengths of particles i.
    masses_j : NDArrayFloat, shape (N_j,)
        Masses of neighbour particles.
    kernel : CubicSplineKernel
        Kernel instance.

    Returns
    -------
    density : NDArrayFloat, shape (N_i,)
        Computed densities ρ_i.

    Notes
    -----
    This is a simple implementation for illustration. Production code
    should use neighbour lists to avoid O(N²) scaling.
    """
    n_i = positions_i.shape[0]
    n_j = positions_j.shape[0]
    density = np.zeros(n_i, dtype=np.float32)

    # For each particle i
    for i in range(n_i):
        # Compute distances to all j particles
        r_ij_vec = positions_i[i] - positions_j  # Shape (N_j, 3)
        r_ij = np.linalg.norm(r_ij_vec, axis=1)  # Shape (N_j,)

        # Compute kernel contributions
        W_ij = kernel.kernel(r_ij, h_i[i])

        # Sum: ρ_i = ∑_j m_j W_ij
        density[i] = np.sum(masses_j * W_ij)

    return density


# Default kernel instance for 3D
default_kernel = CubicSplineKernel(dim=3)
