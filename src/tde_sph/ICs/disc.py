"""
Accretion disc initial conditions generator.

Generates SPH particle distributions for accretion discs around black holes,
including thin Keplerian discs, thick tori, and tilted/warped configurations.

Supports both Newtonian and general relativistic (Kerr/Schwarzschild) disc models
for stream-disc collision studies and long-term accretion flow evolution.

References
----------
- Pringle, J. E. (1981), ARA&A, 19, 137
  "Accretion discs in astrophysics"
- Shakura, N. I. & Sunyaev, R. A. (1973), A&A, 24, 337
  "Black holes in binary systems. Observational appearance"
- Fishbone, L. G. & Moncrief, V. (1976), ApJ, 207, 962
  "Relativistic fluid disks in orbit around Kerr black holes"
- Bonnerot et al. (2023), MNRAS 522, 1269 [arXiv:2303.16230]
  "Spin-induced offset stream self-crossing shocks in TDEs"
- Liptai, D., Price, D. J. & Lodato, G. (2019), MNRAS 487, 4790
  "Disc formation from tidal disruption of stars on eccentric orbits"
"""

import numpy as np
from typing import Tuple, Optional, Literal
from tde_sph.core.interfaces import ICGenerator, NDArrayFloat, Metric
from tde_sph.metric import MinkowskiMetric, SchwarzschildMetric, KerrMetric
from tde_sph.metric.coordinates import cartesian_to_bl_spherical


class DiscGenerator(ICGenerator):
    """
    Generate initial conditions for accretion disc models.

    Supports thin Keplerian discs (α-disc prescription), thick pressure-supported
    tori (Fishbone-Moncrief), and tilted/warped configurations for stream-disc
    collision studies.

    Attributes
    ----------
    eta : float
        Smoothing length factor: h_i = η * (m_i/ρ_i)^(1/3).
    random_seed : Optional[int]
        Random seed for reproducible particle placement.
    metric : Optional[Metric]
        Spacetime metric for GR disc velocities (None = Newtonian).
    """

    def __init__(
        self,
        eta: float = 1.2,
        random_seed: Optional[int] = 42,
        metric: Optional[Metric] = None
    ):
        """
        Initialize disc generator.

        Parameters
        ----------
        eta : float, default 1.2
            SPH smoothing length factor (typically 1.2-1.5).
        random_seed : Optional[int], default 42
            Random seed for reproducible particle placement.
            Set to None for non-reproducible random placement.
        metric : Optional[Metric], default None
            Spacetime metric for GR disc models.
            If None, uses Newtonian velocities.
        """
        self.eta = np.float32(eta)
        self.random_seed = random_seed
        self.metric = metric if metric is not None else MinkowskiMetric()

    def generate(
        self,
        n_particles: int,
        **kwargs
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Generate accretion disc initial conditions.

        Dispatches to specific disc models based on disc_type parameter.

        Parameters
        ----------
        n_particles : int
            Number of SPH particles (10³ to 10⁷).
        **kwargs
            Model parameters (see individual methods for details):
            - disc_type : str, default "thin" - Disc model type
              Options: "thin" (Keplerian), "torus" (Fishbone-Moncrief), "tilted"
            - For thin disc: r_in, r_out, M_disc, surface_density_index, etc.
            - For torus: r_max, l_specific, etc.
            - For tilted: inclination, position_angle, etc.

        Returns
        -------
        positions : NDArrayFloat, shape (n_particles, 3)
            Particle positions.
        velocities : NDArrayFloat, shape (n_particles, 3)
            Particle velocities (Keplerian or GR circular orbits).
        masses : NDArrayFloat, shape (n_particles,)
            Particle masses.
        internal_energies : NDArrayFloat, shape (n_particles,)
            Specific internal energies.
        densities : NDArrayFloat, shape (n_particles,)
            Initial mass densities.

        Notes
        -----
        - Uses dimensionless units: G=1, M_BH=1 internally.
        - GR mode requires metric to be SchwarzschildMetric or KerrMetric.
        - Smoothing lengths computed via compute_smoothing_lengths() after generation.
        """
        disc_type = kwargs.get('disc_type', 'thin')

        if disc_type == 'thin':
            return self.generate_thin_disc(n_particles, **kwargs)
        elif disc_type == 'torus':
            return self.generate_torus(n_particles, **kwargs)
        elif disc_type == 'tilted':
            return self.generate_tilted_disc(n_particles, **kwargs)
        else:
            raise ValueError(f"Unknown disc_type: {disc_type}. Choose 'thin', 'torus', or 'tilted'.")

    def generate_thin_disc(
        self,
        n_particles: int,
        r_in: float = 6.0,
        r_out: float = 100.0,
        M_disc: float = 0.01,
        surface_density_index: float = 1.0,
        aspect_ratio: float = 0.05,
        temperature_index: float = 0.75,
        gamma: float = 5.0 / 3.0,
        M_bh: float = 1.0,
        **kwargs
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Generate thin Keplerian disc (α-disc model).

        Surface density: Σ(r) ∝ r^(-p)
        Vertical structure: H/r = h₀ (r/r₀)^(q) (hydrostatic equilibrium)
        Temperature profile: T(r) ∝ r^(-q_T)

        Parameters
        ----------
        n_particles : int
            Number of particles.
        r_in : float, default 6.0
            Inner disc radius (code units, ≥ 3 R_g for Schwarzschild ISCO).
        r_out : float, default 100.0
            Outer disc radius (code units).
        M_disc : float, default 0.01
            Total disc mass (M_BH units).
        surface_density_index : float, default 1.0
            Power-law index p in Σ ∝ r^(-p).
            Typical: 0.5-1.5 (Shakura-Sunyaev: p ≈ 0.75).
        aspect_ratio : float, default 0.05
            Disc thickness H/r at reference radius r_out.
            Typical thin disc: 0.01-0.1.
        temperature_index : float, default 0.75
            Power-law index q_T in T ∝ r^(-q_T).
            Shakura-Sunyaev: q_T = 0.75.
        gamma : float, default 5/3
            Adiabatic index for thermal structure.
        M_bh : float, default 1.0
            Black hole mass (code units, typically normalized to 1).
        **kwargs
            - position_offset : NDArrayFloat, shape (3,), default [0,0,0]
            - velocity_offset : NDArrayFloat, shape (3,), default [0,0,0]

        Returns
        -------
        positions, velocities, masses, internal_energies, densities
            See generate() docstring.

        Notes
        -----
        - Particles distributed in cylindrical coordinates (R, φ, z).
        - Radial distribution samples Σ(r) via inverse transform.
        - Vertical distribution follows Gaussian with H(r).
        - Velocities: Keplerian v_φ = √(GM/r) (Newtonian) or GR circular orbits.
        """
        # Set random seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Position and velocity offsets
        position_offset = np.array(kwargs.get('position_offset', [0.0, 0.0, 0.0]), dtype=np.float32)
        velocity_offset = np.array(kwargs.get('velocity_offset', [0.0, 0.0, 0.0]), dtype=np.float32)

        # Sample radii from surface density profile
        R_particles = self._sample_radii_from_surface_density(
            n_particles, r_in, r_out, surface_density_index
        )

        # Sample azimuthal angles uniformly
        phi_particles = np.random.uniform(0.0, 2.0 * np.pi, n_particles).astype(np.float32)

        # Compute vertical scale height H(r)
        # H/r = h₀ (r/r_out)^q, where q comes from temperature profile
        # For vertically isothermal: q = 0.5 * (3 - p) ≈ 1 for p=1
        q_vertical = 0.5 * (3.0 - surface_density_index)
        H_particles = aspect_ratio * R_particles * (R_particles / r_out)**(q_vertical - 1.0)

        # Sample vertical heights from Gaussian (hydrostatic equilibrium)
        # ρ(z) ∝ exp(-z²/(2H²))
        z_particles = np.random.normal(0.0, 1.0, n_particles).astype(np.float32) * H_particles

        # Convert to Cartesian coordinates
        x = R_particles * np.cos(phi_particles)
        y = R_particles * np.sin(phi_particles)
        z = z_particles

        positions = np.column_stack([x, y, z]).astype(np.float32) + position_offset[np.newaxis, :]

        # Compute Keplerian velocities
        velocities = self._compute_keplerian_velocities(
            R_particles, phi_particles, z_particles, M_bh
        )
        velocities += velocity_offset[np.newaxis, :]

        # Compute surface density Σ(r)
        # Σ(r) = Σ₀ (r / r_out)^(-p)
        # Normalize so ∫ 2π r Σ(r) dr = M_disc
        p = surface_density_index
        if p != 2.0:
            # ∫_{r_in}^{r_out} r^(1-p) dr = [r^(2-p) / (2-p)]_{r_in}^{r_out}
            integral = (r_out**(2.0 - p) - r_in**(2.0 - p)) / (2.0 - p)
        else:
            # p = 2: ∫ r^(-1) dr = ln(r)
            integral = np.log(r_out / r_in)

        Sigma_0 = M_disc / (2.0 * np.pi * r_out**(-p) * integral)
        Sigma_particles = Sigma_0 * (R_particles / r_out)**(-p)

        # Volume density: ρ(r, z) = Σ(r) / (√(2π) H(r)) * exp(-z²/(2H²))
        # For SPH particles, approximate midplane density
        sqrt_2pi = np.sqrt(2.0 * np.pi)
        rho_particles = (Sigma_particles / (sqrt_2pi * H_particles)) * np.exp(
            -0.5 * (z_particles / H_particles)**2
        )

        # Particle masses (uniform)
        masses = np.full(n_particles, M_disc / n_particles, dtype=np.float32)

        # Temperature profile T(r) ∝ r^(-q_T)
        # T(r) = T_out (r / r_out)^(-q_T)
        # For α-disc: T ∝ r^(-3/4), so q_T = 0.75
        q_T = temperature_index
        # Set T_out from aspect ratio: H/r = c_s / v_K ∝ √(T/M) / √(M/r) = √(T r / M)
        # => T ∝ (H/r)² * M / r
        # At r_out: T_out = (H/r)² * M_bh / r_out
        T_ref = aspect_ratio**2 * M_bh / r_out
        T_particles = T_ref * (R_particles / r_out)**(-q_T)

        # Internal energy from temperature: u = (k_B T) / [(γ-1) μ m_p]
        # In code units with k_B = 1, μ m_p = 1:
        u_particles = T_particles / (gamma - 1.0)

        return (
            positions.astype(np.float32),
            velocities.astype(np.float32),
            masses.astype(np.float32),
            u_particles.astype(np.float32),
            rho_particles.astype(np.float32)
        )

    def generate_torus(
        self,
        n_particles: int,
        r_max: float = 12.0,
        l_specific: Optional[float] = None,
        pressure_exponent: float = 1.5,
        M_torus: float = 0.1,
        gamma: float = 5.0 / 3.0,
        M_bh: float = 1.0,
        **kwargs
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Generate thick pressure-supported torus (Fishbone-Moncrief solution).

        Torus with constant specific angular momentum l = u^φ / u^t.
        Pressure-supported vertical structure, suitable for thick disc studies
        and GRMHD initial conditions.

        Parameters
        ----------
        n_particles : int
            Number of particles.
        r_max : float, default 12.0
            Radius of pressure maximum (code units).
        l_specific : Optional[float], default None
            Specific angular momentum (code units).
            If None, computed from Keplerian value at r_max.
        pressure_exponent : float, default 1.5
            Polytropic index n in P ∝ ρ^(1+1/n) (not same as adiabatic γ).
            Typical: 1.5 for moderately thick torus.
        M_torus : float, default 0.1
            Total torus mass (M_BH units).
        gamma : float, default 5/3
            Adiabatic index for thermal structure.
        M_bh : float, default 1.0
            Black hole mass (code units).
        **kwargs
            - r_in_torus : float, default 0.6 * r_max - Inner edge radius.
            - r_out_torus : float, default 2.0 * r_max - Outer edge radius.
            - position_offset, velocity_offset

        Returns
        -------
        positions, velocities, masses, internal_energies, densities

        Notes
        -----
        - Fishbone-Moncrief torus: constant angular momentum surfaces.
        - In Schwarzschild: l = √(GM r) at r_max for Keplerian.
        - Pressure P(r,θ) from enthalpy balance in rotating frame.
        - This is a simplified model; full FM solution requires iterative integration.
        """
        # Set random seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Position and velocity offsets
        position_offset = np.array(kwargs.get('position_offset', [0.0, 0.0, 0.0]), dtype=np.float32)
        velocity_offset = np.array(kwargs.get('velocity_offset', [0.0, 0.0, 0.0]), dtype=np.float32)

        # Inner and outer radii
        r_in_torus = kwargs.get('r_in_torus', 0.6 * r_max)
        r_out_torus = kwargs.get('r_out_torus', 2.0 * r_max)

        # Specific angular momentum
        if l_specific is None:
            # Keplerian at r_max
            l_specific = np.sqrt(M_bh * r_max)

        # Sample cylindrical radii and heights
        # For torus: sample R in [r_in_torus, r_out_torus], z from torus profile
        R_particles = np.random.uniform(r_in_torus, r_out_torus, n_particles).astype(np.float32)
        phi_particles = np.random.uniform(0.0, 2.0 * np.pi, n_particles).astype(np.float32)

        # Torus vertical structure: approximate with |z| < H(R)
        # H(R) peaks at R = r_max
        # Simplified: H(R) ∝ exp(-(R - r_max)² / σ_R²)
        sigma_R = 0.3 * r_max  # Width parameter
        H_max = 0.3 * r_max    # Maximum height at r_max
        H_particles = H_max * np.exp(-((R_particles - r_max) / sigma_R)**2)

        # Sample z from uniform distribution in [-H, H]
        z_particles = np.random.uniform(-1.0, 1.0, n_particles).astype(np.float32) * H_particles

        # Convert to Cartesian
        x = R_particles * np.cos(phi_particles)
        y = R_particles * np.sin(phi_particles)
        z = z_particles

        positions = np.column_stack([x, y, z]).astype(np.float32) + position_offset[np.newaxis, :]

        # Velocities: approximate with Keplerian at each R
        velocities = self._compute_keplerian_velocities(
            R_particles, phi_particles, z_particles, M_bh
        )
        velocities += velocity_offset[np.newaxis, :]

        # Density profile: ρ ∝ exp(-(R - r_max)²/σ_R² - z²/H²)
        rho_max = 1.0  # Normalized, will rescale
        rho_particles = rho_max * np.exp(
            -((R_particles - r_max) / sigma_R)**2 - (z_particles / H_particles)**2
        )

        # Rescale density to match total mass
        # M_total = Σ m_i, and ρ_i = m_i / V_i
        # For uniform mass particles: m_i = M_torus / N
        # => need to rescale ρ to be consistent
        # Simple approach: set masses uniformly, compute density scale factor
        masses = np.full(n_particles, M_torus / n_particles, dtype=np.float32)

        # Density normalization: ensure mean density consistent with mass/volume
        mean_rho_target = M_torus / (np.pi * (r_out_torus**2 - r_in_torus**2) * 2.0 * H_max)
        rho_scale = mean_rho_target / np.mean(rho_particles)
        rho_particles *= rho_scale

        # Internal energy from polytropic relation
        # P = K ρ^Γ, where Γ = 1 + 1/n_poly
        Gamma_poly = 1.0 + 1.0 / pressure_exponent
        K_poly = 0.1  # Polytropic constant (arbitrary units for torus)
        P_particles = K_poly * rho_particles**Gamma_poly
        u_particles = P_particles / ((gamma - 1.0) * rho_particles)

        return (
            positions.astype(np.float32),
            velocities.astype(np.float32),
            masses.astype(np.float32),
            u_particles.astype(np.float32),
            rho_particles.astype(np.float32)
        )

    def generate_tilted_disc(
        self,
        n_particles: int,
        inclination: float = 30.0,
        position_angle: float = 0.0,
        **kwargs
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Generate tilted/warped disc for stream-disc collision studies.

        Creates a thin Keplerian disc tilted by angle i relative to the
        equatorial plane (or BH spin axis in GR mode).

        Parameters
        ----------
        n_particles : int
            Number of particles.
        inclination : float, default 30.0
            Inclination angle in degrees (0° = equatorial, 90° = polar).
        position_angle : float, default 0.0
            Position angle (rotation around z-axis before tilting), degrees.
        **kwargs
            All parameters from generate_thin_disc(), plus:
            - warp_amplitude : float, default 0.0 - Amplitude of warping (radians).
            - warp_frequency : float, default 1.0 - Radial frequency of warping.

        Returns
        -------
        positions, velocities, masses, internal_energies, densities

        Notes
        -----
        - First generates disc in equatorial plane (z=0).
        - Applies rotation: R_PA(position_angle) * R_incl(inclination).
        - Velocities rotated accordingly.
        - Optional warping: twist angle φ_warp(r) = A sin(k r) added to orientation.
        """
        # Generate thin disc in equatorial plane
        pos_eq, vel_eq, masses, u, rho = self.generate_thin_disc(n_particles, **kwargs)

        # Remove any position/velocity offset (will add back after rotation)
        position_offset = np.array(kwargs.get('position_offset', [0.0, 0.0, 0.0]), dtype=np.float32)
        velocity_offset = np.array(kwargs.get('velocity_offset', [0.0, 0.0, 0.0]), dtype=np.float32)
        pos_eq -= position_offset[np.newaxis, :]
        vel_eq -= velocity_offset[np.newaxis, :]

        # Convert angles to radians
        i_rad = np.deg2rad(inclination)
        PA_rad = np.deg2rad(position_angle)

        # Optional warping
        warp_amplitude = kwargs.get('warp_amplitude', 0.0)
        warp_frequency = kwargs.get('warp_frequency', 1.0)

        if warp_amplitude > 0.0:
            # Compute radial distance
            R_particles = np.sqrt(pos_eq[:, 0]**2 + pos_eq[:, 1]**2)
            # Warp angle as function of radius
            phi_warp = warp_amplitude * np.sin(warp_frequency * R_particles)
            # Apply local rotation around z-axis
            cos_w = np.cos(phi_warp)
            sin_w = np.sin(phi_warp)
            x_warped = cos_w * pos_eq[:, 0] - sin_w * pos_eq[:, 1]
            y_warped = sin_w * pos_eq[:, 0] + cos_w * pos_eq[:, 1]
            pos_eq[:, 0] = x_warped
            pos_eq[:, 1] = y_warped
            # Velocities too
            vx_warped = cos_w * vel_eq[:, 0] - sin_w * vel_eq[:, 1]
            vy_warped = sin_w * vel_eq[:, 0] + cos_w * vel_eq[:, 1]
            vel_eq[:, 0] = vx_warped
            vel_eq[:, 1] = vy_warped

        # Rotation matrix: first PA around z, then inclination around x'
        # R = R_x(i) * R_z(PA)
        # R_z(PA):
        cos_PA = np.cos(PA_rad)
        sin_PA = np.sin(PA_rad)
        R_z = np.array([
            [cos_PA, -sin_PA, 0.0],
            [sin_PA,  cos_PA, 0.0],
            [0.0,     0.0,    1.0]
        ], dtype=np.float32)

        # R_x(i): Rotation around x-axis by angle i
        # Convention: positive i tilts +z toward +y (standard inclination)
        # So we use R_x(-i) to get the standard astronomical convention
        cos_i = np.cos(-i_rad)  # Note: negated for correct convention
        sin_i = np.sin(-i_rad)
        R_x = np.array([
            [1.0, 0.0,    0.0],
            [0.0, cos_i, -sin_i],
            [0.0, sin_i,  cos_i]
        ], dtype=np.float32)

        # Combined rotation
        R_total = R_x @ R_z

        # Rotate positions and velocities
        positions = (R_total @ pos_eq.T).T + position_offset[np.newaxis, :]
        velocities = (R_total @ vel_eq.T).T + velocity_offset[np.newaxis, :]

        return (
            positions.astype(np.float32),
            velocities.astype(np.float32),
            masses.astype(np.float32),
            u.astype(np.float32),
            rho.astype(np.float32)
        )

    def compute_smoothing_lengths(
        self,
        masses: NDArrayFloat,
        densities: NDArrayFloat
    ) -> NDArrayFloat:
        """
        Compute SPH smoothing lengths from masses and densities.

        h_i = η * (m_i / ρ_i)^(1/3)

        Parameters
        ----------
        masses : NDArrayFloat, shape (N,)
            Particle masses.
        densities : NDArrayFloat, shape (N,)
            Particle densities.

        Returns
        -------
        smoothing_lengths : NDArrayFloat, shape (N,)
            SPH smoothing lengths.
        """
        return self.eta * np.power(masses / densities, 1.0 / 3.0).astype(np.float32)

    def _sample_radii_from_surface_density(
        self,
        n_particles: int,
        r_in: float,
        r_out: float,
        p: float
    ) -> np.ndarray:
        """
        Sample radii from power-law surface density Σ ∝ r^(-p).

        Uses inverse transform sampling on cumulative mass M(r) ∝ ∫ r Σ(r) dr.

        Parameters
        ----------
        n_particles : int
            Number of particles.
        r_in : float
            Inner radius.
        r_out : float
            Outer radius.
        p : float
            Power-law index.

        Returns
        -------
        R_particles : np.ndarray, shape (n_particles,)
            Sampled radii.
        """
        # M(r) ∝ ∫_{r_in}^r r^(1-p) dr
        # For p ≠ 2: M(r) ∝ r^(2-p) / (2-p)
        # For p = 2: M(r) ∝ ln(r)

        u_samples = np.random.uniform(0.0, 1.0, n_particles)

        if np.abs(p - 2.0) > 1e-6:
            # General case: p ≠ 2
            # M(r) = [r^(2-p) - r_in^(2-p)] / (2-p)
            # M_total = [r_out^(2-p) - r_in^(2-p)] / (2-p)
            # u = M(r) / M_total = [r^(2-p) - r_in^(2-p)] / [r_out^(2-p) - r_in^(2-p)]
            # => r = [u (r_out^(2-p) - r_in^(2-p)) + r_in^(2-p)]^(1/(2-p))
            exponent = 2.0 - p
            R_particles = (
                u_samples * (r_out**exponent - r_in**exponent) + r_in**exponent
            )**(1.0 / exponent)
        else:
            # p = 2: M(r) ∝ ln(r/r_in)
            # u = ln(r/r_in) / ln(r_out/r_in)
            # => r = r_in * exp(u * ln(r_out/r_in))
            R_particles = r_in * np.exp(u_samples * np.log(r_out / r_in))

        return R_particles.astype(np.float32)

    def _compute_keplerian_velocities(
        self,
        R: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        M_bh: float
    ) -> np.ndarray:
        """
        Compute Keplerian velocities for circular orbits.

        For Newtonian: v_φ = √(GM/R)
        For GR (Schwarzschild/Kerr): uses metric to compute circular orbit velocity.

        Parameters
        ----------
        R : np.ndarray, shape (N,)
            Cylindrical radii.
        phi : np.ndarray, shape (N,)
            Azimuthal angles.
        z : np.ndarray, shape (N,)
            Vertical heights.
        M_bh : float
            Black hole mass.

        Returns
        -------
        velocities : np.ndarray, shape (N, 3)
            Cartesian velocities.
        """
        # Spherical radius (for GR, approximate as R for thin disc)
        r_sph = np.sqrt(R**2 + z**2)

        # Check if GR or Newtonian
        is_flat = isinstance(self.metric, MinkowskiMetric)

        if is_flat:
            # Newtonian: v_φ = √(GM/R)
            v_phi = np.sqrt(M_bh / R)
        else:
            # GR: circular orbit velocity from metric
            # For Schwarzschild at radius r: v_φ = √(M/(r - 2M))
            # For Kerr: more complex, depends on spin
            if isinstance(self.metric, SchwarzschildMetric):
                # Circular orbit in Schwarzschild (equatorial)
                # u^φ / u^t = √(M/r³) / √(1 - 3M/r)
                # v_φ = r u^φ / (γ u^t) where γ = 1/√(1-v²)
                # Simplified: v_φ ≈ √(M/r) for r >> 2M
                # More accurate: from geodesic equation
                r_safe = np.maximum(r_sph, 3.0)  # Ensure r > ISCO
                v_phi_sq = M_bh / r_safe / (1.0 - 3.0 * M_bh / r_safe)
                v_phi = np.sqrt(np.maximum(v_phi_sq, 0.0))
            elif isinstance(self.metric, KerrMetric):
                # Kerr circular orbits (simplified)
                # Full Kerr requires solving for ISCO and orbit properties
                # For now, use Schwarzschild approximation
                r_safe = np.maximum(r_sph, 3.0)
                v_phi_sq = M_bh / r_safe / (1.0 - 3.0 * M_bh / r_safe)
                v_phi = np.sqrt(np.maximum(v_phi_sq, 0.0))
            else:
                # Fallback to Newtonian
                v_phi = np.sqrt(M_bh / R)

        # Convert to Cartesian velocities
        vx = -v_phi * np.sin(phi)
        vy = v_phi * np.cos(phi)
        vz = np.zeros_like(v_phi)

        return np.column_stack([vx, vy, vz]).astype(np.float32)
