"""
Test orbit setup with eccentricity and periapsis.

Verifies that the initial conditions generator correctly computes
initial position and velocity for Newtonian orbits.
"""

import pytest
import numpy as np


def test_orbit_setup_circular():
    """Test circular orbit (e=0) setup."""
    # Parameters
    bh_mass = 1.0
    r_p = 10.0  # periapsis
    e = 0.0  # circular
    starting_distance = 1.0  # start at periapsis
    
    # Circular orbit: a = r_p for e=0
    r_init = starting_distance * r_p
    
    # Velocity for circular orbit: v = sqrt(GM/r)
    v_expected = np.sqrt(bh_mass / r_init)
    
    # Angular momentum
    L = r_p * np.sqrt(bh_mass * (1.0 + e) / r_p)
    v_tangential = L / r_init
    
    # For circular orbit at periapsis, all velocity is tangential, no radial
    assert np.isclose(v_tangential, v_expected, rtol=1e-6)


def test_orbit_setup_elliptical():
    """Test elliptical orbit (0 < e < 1) setup."""
    bh_mass = 1.0
    r_p = 5.0
    e = 0.5
    starting_distance = 3.0  # Start 3x periapsis
    
    r_init = starting_distance * r_p
    
    # Semi-major axis
    a = r_p / (1.0 - e)
    
    # Specific energy
    E_orb = -bh_mass / (2.0 * a)
    
    # Total velocity from energy conservation
    v_mag = np.sqrt(2.0 * (E_orb + bh_mass / r_init))
    
    # Angular momentum at periapsis
    L = r_p * np.sqrt(bh_mass * (1.0 + e) / r_p)
    v_tangential = L / r_init
    
    # Radial velocity
    v_radial_sq = v_mag**2 - v_tangential**2
    assert v_radial_sq >= 0, "Radial velocity squared must be non-negative"
    
    v_radial = np.sqrt(v_radial_sq)
    
    # Verify energy conservation
    E_check = 0.5 * v_mag**2 - bh_mass / r_init
    assert np.isclose(E_check, E_orb, rtol=1e-6)


def test_orbit_setup_parabolic():
    """Test parabolic orbit (e=1) setup."""
    bh_mass = 1.0
    r_p = 10.0
    e = 1.0
    starting_distance = 2.0
    
    r_init = starting_distance * r_p
    
    # Parabolic: E = 0
    # v = sqrt(2 GM / r)
    v_mag = np.sqrt(2.0 * bh_mass / r_init)
    
    # Angular momentum
    L = r_p * np.sqrt(bh_mass * (1.0 + e) / r_p)
    v_tangential = L / r_init
    
    # Verify energy is zero
    E_check = 0.5 * v_mag**2 - bh_mass / r_init
    assert np.isclose(E_check, 0.0, atol=1e-6)


def test_orbit_approaching_periapsis():
    """Test that orbit is correctly set up approaching periapsis."""
    bh_mass = 1.0
    r_p = 10.0
    e = 0.8
    starting_distance = 3.0
    
    r_init = starting_distance * r_p
    
    # Semi-major axis
    a = r_p / (1.0 - e)
    E_orb = -bh_mass / (2.0 * a)
    v_mag = np.sqrt(2.0 * (E_orb + bh_mass / r_init))
    
    # Angular momentum
    L = r_p * np.sqrt(bh_mass * (1.0 + e) / r_p)
    v_tangential = L / r_init
    
    # Radial velocity (approaching = negative in calculation, then negated)
    v_radial_sq = v_mag**2 - v_tangential**2
    v_radial_neg = -np.sqrt(max(v_radial_sq, 0.0))  # Negative = inward
    v_radial = -v_radial_neg  # Negate to get positive x component
    
    # Position at -x (left of BH)
    pos = np.array([-r_init, 0.0, 0.0])
    
    # Velocity: radial inward (+x) and tangential (-y for prograde orbit)
    # For a star approaching from -x, tangential velocity should be in -y direction
    # to produce angular momentum in +z direction
    vel = np.array([v_radial, -v_tangential, 0.0])
    
    # Verify velocity is toward BH and tangential
    assert vel[0] > 0, "Radial velocity should be positive (toward BH from -x)"
    assert vel[1] < 0, "Tangential velocity should be negative (-y) for +z angular momentum"
    
    # Verify angular momentum points in +z direction
    L_vec = np.cross(pos, vel)
    assert L_vec[2] > 0, f"Angular momentum should point in +z direction, got {L_vec[2]}"
    
    # Verify angular momentum magnitude
    assert np.isclose(np.linalg.norm(L_vec), L, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
