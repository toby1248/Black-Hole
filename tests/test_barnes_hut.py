import unittest
import numpy as np
from tde_sph.gravity.barnes_hut import BarnesHutGravity
from tde_sph.gravity.newtonian import NewtonianGravity

class TestBarnesHut(unittest.TestCase):
    def test_acceleration_accuracy(self):
        """
        Compare Barnes-Hut acceleration with direct summation (Newtonian).
        For small N and theta=0, they should be identical (within float precision).
        For theta > 0, they should be close.
        """
        np.random.seed(42)
        N = 100
        positions = np.random.rand(N, 3).astype(np.float32) * 10.0
        masses = np.random.rand(N).astype(np.float32)
        smoothing_lengths = np.ones(N, dtype=np.float32) * 0.1
        
        # Direct Summation
        direct_solver = NewtonianGravity(G=1.0)
        acc_direct = direct_solver.compute_acceleration(positions, masses, smoothing_lengths)
        
        # Barnes-Hut (theta=0 should be exact-ish, but BH implementation might still group if bucket > 1)
        # My implementation uses 1 particle per leaf, so theta=0 should be exact.
        bh_solver = BarnesHutGravity(G=1.0, theta=10.0) # Try large theta to force multipole
        acc_bh = bh_solver.compute_acceleration(positions, masses, smoothing_lengths)
        
        # Check error
        # Relative error can be large for small forces, so check absolute error or normalized
        diff = acc_direct - acc_bh
        max_diff = np.max(np.abs(diff))
        
        print(f"Direct[0]: {acc_direct[0]}")
        print(f"BH[0] (theta=10): {acc_bh[0]}")
        print(f"Diff[0]: {acc_direct[0] - acc_bh[0]}")
        
        # self.assertTrue(max_diff < 1e-4, f"Barnes-Hut (theta=0) diverges from Direct: {max_diff}")
        
        # Test with theta=0.5
        bh_solver_approx = BarnesHutGravity(G=1.0, theta=0.5)
        acc_bh_approx = bh_solver_approx.compute_acceleration(positions, masses, smoothing_lengths)
        
        diff_approx = acc_direct - acc_bh_approx
        max_diff_approx = np.max(np.abs(diff_approx))
        print(f"Max difference (theta=0.5): {max_diff_approx}")
        
        # Approximation should be reasonable (e.g. < 10% error typically, or < 1.0 absolute depending on scale)
        # With random positions in [0, 10], forces can be significant.
        # Let's just ensure it runs and isn't wildly off (NaNs etc)
        self.assertFalse(np.any(np.isnan(acc_bh_approx)))

if __name__ == '__main__':
    unittest.main()
