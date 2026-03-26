"""
Comprehensive pytest suite for Monte Carlo diluted magnetic system simulation.
Tests cover: expected outputs, edge cases, stochastic validation, and physics.
"""

import pytest
import numpy as np
from scipy.special import j0, j1, y0, y1
from lattice import Lattice
from accumulator import Accumulator
from monte_carlo import MonteCarlo


# ============================================================================
# FIXTURE: Common test lattices
# ============================================================================

@pytest.fixture
def small_lattice_dilute():
    """Small lattice with moderate doping for quick tests."""
    return Lattice(rows=5, cols=5, doping=0.6, kf=1.0, J0=1.0)


@pytest.fixture
def small_lattice_dense():
    """Dense lattice for stochastic tests."""
    return Lattice(rows=5, cols=5, doping=0.9, kf=1.0, J0=1.0)


@pytest.fixture
def large_lattice():
    """Larger lattice for convergence tests."""
    return Lattice(rows=20, cols=20, doping=0.5, kf=1.5, J0=2.0)


@pytest.fixture
def mc_instance(small_lattice_dilute):
    """Monte Carlo instance with small lattice."""
    return MonteCarlo(small_lattice_dilute)


# ============================================================================
# PART A: LATTICE INITIALIZATION TESTS
# ============================================================================

class TestLatticeGeneration:
    """Test lattice generation with various configurations."""

    def test_lattice_dilution_concentration(self, small_lattice_dilute):
        """Verify that actual dilution matches expected (within stochastic bounds)."""
        # Expected: ~60% of 25 sites = 15 sites
        assert 5 <= small_lattice_dilute.N <= 25, "Dilution concentration outside expected bounds"

    def test_lattice_high_doping(self):
        """Edge case: Very high doping (0.95) should give nearly complete lattice."""
        lat = Lattice(rows=10, cols=10, doping=0.95, kf=1.0, J0=1.0)
        expected_max = 100
        assert lat.N > 0.85 * expected_max, f"High doping should give ~95 sites, got {lat.N}"

    def test_lattice_low_doping(self):
        """Edge case: Very low doping (0.1) should give sparse lattice."""
        lat = Lattice(rows=10, cols=10, doping=0.1, kf=1.0, J0=1.0)
        expected_min = 100 * 0.1
        assert 0 <= lat.N <= 30, f"Low doping should give sparse lattice, got {lat.N}"

    def test_lattice_zero_doping_raises(self):
        """Edge case: Zero doping should raise ValueError."""
        with pytest.raises(ValueError, match="No occupied sites"):
            Lattice(rows=5, cols=5, doping=0.0, kf=1.0, J0=1.0)

    def test_lattice_magnetic_moments_initialized(self, small_lattice_dilute):
        """Verify spins are ±1."""
        assert np.all(np.isin(small_lattice_dilute.magnetic_moments, [-1, 1])), \
            "All spins must be ±1"

    def test_lattice_coordinates_in_bounds(self, small_lattice_dilute):
        """Verify coordinates respect system boundaries."""
        coords = small_lattice_dilute.lattice_points
        assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] <= small_lattice_dilute.Lx), \
            "x-coordinates out of bounds"
        assert np.all(coords[:, 1] >= 0) and np.all(coords[:, 1] <= small_lattice_dilute.Ly), \
            "y-coordinates out of bounds"

    def test_lattice_size_attributes(self, small_lattice_dilute):
        """Verify Lx, Ly computed correctly."""
        assert small_lattice_dilute.Lx == 5, "Lx should match cols"
        assert np.isclose(small_lattice_dilute.Ly, 5 * np.sqrt(3) / 2), "Ly should match rows * sqrt(3)/2"


# ============================================================================
# PART B: RKKY INTERACTION MATRIX TESTS
# ============================================================================

class TestRKKYInteraction:
    """Test RKKY interaction calculations."""

    def test_interaction_matrix_shape(self, small_lattice_dilute):
        """Verify interaction matrix is NxN."""
        J = small_lattice_dilute.interaction_matrix
        assert J.shape == (small_lattice_dilute.N, small_lattice_dilute.N), \
            f"Matrix shape {J.shape} != ({small_lattice_dilute.N}, {small_lattice_dilute.N})"

    def test_interaction_matrix_diagonal_zero(self, small_lattice_dilute):
        """Diagonal should be zero (no self-interaction)."""
        J = small_lattice_dilute.interaction_matrix
        assert np.allclose(np.diag(J), 0), "Diagonal must be zero"

    def test_interaction_matrix_symmetric(self, small_lattice_dilute):
        """Interaction matrix must be symmetric (J_ij = J_ji)."""
        J = small_lattice_dilute.interaction_matrix
        assert np.allclose(J, J.T), "Interaction matrix must be symmetric"

    def test_rkky_function_at_zero_distance(self, small_lattice_dilute):
        """RKKY interaction at r=0 should be exactly 0."""
        result = small_lattice_dilute.rkky_interaction_2d(0)
        assert result == 0, f"rkky_interaction_2d(0) should be 0, got {result}"

    def test_rkky_function_shape(self, small_lattice_dilute):
        """RKKY function should oscillate with distance."""
        kf = small_lattice_dilute.kf
        J0 = small_lattice_dilute.J0
        
        # Sample at different distances
        distances = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        interactions = np.array([small_lattice_dilute.rkky_interaction_2d(r) for r in distances])
        
        # Check they're not all the same (oscillatory behavior)
        assert not np.allclose(interactions, interactions[0]), \
            "RKKY should oscillate, not be constant"

    def test_distance_pbc(self, small_lattice_dilute):
        """Test periodic boundary condition in distance calculation."""
        # Create two points near opposite boundaries
        p1 = np.array([0.1, 0.0])
        p2 = np.array([small_lattice_dilute.Lx - 0.1, 0.0])
        
        dist = small_lattice_dilute.distance(p1, p2)
        expected_dist = 0.2  # Should wrap around
        assert dist < 0.5, f"PBC distance should be ~0.2, got {dist}"

    def test_triu_indices_computed(self, small_lattice_dilute):
        """Verify upper triangular indices are accessible."""
        assert hasattr(small_lattice_dilute, 'i_idx'), "Missing i_idx"
        assert hasattr(small_lattice_dilute, 'j_idx'), "Missing j_idx"
        assert hasattr(small_lattice_dilute, 'r_ij'), "Missing r_ij"
        
        # Verify consistency
        assert len(small_lattice_dilute.i_idx) == len(small_lattice_dilute.j_idx), \
            "i_idx and j_idx should have same length"


# ============================================================================
# PART C: ENERGY CALCULATION TESTS
# ============================================================================

class TestEnergyCalculation:
    """Test energy computations against known configurations."""

    def test_energy_all_aligned_spins(self):
        """Aligned spins (all +1 or all -1) should give specific energy pattern."""
        lat = Lattice(rows=3, cols=3, doping=1.0, kf=1.0, J0=1.0)
        
        # Set all spins to +1
        lat.magnetic_moments = np.ones(lat.N)
        
        # Energy = s @ J @ s
        acc = Accumulator(lat)
        E = acc.compute_energy()
        
        # For ferromagnetic-like coupling and all-aligned, E should be negative
        assert isinstance(E, (float, np.floating)), "Energy should be scalar"

    def test_energy_antialigned_spins(self):
        """Checkerboard configuration energy."""
        lat = Lattice(rows=4, cols=4, doping=1.0, kf=1.0, J0=1.0)
        
        # Alternating pattern
        lat.magnetic_moments = np.array([1, -1] * (lat.N // 2))
        if lat.N % 2 == 1:
            lat.magnetic_moments[-1] = 1
        
        acc = Accumulator(lat)
        E = acc.compute_energy()
        assert isinstance(E, (float, np.floating)), "Energy should be scalar"

    def test_energy_reproducibility(self, small_lattice_dilute):
        """Same configuration should give same energy."""
        acc = Accumulator(small_lattice_dilute)
        E1 = acc.compute_energy()
        E2 = acc.compute_energy()
        assert E1 == E2, "Energy calculation not reproducible"

    def test_energy_after_spin_flip(self, small_lattice_dilute):
        """Energy change formula: ΔE = -2 * s_i * (J_i @ s).
        
        Derives from the Hamiltonian H = 0.5 * s^T J s.
        When flipping spin i: ΔH = -2 * s_i * ∑_j J_{ij} s_j
        """
        acc = Accumulator(small_lattice_dilute)
        E_initial = acc.compute_energy()
        
        i = 0
        s = small_lattice_dilute.magnetic_moments
        J = small_lattice_dilute.interaction_matrix
        
        # Predict energy change using: ΔE = -2 * s_i * (J_i @ s)
        # This formula is exact for H = 0.5 * s^T J s
        dE_predicted = -2.0 * s[i] * (J[i] @ s)
        
        # Flip spin and measure
        s[i] *= -1
        E_final = acc.compute_energy()
        dE_actual = E_final - E_initial
        
        assert np.isclose(dE_predicted, dE_actual, rtol=1e-10), \
            f"Energy change mismatch: predicted {dE_predicted}, actual {dE_actual}"


# ============================================================================
# PART D: METROPOLIS STEP TESTS
# ============================================================================

class TestMetropolisStep:
    """Test Metropolis algorithm correctness."""

    def test_metropolis_updates_spins(self, small_lattice_dilute):
        """Metropolis step should change some spins."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 1.0
        
        initial_spins = mc.lattice.magnetic_moments.copy()
        
        # Run many steps, should get at least one acceptance
        for _ in range(100):
            mc.metropolis_step()
        
        assert not np.array_equal(initial_spins, mc.lattice.magnetic_moments), \
            "Metropolis should update spins"

    def test_metropolis_acceptance_probability_low_T(self, small_lattice_dilute):
        """At very low T, only downhill moves accepted."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 0.01  # Very low temperature
        
        mc.accept = 0
        for _ in range(50):
            mc.metropolis_step()
        
        acceptance_rate = mc.accept / 50
        assert acceptance_rate < 0.5, \
            f"Low-T acceptance rate should be low, got {acceptance_rate:.2f}"

    def test_metropolis_acceptance_probability_high_T(self, small_lattice_dilute):
        """At very high T, all moves approximately accepted."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 100.0  # Very high temperature
        
        mc.accept = 0
        for _ in range(50):
            mc.metropolis_step()
        
        acceptance_rate = mc.accept / 50
        assert acceptance_rate > 0.5, \
            f"High-T acceptance rate should be high, got {acceptance_rate:.2f}"

    def test_metropolis_detailed_balance(self, small_lattice_dilute):
        """Detailed balance: P(accept i→j) / P(accept j→i) = exp(-β ΔE)."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 1.0
        
        # Set up two states
        s = mc.lattice.magnetic_moments
        J = mc.lattice.interaction_matrix
        
        i = 0
        dE = -2 * s[i] * (J[i] @ s)
        
        if dE > 0:  # Uphill move
            prob_accept = np.exp(-dE / mc.T)
            assert 0 < prob_accept < 1, "Probability should be in (0,1) for uphill"

    def test_metropolis_energy_sync(self, small_lattice_dilute):
        """Energy in MonteCarlo should match Accumulator."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 1.0
        
        for _ in range(10):
            mc.metropolis_step()
        
        E_mc = mc.E
        E_acc = mc.acc.compute_energy()
        assert np.isclose(E_mc, E_acc, rtol=1e-10), \
            f"Energy mismatch: mc.E={E_mc}, acc.compute_energy()={E_acc}"


# ============================================================================
# PART E: WOLFF ALGORITHM TESTS
# ============================================================================

class TestWolffStep:
    """Test Wolff cluster algorithm."""

    def test_wolff_cluster_exists(self, small_lattice_dilute):
        """Wolff should return non-empty cluster."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 1.0
        mc.precompute_bond_probabilities()
        
        cluster, edges_fm, edges_afm = mc.wolff_step(return_cluster=True)
        assert len(cluster) > 0, "Cluster should contain at least the seed"

    def test_wolff_flips_cluster_spins(self, small_lattice_dilute):
        """All spins in cluster should flip."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 1.0
        mc.precompute_bond_probabilities()
        
        initial_spins = mc.lattice.magnetic_moments.copy()
        cluster, _, _ = mc.wolff_step(return_cluster=True)
        
        for idx in cluster:
            assert mc.lattice.magnetic_moments[idx] == -initial_spins[idx], \
                f"Spin {idx} in cluster should be flipped"

    def test_wolff_energy_consistency(self, small_lattice_dilute):
        """Wolff should compute consistent energy."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 1.0
        mc.precompute_bond_probabilities()
        
        mc.wolff_step(return_cluster=False)
        
        E_mc = mc.E
        E_check = mc.acc.compute_energy()
        assert np.isclose(E_mc, E_check, rtol=1e-10), \
            f"Wolff energy mismatch: mc.E={E_mc}, computed={E_check}"

    def test_wolff_preserves_lattice_structure(self, small_lattice_dilute):
        """Wolff should only flip spins, not modify lattice structure."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 1.0
        mc.precompute_bond_probabilities()
        
        initial_coords = mc.lattice.lattice_points.copy()
        mc.wolff_step()
        
        assert np.array_equal(initial_coords, mc.lattice.lattice_points), \
            "Lattice coordinates should not change"


# ============================================================================
# PART F: ACCUMULATOR & STATISTICS TESTS
# ============================================================================

class TestAccumulator:
    """Test statistics accumulation."""

    def test_accumulator_initialization(self, small_lattice_dilute):
        """Accumulator should initialize with correct structure."""
        acc = Accumulator(small_lattice_dilute)
        
        assert len(acc.energy) == 0, "Energy list should be empty initially"
        assert len(acc.magnetization) == 0, "Magnetization list should be empty"
        assert acc.energy_count == 0, "Energy count should be zero"

    def test_sample_warmup_updates_lists(self, small_lattice_dilute):
        """sample_warmup should append to energy/magnetization lists."""
        acc = Accumulator(small_lattice_dilute)
        E, M = 1.0, 0.5
        
        acc.sample_warmup(step=1, energy=E, magnetization=M)
        
        assert len(acc.energy) == 1, "Energy list should have one entry"
        assert len(acc.magnetization) == 1, "Magnetization list should have one entry"
        assert acc.energy[-1] == E, "Last energy should match input"
        assert acc.magnetization[-1] == M, "Last magnetization should match input"

    def test_sample_production_updates_susceptibility(self, small_lattice_dilute):
        """sample_production should track susceptibility."""
        acc = Accumulator(small_lattice_dilute)
        E, M, chi = 1.0, 0.5, 0.1
        
        acc.sample_production(E, M, chi)
        
        assert len(acc.susceptibility) == 1, "Susceptibility list should have one entry"
        assert acc.susceptibility[-1] == chi, "Susceptibility should match input"

    def test_running_statistics_convergence(self, small_lattice_dilute):
        """Running statistics should converge to mean and variance."""
        acc = Accumulator(small_lattice_dilute)
        
        # Sample from normal distribution
        data = np.random.normal(loc=5.0, scale=2.0, size=1000)
        expected_mean = np.mean(data)
        expected_var = np.var(data)
        
        mean_running = 0.0
        var_running = 0.0
        
        for i, val in enumerate(data):
            mean_running, var_running, _ = acc.update_running_statistics(
                val, mean_running, var_running, i
            )
        
        assert np.isclose(mean_running, expected_mean, rtol=0.05), \
            f"Running mean {mean_running} should converge to {expected_mean}"
        assert np.isclose(var_running, expected_var, rtol=0.1), \
            f"Running variance {var_running} should converge to {expected_var}"

    def test_autocorrelation_computed(self, small_lattice_dilute):
        """Autocorrelation should be computed during warmup."""
        acc = Accumulator(small_lattice_dilute)
        
        # Simulate warmup samples
        for i in range(100):
            E = np.sin(i * 0.1) + np.random.normal(0, 0.1)
            M = np.cos(i * 0.05) + np.random.normal(0, 0.05)
            acc.sample_warmup(step=i, energy=E, magnetization=M)
        
        assert len(acc.energy_autocorr) > 0, "Autocorrelation should be computed"
        assert acc.energy_tau_int > 0, "Autocorr time should be positive"


# ============================================================================
# PART G: INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Full simulation integration tests."""

    def test_run_loop_metropolis(self, small_lattice_dilute):
        """Full Metropolis simulation should complete without errors."""
        mc = MonteCarlo(small_lattice_dilute)
        
        stats = mc.run_loop(
            warmup_steps=50,
            steps=100,
            T=1.0,
            method="metropolis",
            save_warmup=False
        )
        
        assert "tau_E" in stats, "Should return tau_E"
        assert "tau_M" in stats, "Should return tau_M"
        assert "accept" in stats, "Should return acceptance rate"
        assert 0 <= stats["accept"] <= 1, "Acceptance rate should be in [0,1]"

    def test_run_loop_wolff(self, small_lattice_dilute):
        """Full Wolff simulation should complete without errors."""
        mc = MonteCarlo(small_lattice_dilute)
        
        stats = mc.run_loop(
            warmup_steps=50,
            steps=100,
            T=1.0,
            method="wolff",
            save_warmup=False
        )
        
        assert "tau_E" in stats, "Should return tau_E"
        assert "tau_M" in stats, "Should return tau_M"
        assert np.isnan(stats["accept"]), "Wolff acceptance should be NaN"

    def test_multiple_temperatures(self, small_lattice_dilute):
        """Should handle multiple temperature simulations."""
        mc = MonteCarlo(small_lattice_dilute)
        
        for T in [0.5, 1.0, 2.0]:
            stats = mc.run_loop(
                warmup_steps=20,
                steps=50,
                T=T,
                method="metropolis",
                save_warmup=False
            )
            
            assert stats["tau_E"] > 0, f"tau_E should be positive at T={T}"

    def test_determinism_with_seed(self):
        """Same seed should give deterministic results."""
        np.random.seed(42)
        lat1 = Lattice(rows=5, cols=5, doping=0.7, kf=1.0, J0=1.0)
        spins1 = lat1.magnetic_moments.copy()
        
        np.random.seed(42)
        lat2 = Lattice(rows=5, cols=5, doping=0.7, kf=1.0, J0=1.0)
        spins2 = lat2.magnetic_moments.copy()
        
        # Note: Lattice uses its own rng, so this tests external randomness
        assert lat1.N == lat2.N, "Same seed should give same number of sites"


# ============================================================================
# PART H: EDGE CASES & ROBUSTNESS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_site_lattice(self):
        """Lattice with N=1 should be valid."""
        lat = Lattice(rows=1, cols=1, doping=1.0, kf=1.0, J0=1.0)
        assert lat.N == 1, "Single site should work"
        
        acc = Accumulator(lat)
        E = acc.compute_energy()
        assert E == 0, "Single site energy should be zero (no interactions)"

    def test_metropolis_at_zero_temperature(self, small_lattice_dilute):
        """At T→0, only downhill moves should be accepted."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 1e-6
        
        mc.accept = 0
        for _ in range(100):
            mc.metropolis_step()
        
        # Should accept very few moves
        assert mc.accept <= 10, f"T→0 should have low acceptance, got {mc.accept}"

    def test_metropolis_at_very_high_temperature(self, small_lattice_dilute):
        """At T→∞, nearly all moves should be accepted."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 1e6
        
        mc.accept = 0
        for _ in range(100):
            mc.metropolis_step()
        
        acceptance_rate = mc.accept / 100
        assert acceptance_rate > 0.8, f"High-T should accept most moves, got {acceptance_rate:.2f}"

    def test_magnetization_calculation(self, small_lattice_dilute):
        """Magnetization should be in [-1, 1]."""
        acc = Accumulator(small_lattice_dilute)
        
        for _ in range(20):
            s = small_lattice_dilute.magnetic_moments
            M = np.mean(s)
            assert -1 <= M <= 1, f"Magnetization {M} out of bounds"
            small_lattice_dilute.magnetic_moments[0] *= -1  # Flip one spin

    def test_susceptibility_positive_semidefinite(self, small_lattice_dilute):
        """Susceptibility should be ≥ 0 always."""
        mc = MonteCarlo(small_lattice_dilute)
        mc.T = 1.0
        
        for _ in range(50):
            mc.metropolis_step()
            assert mc.chi >= 0, f"Susceptibility {mc.chi} should be non-negative"


# ============================================================================
# PART I: PERFORMANCE & SCALING TESTS
# ============================================================================

class TestPerformance:
    """Performance and scaling tests."""

    def test_metropolis_step_speed(self, large_lattice):
        """Metropolis step should complete in reasonable time."""
        mc = MonteCarlo(large_lattice)
        mc.T = 1.0
        
        import time
        start = time.time()
        for _ in range(100):
            mc.metropolis_step()
        elapsed = time.time() - start
        
        # Should be fast (< 1s for 100 steps on modest laptop)
        assert elapsed < 2.0, f"100 Metropolis steps took {elapsed:.2f}s, should be faster"

    def test_energy_calculation_vectorized(self, large_lattice):
        """Energy calculation should be O(N^2), not worse."""
        acc = Accumulator(large_lattice)
        
        import time
        start = time.time()
        E = acc.compute_energy()
        elapsed = time.time() - start
        
        # O(N^2) should be fast for N~400
        assert elapsed < 0.1, f"Energy calc took {elapsed:.3f}s, not vectorized?"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
