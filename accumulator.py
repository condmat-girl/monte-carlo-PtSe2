from typing import List, Tuple, Dict, Optional
import numpy as np
from numpy.typing import NDArray


class Accumulator:
    """
    Online statistics accumulator for Monte Carlo simulations.
    
    Tracks energy, magnetization, susceptibility during warmup and production phases.
    Computes running statistics and autocorrelations efficiently.
    
    Attributes:
        lattice: Reference to Lattice instance
        max_lag: Maximum lag for autocorrelation calculation
        energy: List of energy samples
        magnetization: List of magnetization samples
        susceptibility: List of magnetic susceptibility samples
    """
    
    def __init__(self, lattice, max_lag: int = 1_000) -> None:
        """
        Initialize accumulator.
        
        Args:
            lattice: Lattice instance to reference
            max_lag: Maximum lag for autocorrelation (default: 1000)
        """
        self.lattice = lattice
        self.max_lag = max_lag
        
        # Data storage
        self.energy: List[float] = []
        self.magnetization: List[float] = []
        self.susceptibility: List[float] = []
        self.m2_array: List[float] = []
        self.m4_array: List[float] = []  # For Binder parameter calculation
        self.m_abs_array: List[float] = []
        self.binder_parameter: List[float] = []  # U4 = 1 - <M^4>/(3<M^2>^2)

        # Running statistics for warmup phase (Welford's algorithm)
        self.energy_mean: float = 0.0
        self.energy_variance: float = 0.0
        self.energy_count: int = 0

        self.magnetization_mean: float = 0.0
        self.magnetization_variance: float = 0.0
        self.magnetization_count: int = 0

        # Autocorrelation time estimates
        self.max_lag = max_lag
        self.energy_autocorr: NDArray[np.float64] = np.zeros(max_lag)
        self.magnetization_autocorr: NDArray[np.float64] = np.zeros(max_lag)
        self.energy_tau_int: float = 0.0
        self.magnetization_tau_int: float = 0.0

        # Pair correlation (computed if needed)
        self.correlation_matrix: NDArray[np.float64] = np.zeros(
            (self.lattice.N, self.lattice.N)
        )

    def update_running_statistics(
        self, 
        new_value: float, 
        current_mean: float, 
        current_variance: float, 
        count: int
    ) -> Tuple[float, float, int]:
        """
        Update running mean and variance using Welford's algorithm.
        
        Numerically stable online algorithm with O(1) memory.
        
        Args:
            new_value: New data point
            current_mean: Accumulated mean so far
            current_variance: Accumulated variance so far
            count: Number of samples accumulated (before this one)
            
        Returns:
            Tuple of (updated_mean, updated_variance, updated_count)
        """
        count += 1
        delta = new_value - current_mean
        updated_mean = current_mean + delta / count
        
        # Welford's variance update
        updated_variance = (
            (count - 1) / count * current_variance + 
            (new_value - updated_mean) * delta / count
        )

        return updated_mean, updated_variance, count

    def sample_warmup(
        self, 
        step: int, 
        energy: float, 
        magnetization: float
    ) -> None:
        """
        Record sample during warmup phase.
        
        Updates running statistics and computes autocorrelations incrementally.
        
        Args:
            step: Step number (0-indexed)
            energy: Current system energy
            magnetization: Current magnetization (mean of spins)
        """
        # Update running statistics
        self.energy_mean, self.energy_variance, self.energy_count = \
            self.update_running_statistics(
                energy, self.energy_mean, self.energy_variance, step
            )

        self.magnetization_mean, self.magnetization_variance, self.magnetization_count = \
            self.update_running_statistics(
                magnetization, self.magnetization_mean, self.magnetization_variance, step
            )

        # Append to history for ACF calculation
        self.energy.append(energy)
        self.magnetization.append(magnetization)

        # Compute autocorrelations
        self.energy_autocorr = self.incremental_autocorrelation(
            self.energy, self.energy_mean, self.energy_variance
        )
        self.energy_tau_int = self.calculate_autocorrelation_time(self.energy_autocorr)

        self.magnetization_autocorr = self.incremental_autocorrelation(
            self.magnetization, self.magnetization_mean, self.magnetization_variance
        )
        self.magnetization_tau_int = self.calculate_autocorrelation_time(
            self.magnetization_autocorr
        )

    def sample_production(
        self, 
        E: float, 
        M: float, 
        chi: float,
        M2: float = None,
        M4: float = None
    ) -> None:
        """
        Record sample during production phase.
        
        Args:
            E: Current energy
            M: Current magnetization (mean of spins)
            chi: Current magnetic susceptibility
            M2: M² (used for Binder parameter, optional)
            M4: M⁴ (used for Binder parameter, optional)
        """
        self.energy.append(E)
        self.magnetization.append(M)
        self.susceptibility.append(chi)
        
        # Store moments for Binder parameter calculation
        if M2 is not None:
            self.m2_array.append(M2)
        else:
            self.m2_array.append(M * M)
        
        if M4 is not None:
            self.m4_array.append(M4)
        else:
            self.m4_array.append((M * M) ** 2)

    def incremental_autocorrelation(
        self, 
        series: List[float], 
        mean: float, 
        variance: float
    ) -> NDArray[np.float64]:
        """
        Compute autocorrelation function.
        
        Args:
            series: Time series data
            mean: Mean of series
            variance: Variance of series
            
        Returns:
            Normalized ACF array, shape (min(max_lag, len(series)))
        """
        x = np.asarray(series, dtype=np.float64)
        
        if x.size < 2 or variance <= 0:
            return np.array([1.0])
        
        return self.autocorr_fft(x, unbiased=True)

    def calculate_autocorrelation_time(
        self, 
        acf: NDArray[np.float64]
    ) -> float:
        """
        Estimate autocorrelation time from ACF.
        
        τ_int = 0.5 + Σ_{k=1}^{M} ρ(k)
        
        Args:
            acf: Autocorrelation function
            
        Returns:
            Integrated autocorrelation time (τ_int ≥ 0.5)
        """
        return self.tau_int_from_acf(np.asarray(acf, dtype=np.float64))

    def compute_energy(self) -> float:
        """
        Compute total energy: H = 0.5 * s^T J s (true Hamiltonian).
        
        The 0.5 factor accounts for the fact that J/2 couples each pair.
        This ensures the energy change formula ΔE = -2*s_i*(J_i·s) is consistent.
        
        Returns:
            Total energy of the system
        """
        s = self.lattice.magnetic_moments.astype(np.float64)
        J = self.lattice.interaction_matrix
        
        # E = 0.5 * s @ J @ s avoids double-counting pairs
        # This is the true Hamiltonian H, not 2*H
        energy = float(0.5 * (s @ J @ s))
        return energy

    def calculate_error(
        self, 
        data: NDArray[np.float64], 
        tau_int: float
    ) -> float:
        """
        Standard error accounting for autocorrelation.
        
        SE = sqrt(2 * τ_int * σ² / N)
        
        Args:
            data: Sample array
            tau_int: Integrated autocorrelation time
            
        Returns:
            Standard error accounting for correlations
        """
        N = len(data)
        if N <= 1:
            return float('nan')
        return float(np.sqrt(2 * tau_int * np.var(data) / N))

    def compute_susceptibility(
        self, 
        magnetization: List[float], 
        T: float, 
        N: int
    ) -> float:
        """
        Compute magnetic susceptibility.
        
        χ = (N / T) * (⟨M²⟩ - ⟨M⟩²)
        
        Args:
            magnetization: List of magnetization samples
            T: Temperature
            N: Number of spins
            
        Returns:
            Magnetic susceptibility
        """
        M = np.asarray(magnetization, dtype=np.float64)
        mean_M = np.mean(M)
        mean_M2 = np.mean(M**2)
        
        chi = (N / T) * (mean_M2 - mean_M**2)
        return float(chi)

    def autocorr_fft(
        self, 
        x: NDArray[np.float64], 
        unbiased: bool = True
    ) -> NDArray[np.float64]:
        """
        Compute autocorrelation via FFT (fast for large N).
        
        Equivalent to np.correlate but O(N log N) instead of O(N²).
        
        Args:
            x: Time series data
            unbiased: If True, normalize by N-lag instead of N
            
        Returns:
            Normalized autocorrelation function
        """
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        x = x - x.mean()
        
        # Pad to next power of 2 for FFT efficiency
        nfft = 1 << (2*n - 1).bit_length()
        
        # FFT-based correlation
        f = np.fft.rfft(x, n=nfft)
        acov = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
        
        # Normalize
        var = acov[0] / n
        if unbiased:
            denom = var * (n - np.arange(n))
        else:
            denom = var * n
        
        # Avoid division by zero
        denom[denom == 0] = 1.0
        
        return acov / denom

    def tau_int_from_acf(
        self, 
        acf: NDArray[np.float64]
    ) -> float:
        """
        Integrated autocorrelation time from ACF.
        
        Stops at first sign change to avoid noise.
        τ_int = 0.5 + Σ_{k=1}^{M} ρ(k)
        
        Args:
            acf: Autocorrelation function (normalized, ρ(0) = 1)
            
        Returns:
            Integrated autocorrelation time (≥ 0.5)
        """
        # Find where ACF becomes negative (stops counting)
        for i, v in enumerate(acf[1:], 1):
            if v <= 0:
                tau = 1.0 + 2.0 * np.sum(acf[1:i])
                return float(max(tau, 0.5))  # Ensure τ_int ≥ 0.5
        
        # Didn't find sign change, use full ACF
        tau = 1.0 + 2.0 * np.sum(acf[1:])
        return float(max(tau, 0.5))

    def compute_binder_parameter(self) -> float:
        """
        Compute Binder parameter (reduced fourth order cumulant).
        
        U4 = 1 - ⟨M⁴⟩ / (3⟨M²⟩²)
        
        Physical significance:
        - U4 = 2/3 for paramagnetic (disordered) phase at high T
        - U4 → 1 for ferromagnetic (ordered) phase at low T
        - Curves for different system sizes intersect at T_c
        - Scale-invariant universal value at criticality
        
        References:
        - Binder, K. (1981). "Critical properties from Monte Carlo coarse graining
          and renormalization." Phys. Rev. B 23:3344.
        
        Returns:
            Binder parameter U4 (scalar)
            
        Raises:
            ValueError: If insufficient data collected or ⟨M²⟩ = 0
        """
        if not self.m2_array or not self.m4_array:
            raise ValueError("No production data collected. Run simulation first.")
        
        if len(self.m2_array) < 2:
            raise ValueError("Need at least 2 samples for Binder parameter")
        
        m2_arr = np.asarray(self.m2_array, dtype=np.float64)
        m4_arr = np.asarray(self.m4_array, dtype=np.float64)
        
        mean_m2 = np.mean(m2_arr)
        mean_m4 = np.mean(m4_arr)
        
        if mean_m2 <= 0:
            raise ValueError(f"⟨M²⟩ = {mean_m2} ≤ 0, cannot compute Binder parameter")
        
        # U4 = 1 - ⟨M⁴⟩ / (3⟨M²⟩²)
        u4 = 1.0 - mean_m4 / (3.0 * mean_m2 ** 2)
        
        return float(u4)

    def compute_binder_error(self) -> float:
        """
        Estimate error in Binder parameter using jackknife resampling.
        
        Returns:
            Standard error in U4 estimate
        """
        if not self.m2_array or not self.m4_array or len(self.m2_array) < 2:
            return float('nan')
        
        m2_arr = np.asarray(self.m2_array, dtype=np.float64)
        m4_arr = np.asarray(self.m4_array, dtype=np.float64)
        n = len(m2_arr)
        
        # Jackknife: compute U4 with each sample removed
        u4_values = []
        for i in range(n):
            m2_jack = np.mean(np.delete(m2_arr, i))
            m4_jack = np.mean(np.delete(m4_arr, i))
            
            if m2_jack > 0:
                u4_jack = 1.0 - m4_jack / (3.0 * m2_jack ** 2)
                u4_values.append(u4_jack)
        
        if len(u4_values) < 2:
            return float('nan')
        
        u4_values = np.asarray(u4_values, dtype=np.float64)
        # Jackknife standard error
        se = np.sqrt((n - 1) * np.var(u4_values, ddof=0))
        
        return float(se)

    def compute_magnetic_moment_moments(
        self
    ) -> Tuple[float, float, float, float]:
        """
        Compute moments of the magnetization distribution.
        
        Returns ⟨M⟩, ⟨|M|⟩, ⟨M²⟩, ⟨M⁴⟩
        
        Returns:
            Tuple of (⟨M⟩, ⟨|M|⟩, ⟨M²⟩, ⟨M⁴⟩)
        """
        if not self.magnetization:
            return float('nan'), float('nan'), float('nan'), float('nan')
        
        m_arr = np.asarray(self.magnetization, dtype=np.float64)
        m2_arr = np.asarray(self.m2_array, dtype=np.float64)
        m4_arr = np.asarray(self.m4_array, dtype=np.float64)
        
        mean_m = np.mean(m_arr)
        mean_abs_m = np.mean(np.abs(m_arr))
        mean_m2 = np.mean(m2_arr)
        mean_m4 = np.mean(m4_arr)
        
        return float(mean_m), float(mean_abs_m), float(mean_m2), float(mean_m4)

    def compute_critical_exponents_info(
        self
    ) -> Dict[str, float]:
        """
        Compute derived quantities useful for critical point analysis.
        
        Computes susceptibility (via χ = N*⟨|M|²⟩/T without explicit tracking),
        and Binder parameter for use in finite-size scaling.
        
        Returns:
            Dictionary with keys:
            - 'binder': Binder parameter U4
            - 'binder_error': Error in Binder parameter
            - 'mean_m': ⟨M⟩
            - 'mean_abs_m': ⟨|M|⟩
            - 'mean_m2': ⟨M²⟩
            - 'mean_m4': ⟨M⁴⟩
            - 'dimensionless_ratio': ⟨M⁴⟩/⟨M²⟩² (raw ratio before U4 calc)
        """
        try:
            u4 = self.compute_binder_parameter()
            u4_err = self.compute_binder_error()
            mean_m, mean_abs_m, mean_m2, mean_m4 = self.compute_magnetic_moment_moments()
            
            ratio = mean_m4 / (mean_m2 ** 2) if mean_m2 > 0 else float('nan')
            
            return {
                'binder': u4,
                'binder_error': u4_err,
                'mean_m': mean_m,
                'mean_abs_m': mean_abs_m,
                'mean_m2': mean_m2,
                'mean_m4': mean_m4,
                'dimensionless_ratio': ratio,
            }
        except Exception as e:
            return {
                'binder': float('nan'),
                'binder_error': float('nan'),
                'mean_m': float('nan'),
                'mean_abs_m': float('nan'),
                'mean_m2': float('nan'),
                'mean_m4': float('nan'),
                'dimensionless_ratio': float('nan'),
                'error': str(e),
            }

