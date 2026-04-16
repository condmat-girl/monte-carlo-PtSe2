import numpy as np


class Accumulator:
    def __init__(self, lattice, max_lag=1_000):
        self.lattice = lattice

        # time-series storage
        self.energy         = []
        self.magnetization  = []
        self.susceptibility = []
        self.m2_array       = []
        self.m4_array       = []
        self.m_abs_array    = []
        self.e2_array       = []

        # for warmup phase
        self.energy_mean       = 0.0
        self.energy_variance   = 0.0
        self.energy_count      = 0

        self.magnetization_mean     = 0.0
        self.magnetization_variance = 0.0
        self.magnetization_count    = 0

        # for acf
        self.max_lag = max_lag
        self.energy_autocorr       = np.zeros(max_lag)
        self.magnetization_autocorr = np.zeros(max_lag)
        self.energy_tau_int        = 0.0
        self.magnetization_tau_int = 0.0

        self.correlation_matrix = np.zeros((self.lattice.N, self.lattice.N))

        # костыли
        self.corelation = []

    # ── running statistics (warmup) ───────────────────────────────────────────
    def update_running_statistics(self, new_value, current_mean, current_variance, count):
        count += 1
        delta = new_value - current_mean
        updated_mean = current_mean + delta / count
        updated_variance = (count - 1) / count * current_variance + (new_value - updated_mean) * delta / count
        return updated_mean, updated_variance, count

    def sample_warmup(self, step, energy, magnetization):
        self.energy_mean, self.energy_variance, self.energy_count = self.update_running_statistics(
            energy, self.energy_mean, self.energy_variance, step
        )
        self.magnetization_mean, self.magnetization_variance, self.magnetization_count = self.update_running_statistics(
            magnetization, self.magnetization_mean, self.magnetization_variance, step
        )
        self.energy.append(energy)
        self.magnetization.append(magnetization)

        self.energy_autocorr = self.incremental_autocorrelation(
            self.energy, self.energy_mean, self.energy_variance
        )
        self.energy_tau_int = self.calculate_autocorrelation_time(self.energy_autocorr)

        self.magnetization_autocorr = self.incremental_autocorrelation(
            self.magnetization, self.magnetization_mean, self.magnetization_variance
        )
        self.magnetization_tau_int = self.calculate_autocorrelation_time(self.magnetization_autocorr)

    # ── production sampling ───────────────────────────────────────────────────
    def sample_production(self, E, M, chi, m2, m4, mabs, e2):
        self.energy.append(E)
        self.magnetization.append(M)
        spins = self.lattice.magnetic_moments
        self.correlation_matrix += np.outer(spins, spins)
        self.susceptibility.append(chi)
        self.m2_array.append(m2)
        self.m4_array.append(m4)
        self.m_abs_array.append(mabs)
        self.e2_array.append(e2)

    # ── energy ────────────────────────────────────────────────────────────────
    def compute_energy(self):
        s = self.lattice.magnetic_moments
        J = self.lattice.interaction_matrix
        return s @ J @ s

    # ── Binder cumulant  U4 = 1 - <M⁴> / (3 <M²>²) ──────────────────────────
    def compute_binder_and_error(self, m2_array, m4_array, n_blocks=20):
        m2 = np.asarray(m2_array)
        m4 = np.asarray(m4_array)
        n  = len(m2)

        # trim to be divisible by n_blocks
        n_trim  = (n // n_blocks) * n_blocks
        m2      = m2[:n_trim].reshape(n_blocks, -1)
        m4      = m4[:n_trim].reshape(n_blocks, -1)

        # full estimate
        M2_mean = m2.mean()
        M4_mean = m4.mean()
        binder  = 1.0 - M4_mean / (3.0 * M2_mean ** 2)

        # block jackknife error
        jk = np.zeros(n_blocks)
        for k in range(n_blocks):
            m2_jk  = np.delete(m2, k, axis=0).mean()
            m4_jk  = np.delete(m4, k, axis=0).mean()
            jk[k]  = 1.0 - m4_jk / (3.0 * m2_jk ** 2)

        error = np.sqrt((n_blocks - 1) * np.var(jk, ddof=0))
        return binder, error

    # ── specific heat  C = (<E²> - <E>²) / (N T²) ────────────────────────────
    def compute_specific_heat(self, T):
        E = np.asarray(self.energy)
        return (np.mean(E ** 2) - np.mean(E) ** 2) / (self.lattice.N * T ** 2)

    # ── susceptibility  χ = N(<|M|²> - <|M|>²) / T ───────────────────────────
    def compute_susceptibility(self, T):
        mabs = np.asarray(self.m_abs_array)
        return self.lattice.N * (np.mean(mabs ** 2) - np.mean(mabs) ** 2) / T

    # ── error via integrated autocorrelation time ─────────────────────────────
    def mean_and_error(self, data):
        data    = np.asarray(data, dtype=float)
        acf     = self.autocorr_fft(data)
        tau_int = self.tau_int_from_acf(acf)
        return data.mean(), self.calculate_error(data, tau_int)

    def calculate_error(self, data, tau_int):
        N = len(data)
        return np.sqrt(2 * tau_int * np.var(data) / N)

    # ── pair correlation ──────────────────────────────────────────────────────
    def compute_pair_correlation(self):
        if self.pair_correlation is None:
            raise RuntimeError("Pair correlation has not been computed yet. Run the simulation first.")
        return self.pair_correlation

    def get_binned_pair_correlation(self):
        if self.binned_pair_correlation is None:
            raise RuntimeError("Pair correlation has not been computed yet.")
        return self.lattice.bin_centers, self.binned_pair_correlation

    def process_pair_correlation(self, steps):
        correlation_matrix = self.correlation_matrix / steps
        r_ij = self.lattice.distances

        i, j = np.triu_indices(self.lattice.N, k=0)
        r_flat            = r_ij[i, j]
        correlation_matrix_flat = correlation_matrix[i, j]

        bin_edges = self.lattice.bin_edges
        hist,     _ = np.histogram(r_flat, bins=bin_edges)
        corr_sum, _ = np.histogram(r_flat, bins=bin_edges, weights=correlation_matrix_flat)

        self.binned_pair_correlation = corr_sum / (hist + 1e-10)
        self.bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # ── autocorrelation ───────────────────────────────────────────────────────
    def autocorr_fft(self, x, unbiased=True):
        x = np.asarray(x, float)
        n = x.size
        x = x - x.mean()
        nfft  = 1 << (2 * n - 1).bit_length()
        f     = np.fft.rfft(x, n=nfft)
        acov  = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
        var   = acov[0] / n
        denom = var * (n - np.arange(n)) if unbiased else var * n
        return acov / denom

    def tau_int_from_acf(self, acf):
        for i, v in enumerate(acf[1:], 1):
            if v <= 0:
                return 1.0 + 2.0 * np.sum(acf[1:i])
        return 1.0 + 2.0 * np.sum(acf[1:])
