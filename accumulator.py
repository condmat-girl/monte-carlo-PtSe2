import numpy as np

class Accumulator:
    def __init__(self, lattice, max_lag=1024):
        self.lattice = lattice
        self.max_lag = int(max_lag)

        self.energy = []
        self.magnetization = []

        # Welford-like incremental mean/variance
        self.energy_mean = 0.0
        self.energy_var  = 0.0
        self.energy_count = 0

        self.mag_mean = 0.0
        self.mag_var  = 0.0
        self.mag_count = 0

        # ACFs 
        self.energy_autocorr = None
        self.magnetization_autocorr = None
        self.energy_tau_int = 0.0
        self.magnetization_tau_int = 0.0

        # pair correlation 
        self.correlation_matrix = np.zeros((self.lattice.N, self.lattice.N))
        self.binned_pair_correlation = None
        self.bin_centers = None

    @staticmethod
    def _update_running_stats(x, mean, var, count):

        count += 1
        delta = x - mean
        mean += delta / count
        var += (delta * (x - mean) - var) / count
        return mean, var, count

    def sample_warmup(self, step, energy, magnetization):

        self.energy_mean, self.energy_var, self.energy_count = \
            self._update_running_stats(energy, self.energy_mean, self.energy_var, self.energy_count)

        self.mag_mean, self.mag_var, self.mag_count = \
            self._update_running_stats(magnetization, self.mag_mean, self.mag_var, self.mag_count)

        self.energy.append(energy)
        self.magnetization.append(magnetization)

        # dynamic-length autocorrelation
        self.energy_autocorr = self._autocorr_dynamic(
            self.energy, self.energy_mean, self.energy_var
        )
        self.energy_tau_int = self._tau_int(self.energy_autocorr)

        self.magnetization_autocorr = self._autocorr_dynamic(
            self.magnetization, self.mag_mean, self.mag_var
        )
        self.magnetization_tau_int = self._tau_int(self.magnetization_autocorr)

    def sample_production(self, energy, magnetization):
        self.energy.append(energy)
        self.magnetization.append(magnetization)

        s = self.lattice.magnetic_moments
        self.correlation_matrix += np.outer(s, s)

    def compute_energy(self):
        s = self.lattice.magnetic_moments
        J = self.lattice.interaction_matrix
        return s @ J @ s

    def _autocorr_dynamic(self, data, mean, variance):
        n = len(data)
        if n < 2 or variance <= 1e-15:
            return np.array([10.0])  # trivial ACF

        lag_max = int(min(self.max_lag, n - 1))
        x = np.asarray(data, dtype=float) - mean

        acf = np.empty(lag_max + 1, dtype=float)
        for k in range(lag_max + 1):
            num = np.dot(x[:n - k], x[k:])
            den = (n - k) * variance
            acf[k] = num / den
        return acf

    @staticmethod
    def _tau_int(acf):
        if acf is None or acf.size == 0:
            return 0.0
        tail = acf[1:]
        s = 0.0
        for v in tail:
            if v <= 0.0:
                break
            s += v
        return 1.0 + 2.0 * s

    @staticmethod
    def calculate_error(data, tau_int):
        n = len(data)
        if n == 0:
            return np.nan
        return np.sqrt(2.0 * max(tau_int, 1.0) * np.var(data, ddof=0) / n)

    def process_pair_correlation(self, n_prod_steps):
        corr = self.correlation_matrix / max(n_prod_steps, 1)
        r_ij = self.lattice.distances

        iu, ju = np.triu_indices(self.lattice.N, k=0)
        r_flat = r_ij[iu, ju]
        w_flat = corr[iu, ju]

        edges = self.lattice.bin_edges
        counts, _ = np.histogram(r_flat, bins=edges)
        sums,   _ = np.histogram(r_flat, bins=edges, weights=w_flat)

        self.binned_pair_correlation = sums / (counts + 1e-12)
        self.bin_centers = 0.5 * (edges[:-1] + edges[1:])

    def get_binned_pair_correlation(self):
        if self.binned_pair_correlation is None:
            raise RuntimeError("Pair correlation not computed yet. Call process_pair_correlation first.")
        return self.bin_centers, self.binned_pair_correlation
