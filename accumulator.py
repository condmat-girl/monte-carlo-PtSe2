import numpy as np

class Accumulator:
    def __init__(self, lattice, max_lag=1_000):
        self.lattice = lattice
        self.energy = []
        self.magnetization = []
        self.susceptibility = []
        self.m2_array = []
        self.m_abs_array = []

        # for warmup phase
        self.energy_mean = 0.0
        self.energy_variance = 0.0
        self.energy_count = 0

        self.magnetization_mean = 0.0
        self.magnetization_variance = 0.0
        self.magnetization_count = 0

        ## for acf
        self.max_lag = max_lag 
        self.energy_autocorr = np.zeros(max_lag)
        self.magnetization_autocorr = np.zeros(max_lag)
        self.energy_tau_int = 0.0
        self.magnetization_tau_int = 0.0

   
        self.correlation_matrix = np.zeros((self.lattice.N, self.lattice.N))

        # костыли
        self.corelation = []




    def update_running_statistics(self, new_value, current_mean, current_variance, count):

        count += 1
        delta = new_value - current_mean
        updated_mean = current_mean + delta / count
        updated_variance = (count - 1) / count * current_variance + (new_value - updated_mean)*delta / count

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

    def sample_production(self, E, M, chi, m2,mabs):
        self.energy.append(E)
        self.magnetization.append(M)
        # print(f"Production: E = {energy:10.4f}, M = {magnetization:10.4f}")
        # self.pair_correlation_accum += self.lattice.compute_pair_correlation()
        spins = self.lattice.magnetic_moments
        self.correlation_matrix += np.outer(spins, spins)
        self.susceptibility.append(chi)
        self.m2_array.append(m2)
        self.m_abs_array.append(mabs)

        # self.susceptibility.append(self.compute_susceptibility(M,spins,T))


    def compute_energy(self):
        s = self.lattice.magnetic_moments
        J = self.lattice.interaction_matrix
        return s @ J @ s

    def autocorr(self, x):
        x = np.asarray(x, float)
        x = x - x.mean()
        n = len(x)
        result = np.correlate(x, x, mode='full')
        acov = result[n-1:]   # lags 0..n-1
        acov /= np.arange(n, 0, -1)   # unbiased: divide by N-k
        return acov / acov[0]         # normalize → acf[0]=1


    # def incremental_autocorrelation(self, data, mean, variance):
    #     n = len(data)
    #     data = np.asarray(data)

    #     max_lag = min(self.max_lag, n)

    #     cor = np.zeros(max_lag)
    #     for k in range(max_lag):
    #         num = np.dot(data[:n - k] - mean, data[k:] - mean)
    #         den = (n - k) * variance
    #         cor[k] = num / den

    #     return cor

    def calculate_autocorrelation_time(self, correlation):
        self.corelation = correlation
        for i, val in enumerate(correlation[1:], 1):
            if val < 0:
                return 1 + 2 * np.sum(correlation[1:i])
        return 1 + 2 * np.sum(correlation[1:])

    def calculate_error(self, data, tau_int):
        N = len(data)
        return np.sqrt(2 * tau_int * np.var(data) / N)
    

    def compute_susceptibility(self, magnetization, T, N):
        M = np.asarray(magnetization)
        mean_M = np.mean(M)
        mean_M2 = np.mean(M**2)
        return (N / T) * (mean_M2 - mean_M**2)


    
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
        r_flat = r_ij[i, j]
        correlation_matrix_flat = correlation_matrix[i, j]

        bin_edges = self.lattice.bin_edges
        hist, _ = np.histogram(r_flat, bins=bin_edges)
        corr_sum, _ = np.histogram(r_flat, bins=bin_edges, weights=correlation_matrix_flat)

        self.binned_pair_correlation = corr_sum / (hist + 1e-10)
        self.bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
