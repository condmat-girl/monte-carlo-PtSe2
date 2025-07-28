import numpy as np

class Accumulator:


    def __init__(self, lattice):
        self.lattice = lattice
        self.energy = []       
        self.magnetization = [] 

    def compute_energy(self):
        s = self.lattice.magnetic_moments
        J = self.lattice.interaction_matrix
        return  s @ J @ s   

    def compute_pair_correlation(self):
        s = self.lattice.magnetic_moments
        return s[self.lattice.i_idx] * s[self.lattice.j_idx]

    def sample(self):
        self.energy.append(self.compute_energy())
        self.magnetization.append(self.lattice.magnetic_moments.mean())

    def autocorrelation(self, data):
        data = np.asarray(data, float)
        n = len(data)
        data -= data.mean()
        sigma2 = data.var()
        if sigma2 == 0:
            return np.ones(n)
        cor = np.correlate(data, data, mode="full")[n - 1:]
        cor /= sigma2 * np.arange(n, 0, -1)
        return cor
    

    def autocorrelation(self, data):

        data = np.asarray(data)
        n = len(data)

        mean_data = data.mean()
        data_cent = data - mean_data
        sigma2 = (data_cent * data_cent).sum() / n

        if sigma2 == 0:
            return np.ones(n)  

        cor = np.zeros(n)
        for k in range(n):
            num = np.dot(data_cent[0 : n - k], data_cent[k : n])
            den = (n - k) * sigma2
            cor[k] = num / den

        return cor
 

    def calculate_autocorrelation_time(self, correlation):
        for i, val in enumerate(correlation[1:], 1):
            if val < 0:
                return 1 + 2 * np.sum(correlation[1:i])
        return 1 + 2 * np.sum(correlation[1:])

    def calculate_error(self, data, tau_int):
        N = len(data)
        return np.sqrt(2 * tau_int * np.var(data) / N)
