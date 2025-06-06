import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.special import j0, j1, y0, y1
from tqdm import tqdm


class TriangularLattice:
    def __init__(self, kf=1.0, J0=1.0):
        self.kf = kf
        self.J0 = J0

    def generate_lattice(self, rows, cols, doping):
        self.rows = rows
        self.cols = cols
        self.Lx = cols
        self.Ly = rows * (np.sqrt(3) / 2)

        points = []
        for i in range(rows):
            for j in range(cols):
                if np.random.rand() < doping:
                    x = j * 1 + (i % 2) * (1 / 2)
                    y = i * (1 * np.sqrt(3) / 2)
                    points.append((x, y))

        self.N = len(points)
        self.lattice_points = np.array(points)
        self.magnetic_moments = self.initialize_magnetic_moments()
        self.interaction_matrix = self.compute_rkky_matrix()

    def initialize_magnetic_moments(self):
        return 2 * (np.random.random(self.N) > 0.5) - 1

    def rkky_interaction_2d(self, r):
        if r == 0:
            return 0
        x = self.kf * r
        return -self.J0 * (j0(x) * y0(x) + j1(x) * y1(x))

    def distance(self, r1, r2):
        dx = r1[0] - r2[0]
        dy = r1[1] - r2[1]

        # PBC
        dx -= self.Lx * np.round(dx / self.Lx)
        dy -= self.Ly * np.round(dy / self.Ly)

        return np.sqrt(dx**2 + dy**2)

    def compute_rkky_matrix(self):

        distances = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                distances[i, j] = self.distance(
                    self.lattice_points[i], self.lattice_points[j]
                )

        interaction_matrix = -self.J0 * (
            j0(self.kf * distances) * y0(self.kf * distances)
            + j1(self.kf * distances) * y1(self.kf * distances)
        )

        np.fill_diagonal(interaction_matrix, 0)
        self.distances = distances

        self.i_idx, self.j_idx = np.triu_indices(self.N, k=1)  #  only unique pairs
        self.r_ij = self.distances[self.i_idx, self.j_idx]

        self.hist, self.bin_edges = np.histogram(
            self.r_ij, bins="fd", range=(0, np.max(self.r_ij))
        )

        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        return interaction_matrix

    def compute_energy(self):
        energy = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                energy += (
                    self.interaction_matrix[i, j]
                    * self.magnetic_moments[i]
                    * self.magnetic_moments[j]
                )
        return energy

    def metropolis_step(self):
        i = np.random.randint(0, self.N)
        old_moment = self.magnetic_moments[i]
        new_moment = -old_moment

        delta_energy = (new_moment - old_moment) * np.sum(
            self.interaction_matrix[i, :] * self.magnetic_moments
        )
        if delta_energy < 0 or np.exp(-delta_energy / self.T) > np.random.random():
            self.magnetic_moments[i] = new_moment
            self.E += delta_energy
            self.accept += 1

        self.M = np.mean(self.magnetic_moments)

    def compute_pair_correlation(self):
        correlation = (
            self.magnetic_moments[self.i_idx] * self.magnetic_moments[self.j_idx]
        )
        return correlation

    def monte_carlo_loop(self, steps, warmup, T):
        self.T = T
        self.E = self.compute_energy()

        self.accept = 0
        for _ in range(warmup):
            self.metropolis_step()

        self.accept = 0
        self.energy = []
        self.magnetization = []
        # self.pair_correlation = np.zeros((self.N * (self.N - 1) // 2, 2))

        correlation_accumulated = self.compute_pair_correlation()

        for _ in tqdm(range(steps)):
            self.metropolis_step()
            self.energy.append(self.E)
            self.magnetization.append(self.M)
            # correlation_accumulated += self.compute_pair_correlation()

        # self.correlation = correlation_accumulated / (steps + 1)
        # correlation_sum, _ = np.histogram(
        #     self.r_ij, bins=self.bin_edges, weights=self.correlation
        # )
        # self.pair_correlation = correlation_sum / (self.hist + 1e-10)


        # ### calculation <S_i S_j>
        # self.q_array = np.linspace(0,2*np.pi, 151)
        # self.ft_correlation = np.array([np.sum(self.correlation * np.exp(1j * self.r_ij * q)) for q in self.q_array])
        # self.ft_correlation /= len(self.r_ij)

        ## calculation for q = 0, q = 2pikF
        # self.ft_correlation = np.array([np.sum(self.correlation * np.exp(1j * self.r_ij * q)) for q in [0,2*np.pi*self.kf]])
        # self.ft_correlation /= len(self.r_ij)

        self.acceptance_rate = self.accept / steps

    def plot_lattice(self):
        rows, cols = self.rows, self.cols
        full_points = []

        for i in range(rows):
            for j in range(cols):
                x = j * 1 + (i % 2) * (1 / 2)
                y = i * (1 * np.sqrt(3) / 2)
                full_points.append((x, y))

        full_points = np.array(full_points)
        triang = tri.Triangulation(full_points[:, 0], full_points[:, 1])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.triplot(triang, color="lightgray", alpha=0.5, linewidth=0.5)
        sc = ax.scatter(
            self.lattice_points[:, 0],
            self.lattice_points[:, 1],
            s=50,
            c=self.magnetic_moments,
        )

        plt.xlabel("x")
        plt.ylabel("y")
        # plt.show()

    def plot_magnetization(self):
        if self.magnetization is None:
            raise ValueError("Run monte_carlo_loop first!")

        magnetization_autocorr = self.autocorrelation(self.magnetization)

        tau_int = self.calculate_autocorrelation_time(magnetization_autocorr)

        error = self.calculate_error(self.magnetization, tau_int)
        print(error)

        steps = np.arange(len(self.magnetization))
        errors = np.full_like(steps, error) 

        plt.figure(figsize=(5, 4))
        plt.plot(self.magnetization)
        plt.errorbar(steps, self.magnetization, yerr=error, label=f"τ = {tau_int:.2f}\neror = {error:.4f}\n<M> = {np.mean(self.magnetization):.3f}")
        plt.ylabel("Magnetization")
        plt.legend()
        plt.show()

    def plot_pair_correlation(self):
        if self.pair_correlation is None:
            raise ValueError("Run monte_carlo_loop first!")

        r, avg_corr = self.compute_pair_correlation()

        plt.figure(figsize=(6, 4))
        plt.plot(r, avg_corr)
        plt.xlabel("r")
        plt.ylabel("$<S_i S_j> (r)$")
        plt.show()

    def plot_ft_pair_correlation(self):
        if self.ft_correlation is None:
            raise ValueError("Run monte_carlo_loop first!")

        plt.figure(figsize=(6, 4))
        
        plt.plot(self.q_array,np.real(self.ft_correlation), label = 'real')
        plt.plot(self.q_array,np.imag(self.ft_correlation), label = 'imag')
        plt.xlabel("q")
        plt.ylabel("$<S_i S_j> (q)$")
        plt.legend()
        plt.show()

    ## universal for any quantities 
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
    
    def calculate_autocorrelation_time(self,correlation):
        
        for i, val in enumerate(correlation):
            if val < 0:
                return 1 + 2 * np.sum(correlation[1:i])

        return 1 + 2 * np.sum(correlation[1:])
        
    

    def calculate_error(self,data, tau_int):
       
        N = len(data)
        variance = np.var(data)
        error = np.sqrt(2 * tau_int * variance / N)

        return error
