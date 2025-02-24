import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.special import j0, j1, y0, y1
from tqdm import tqdm

class TriangularLattice:
    def __init__(self, kf=1.0, J0=1.0):
        self.kf = kf
        self.J0 = J0
        self.sqrt3_2 = np.sqrt(3) / 2  

    def generate_lattice(self, rows, cols, doping):
        self.rows, self.cols = rows, cols
        self.Lx, self.Ly = cols, rows * self.sqrt3_2
        
        i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        mask = np.random.rand(rows, cols) < doping
        x = j[mask] + (i[mask] % 2) * 0.5
        y = i[mask] * self.sqrt3_2
        
        self.lattice_points = np.column_stack((x, y))
        self.N = len(self.lattice_points)
        
        self.magnetic_moments = np.random.choice([-1, 1], self.N)
        self.compute_rkky_matrix()

    def distance(self, r1, r2):
        d = r1 - r2
        d[:, 0] -= self.Lx * np.round(d[:, 0] / self.Lx)
        d[:, 1] -= self.Ly * np.round(d[:, 1] / self.Ly)
        return np.linalg.norm(d, axis=1)
    
    def compute_rkky_matrix(self):
        r_vectors = self.lattice_points[:, np.newaxis, :] - self.lattice_points[np.newaxis, :, :]
        r_vectors[:, :, 0] -= self.Lx * np.round(r_vectors[:, :, 0] / self.Lx)
        r_vectors[:, :, 1] -= self.Ly * np.round(r_vectors[:, :, 1] / self.Ly)
        distances = np.linalg.norm(r_vectors, axis=-1)
        
        x = self.kf * distances
        interaction_matrix = -self.J0 * (j0(x) * y0(x) + j1(x) * y1(x))
        np.fill_diagonal(interaction_matrix, 0)
        
        self.interaction_matrix = interaction_matrix
        self.distances = distances

    def compute_energy(self):
        return 0.5 * np.sum(
            self.interaction_matrix * np.outer(self.magnetic_moments, self.magnetic_moments)
        )
    
    def metropolis_step(self, T):
        i = np.random.randint(0, self.N)
        delta_energy = -2 * self.magnetic_moments[i] * np.dot(self.interaction_matrix[i], self.magnetic_moments)
        
        accepted = False
        if delta_energy < 0 or np.exp(-delta_energy / T) > np.random.random():
            self.magnetic_moments[i] *= -1
            self.fast_update_pair_correlation(i)
            accepted = True
        return accepted
    
    def monte_carlo_loop(self, steps, warmup, T):
        self.pair_correlation_sum = np.zeros_like(self.distances)
        self.pair_correlation_count = np.zeros_like(self.distances)
        accepted_moves = 0
        for _ in range(warmup):
            self.metropolis_step(T)
        
        self.energy, self.magnetization = [], []
        for _ in tqdm(range(steps)):
            if self.metropolis_step(T):
                accepted_moves += 1
            self.energy.append(self.compute_energy())
            self.magnetization.append(np.mean(self.magnetic_moments))
        self.acceptance_rate = accepted_moves / steps  
        self.pair_correlation = self.pair_correlation_sum / (self.pair_correlation_count + 1e-10)
        return True   
    
    def fast_update_pair_correlation(self, i):
        # Update pair correlation sum and count for affected spin
        for j in range(self.N):
            if i != j:
                corr = self.magnetic_moments[i] * self.magnetic_moments[j]
                self.pair_correlation_sum[i, j] += corr
                self.pair_correlation_sum[j, i] += corr  # symmetry
                self.pair_correlation_count[i, j] += 1
                self.pair_correlation_count[j, i] += 1

    def compute_pair_correlation(self):
        distances = self.distances[np.triu_indices(self.N, k=1)]
        correlations = self.pair_correlation[np.triu_indices(self.N, k=1)]
        
        bins = np.linspace(0, np.max(distances), 50)
        bin_indices = np.digitize(distances, bins)
        bin_sums = np.zeros(len(bins) - 1)
        bin_counts = np.zeros(len(bins) - 1)
        
        for idx, corr in zip(bin_indices, correlations):
            if 1 <= idx <= len(bin_sums):  # Avoid out-of-range errors
                bin_sums[idx - 1] += corr
                bin_counts[idx - 1] += 1
        
        avg_correlation = bin_sums / (bin_counts + 1e-10)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        return bin_centers, avg_correlation

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

        plt.figure(figsize=(6, 4))
        plt.plot(self.magnetization)
        plt.xlabel("Monte Carlo Step")
        plt.ylabel("Magnetization")
        plt.show()

    def plot_pair_correlation(self):
        if self.pair_correlation is None:
            raise ValueError("Run monte_carlo_loop first!")

        r, avg_corr = self.compute_pair_correlation()

        plt.figure(figsize=(6, 4))
        plt.plot(r, avg_corr)
        plt.xlabel("Distance")
        plt.ylabel("Pair Correlation")
        plt.show()
