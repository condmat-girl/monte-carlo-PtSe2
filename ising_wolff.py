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
        self.spins = self.initialize_spins()
        self.Jij = self.compute_rkky_matrix()
        self.nearest_neighbors = self.compute_nearest_neighbors()

    def initialize_spins(self):
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

        Jij = -self.J0 * (
            j0(self.kf * distances) * y0(self.kf * distances)
            + j1(self.kf * distances) * y1(self.kf * distances)
        )

        np.fill_diagonal(Jij, 0)
        self.distances = distances

        self.i_idx, self.j_idx = np.triu_indices(self.N, k=1)  #  only unique pairs
        self.r_ij = self.distances[self.i_idx, self.j_idx]

        self.hist, self.bin_edges = np.histogram(
            self.r_ij, bins="fd", range=(0, np.max(self.r_ij))
        )

        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        return Jij

    def compute_nearest_neighbors(self):
        nearest_neighbors = []
        for i in range(self.N):
            distances = self.distances[i]
            nearest = np.argsort(distances)
            nearest = nearest[nearest != i]  # Exclude self
            
            # the closest distance
            closest_distance = distances[nearest[0]]
            
            # all neighbors with the closest distance
            closest_neighbors = nearest[distances[nearest] == closest_distance]
            
            nearest_neighbors.append(closest_neighbors)
        return nearest_neighbors

    def compute_energy(self):
        return np.einsum('ij,i,j',self.Jij,self.spins,self.spins)/2

    def wolff_step(self):
        cluster = set()
        visited = set()

        i = np.random.randint(0, self.N)
        cluster.add(i)
        visited.add(i)


        while True:
            current = i
            
            nn = self.nearest_neighbors[current]
            stepen = np.min([0,-2 * self.Jij[i,nn][0] / self.T])
            P_add = 1 - np.exp(stepen)
            added_to_cluster = False

            for j in nn:
                if j not in visited and self.spins[j] == self.spins[i]:
                    if np.random.random() < P_add:
                        cluster.add(j)
                        visited.add(j)
                        i = j 
                        added_to_cluster = True
                        break
            
            if not added_to_cluster:
                break  

        for atom in cluster:
            self.spins[atom] *= -1

        self.E = self.compute_energy()
        self.M = np.mean(self.spins)

    def compute_pair_correlation(self):
        correlation = (
            self.spins[self.i_idx] * self.spins[self.j_idx]
        )
        return correlation

    def monte_carlo_loop(self, steps, warmup, T):
        self.T = T
        self.E = self.compute_energy()

        for _ in range(warmup):
            self.wolff_step()

        self.energy = []
        self.magnetization = []
        self.susceptibility = []
        self.pair_correlation = np.zeros((self.N * (self.N - 1) // 2, 2))

        correlation_accumulated = self.compute_pair_correlation()

        for _ in tqdm(range(steps)):
            self.wolff_step()
            self.energy.append(self.E)
            self.magnetization.append(self.M)
            # self.susceptibility.append(self.compute_susceptibility())
            correlation_accumulated += self.compute_pair_correlation()

        self.correlation = correlation_accumulated / (steps + 1)
        correlation_sum, _ = np.histogram(
            self.r_ij, bins=self.bin_edges, weights=self.correlation
        )
        self.pair_correlation = correlation_sum / (self.hist + 1e-10)

    def compute_susceptibility(self):
        # return np.mean(self.spins**2) - self.M**2
        return np.var(self.spins)

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
            c=self.spins,
        )

        plt.xlabel("x")
        plt.ylabel("y")
        # plt.show()

    def plot_nearest_neighbors(self):

        plt.figure(figsize=(10, 10))
        
        plt.scatter(
            self.lattice_points[:, 0],
            self.lattice_points[:, 1],
            c=self.spins,
            # cmap="coolwarm",
            s=50#,
            # edgecolors="k"
        )

        for i in range(self.N):
            x0, y0 = self.lattice_points[i]
            
            # Рисуем стрелки к соседям
            for j in self.nearest_neighbors[i]:
                dx = self.lattice_points[j][0] - x0
                dy = self.lattice_points[j][1] - y0
                
                # Корректируем для PBC
                dx -= self.Lx * np.round(dx / self.Lx)
                dy -= self.Ly * np.round(dy / self.Ly)
                
                plt.arrow(
                    x0, y0,
                    dx*0.9, dy*0.9,  
                    head_width=0.45,
                    head_length=0.5,
                    fc="black",
                    ec="black",
                    alpha=0.5
                )
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    
    def plot_energy(self):
        if self.energy is None:
            raise ValueError("Run monte_carlo_loop first!")

        plt.figure(figsize=(6, 4))
        plt.plot(self.magnetization)
        plt.xlabel("Monte Carlo Step")
        plt.ylabel("Energy")
        plt.show()

    def plot_magnetization(self):
        if self.magnetization is None:
            raise ValueError("Run monte_carlo_loop first!")

        plt.figure(figsize=(6, 4))
        plt.plot(self.magnetization)
        plt.xlabel("Monte Carlo Step")
        plt.ylabel("Magnetization")
        plt.show()

    def plot_susceptibility(self):
        if self.susceptibility is None:
            raise ValueError("Run monte_carlo_loop first!")

        plt.figure(figsize=(6, 4))
        plt.plot(self.magnetization)
        plt.xlabel("Monte Carlo Step")
        plt.ylabel("Susceptibility")
        plt.show()


    def plot_pair_correlation(self):
        if self.pair_correlation is None:
            raise ValueError("Run monte_carlo_loop first!")

        plt.figure(figsize=(6, 4))
        plt.plot(self.pair_correlation[:self.cols-15])
        plt.xlabel("Distance")
        plt.ylabel("Pair Correlation")
        plt.show()