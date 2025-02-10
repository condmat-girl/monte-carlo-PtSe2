import numpy as np
from scipy.special import j0, j1, y0, y1

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
        n = len(self.lattice_points)
        interaction_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                r = self.distance(self.lattice_points[i], self.lattice_points[j])
                interaction_matrix[i, j] = self.rkky_interaction_2d(r)
                interaction_matrix[j, i] = interaction_matrix[i, j]

        return interaction_matrix

    def compute_energy(self):
        energy = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                energy += self.interaction_matrix[i, j] * self.magnetic_moments[i] * self.magnetic_moments[j]
        return energy

    def metropolis_step(self):
        i = np.random.randint(0, self.N)
        old_moment = self.magnetic_moments[i]
        new_moment = -old_moment

        delta_energy = 0.0
        for j in range(len(self.lattice_points)):
            if j != i:
                delta_energy += self.interaction_matrix[i, j] * (new_moment - old_moment) * self.magnetic_moments[j]

        if delta_energy < 0 or np.exp(-delta_energy / self.T) > np.random.random():
            self.magnetic_moments[i] = new_moment
            self.E += delta_energy
            self.accept += 1

        self.M = np.mean(self.magnetic_moments, axis=0)

    def compute_pair_correlation(self):
        correlation = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = self.distance(self.lattice_points[i], self.lattice_points[j])
                correlation.append((r, self.magnetic_moments[i] * self.magnetic_moments[j]))

        return np.array(correlation)

    def monte_carlo_loop(self, steps, warmup, T):
        self.T = T
        self.E = self.compute_energy()

        self.accept = 0
        for _ in range(warmup):
            self.metropolis_step()

 
        self.accept = 0
        self.energy = []
        self.magnetization = []
        pair_correlation = []
        for _ in range(steps):
            self.metropolis_step()
            self.energy.append(self.E)
            self.magnetization.append(self.M)
            pair_correlation.append(self.compute_pair_correlation())

    

        self.pair_correlation = np.mean(pair_correlation,axis=0)

        self.acceptance_rate = self.accept / steps
