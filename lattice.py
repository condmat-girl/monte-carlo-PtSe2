import numpy as np
from scipy.special import j0, j1, y0, y1


class Lattice:

    def __init__(self, rows: int, cols: int, doping: float,
                 kf: float = 1.0, J0: float = 1.0):
        
        self.rng = np.random.default_rng(seed=12)
        self.kf, self.J0 = kf, J0

        self.rows, self.cols = rows, cols
        self.Lx = cols
        self.Ly = rows * (np.sqrt(3) / 2)

        self.generate_lattice(rows, cols, doping)

 
    def generate_lattice(self, rows, cols, doping):
        self.rows = rows
        self.cols = cols
        self.Lx = cols
        self.Ly = rows * (np.sqrt(3) / 2)

        points = []
        for i in range(rows):
            for j in range(cols):
                if self.rng.random() < doping:
                    x = j + 0.5 * (i % 2)
                    y = i * (np.sqrt(3) / 2)
                    points.append((x, y))

        if not points:
            raise ValueError("No occupied sites – increase doping")

        self.N = len(points)
        self.lattice_points = np.array(points)
        self.coords = self.lattice_points

        self.magnetic_moments = self.initialize_magnetic_moments()
        self.interaction_matrix = self.compute_rkky_matrix()

    def initialize_magnetic_moments(self):
        return 2 * (self.rng.random(self.N) > 0.5) - 1

    def rkky_interaction_2d(self, r):
        if r == 0:
            return 0
        x = self.kf * r
        return -self.J0 * (j0(x) * y0(x) + j1(x) * y1(x))

    def distance(self, r1, r2):
        dx = r1[0] - r2[0]
        dy = r1[1] - r2[1]
        dx -= self.Lx * np.round(dx / self.Lx)
        dy -= self.Ly * np.round(dy / self.Ly)
        return np.hypot(dx, dy)

    def compute_rkky_matrix(self):
        distances = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                distances[i, j] = self.distance(self.lattice_points[i],
                                                 self.lattice_points[j])

        interaction_matrix = -self.J0 * (
            j0(self.kf * distances) * y0(self.kf * distances) +
            j1(self.kf * distances) * y1(self.kf * distances)
        )
        np.fill_diagonal(interaction_matrix, 0)

        self.distances = distances
        self.i_idx, self.j_idx = np.triu_indices(self.N, k=1)
        self.r_ij = distances[self.i_idx, self.j_idx]
        ## experiment
        self.hist, self.bin_edges = np.histogram(self.r_ij, bins="fd")
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        return interaction_matrix
    
    def compute_pair_correlation(self):
        
        s = self.magnetic_moments
        return s[self.i_idx] * s[self.j_idx]

    