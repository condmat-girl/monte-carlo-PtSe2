import numpy as np


class IsingLattice:
    def __init__(self, Lx, Ly, J=1.0):
        self.Lx, self.Ly = Lx, Ly
        self.N = Lx * Ly
        self.J = J

        self.magnetic_moments = np.random.choice([-1, 1], size=self.N)
        self.interaction_matrix = self.build_nn_matrix()
        idx = np.arange(self.N)
        self.coords = np.column_stack([idx // self.Ly, idx % self.Ly]).astype(float)
        self.distances = self.build_distance_matrix()
        self.i_idx, self.j_idx = np.triu_indices(self.N, k=1)
        self.r_ij = self.distances[self.i_idx, self.j_idx]
        r_max = self.r_ij.max() if self.r_ij.size else 1.0
        self.bin_edges = np.linspace(0, r_max + 1e-9, 10)
        self.hist, _ = np.histogram(self.r_ij, bins=self.bin_edges)


    def reset_spins(self, ordered=False):
        if ordered:
            self.magnetic_moments = np.ones(self.N, dtype=int)
        else:
            self.magnetic_moments = np.random.choice([-1, 1], size=self.N)


    def build_nn_matrix(self):
        matrix = np.zeros((self.N, self.N))
        for i in range(self.Lx):
            for j in range(self.Ly):
                idx = i * self.Ly + j
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni = (i + dx) % self.Lx
                    nj = (j + dy) % self.Ly
                    nidx = ni * self.Ly + nj
                    matrix[idx, nidx] += self.J 
        return matrix

    def compute_energy(self):
        s = self.magnetic_moments
        J = self.interaction_matrix
        return 0.5 * s @ J @ s

    def build_distance_matrix(self):
        x = self.coords[:, 0][:, None]
        y = self.coords[:, 1][:, None]
        dx = np.abs(x - x.T)
        dy = np.abs(y - y.T)
        dx = np.minimum(dx, self.Lx - dx)
        dy = np.minimum(dy, self.Ly - dy)
        return np.sqrt(dx ** 2 + dy ** 2)

    ## мапы от индекса до координат и обратно
    def index_to_coords(self, i):
        x = i // self.Ly
        y = i % self.Ly
        return x, y

    def coords_to_index(self, x, y):
        return (x % self.Lx) * self.Ly + (y % self.Ly)

   ## helper for wolff step
    def get_neighbors_coords(self, x, y):
        return [
            ((x + 1) % self.Lx, y),
            ((x - 1) % self.Lx, y),
            (x, (y + 1) % self.Ly),
            (x, (y - 1) % self.Ly)
        ]
