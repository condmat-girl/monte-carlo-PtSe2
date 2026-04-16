import numpy as np


class IsingLattice:
    def __init__(self, Lx, Ly, J=1.0):
        self.Lx, self.Ly = Lx, Ly
        self.N = Lx * Ly
        self.J = J

        self.magnetic_moments = np.random.choice([-1, 1], size=self.N)
        self.interaction_matrix = self.build_nn_matrix()
        self.distances = np.ones((self.N, self.N))  
        self.i_idx, self.j_idx = np.triu_indices(self.N, k=1)
        self.r_ij = self.distances[self.i_idx, self.j_idx]
        self.bin_edges = np.linspace(0, 2, 10)
        self.hist, _ = np.histogram(self.r_ij, bins=self.bin_edges)


    def reset_spins(self, ordered=False):
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
                    matrix[idx, nidx] = self.J 
        return matrix

    def compute_energy(self):
        s = self.magnetic_moments
        J = self.interaction_matrix
        return s @ J @ s

    ## мапы от индекса до координат и обратно
    def index_to_coords(self, i):
        x = i % self.Lx
        y = i // self.Lx
        return x, y

    def coords_to_index(self, x, y):
        return (y % self.Ly) * self.Lx + (x % self.Lx)

   ## helper for wolff step
    def get_neighbors_coords(self, x, y):
        return [
            ((x + 1) % self.Lx, y),
            ((x - 1) % self.Lx, y),
            (x, (y + 1) % self.Ly),
            (x, (y - 1) % self.Ly)
        ]
