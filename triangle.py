import numpy as np
from scipy.special import j0, j1, y0, y1
# import numba

class TriangularLattice:
    def __init__(self, rows, cols, spacing=1.0, k_f=1.0, U0=1.0):
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
        self.k_f = k_f
        self.U0 = U0
        self.lattice_points = self.generate_lattice()
        self.magnetic_moments = self.initialize_magnetic_moments()
        self.interaction_matrix = self.compute_rkky_matrix()

    def generate_lattice(self):
        points = []
        for i in range(self.rows):
            for j in range(self.cols):
                x = j * self.spacing + (i % 2) * (self.spacing / 2)
                y = i * (self.spacing * np.sqrt(3) / 2)
                points.append((x, y))
        return np.array(points)
    
    def initialize_magnetic_moments(self):
        moments = np.random.uniform(-1, 1, (len(self.lattice_points), 3))
        norms = np.linalg.norm(moments, axis=1, keepdims=True)
        return moments / norms  

    # for the same configuration/system this matrix suppose to stay const 
    
    def rkky_interaction_2d(self, r):
        if r == 0:
            return 0
        x = self.k_f * r
        return -self.U0 * (j0(x) * y0(x) + j1(x) * y1(x))
    
    def compute_rkky_matrix(self):
        n = len(self.lattice_points)
        interaction_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(self.lattice_points[i] - self.lattice_points[j])
                interaction_matrix[i, j] = self.rkky_interaction_2d(r)
                interaction_matrix[j, i] = interaction_matrix[i, j]
        return interaction_matrix
    
    def compute_energy(self):
        energy = 0.0
        for i in range(len(self.lattice_points)):
            for j in range(i + 1, len(self.lattice_points)):
                energy += self.interaction_matrix[i, j] * np.dot(self.magnetic_moments[i], self.magnetic_moments[j])
        return energy
    
    #more optimised, computing only energy diff

    def metropolis_step(self, temperature):
        index = np.random.randint(0, len(self.lattice_points))
        old_moment = self.magnetic_moments[index].copy()
        new_moment = np.random.uniform(-1, 1, 3)
        new_moment /= np.linalg.norm(new_moment)
        
        # Compute energy difference
        delta_energy = 0.0
        for j in range(len(self.lattice_points)):
            if j != index:
                delta_energy += self.interaction_matrix[index, j] * (np.dot(new_moment, self.magnetic_moments[j]) - np.dot(old_moment, self.magnetic_moments[j]))
        
        # Accept or reject
        if delta_energy < 0 or np.exp(-delta_energy / temperature) > np.random.random():
            self.magnetic_moments[index] = new_moment
            return True
        else:
            return False

    # def metropolis_step(self, temperature):
    #     index = random.randint(0, len(self.lattice_points) - 1)
    #     old_moment = self.magnetic_moments[index].copy()
    #     new_moment = np.random.uniform(-1, 1, 3)
    #     new_moment /= np.linalg.norm(new_moment)
    #     self.magnetic_moments[index] = new_moment
        
    #     old_energy = self.compute_energy()
    #     new_energy = self.compute_energy()
        
    #     if new_energy < old_energy or np.exp((old_energy - new_energy) / temperature) > random.random():
    #         return True
    #     else:
    #         self.magnetic_moments[index] = old_moment
    #         return False
    
    def run_monte_carlo(self, steps, temperature):
        for _ in range(steps):
            self.metropolis_step(temperature)

    def average_magnetization_x(self):
        return np.mean(self.magnetic_moments[:, 0])

