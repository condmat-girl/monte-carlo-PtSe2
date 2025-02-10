import numpy as np
from scipy.special import j0, j1, y0, y1
# import numba

class TriangularLattice:
    def __init__(self,kf=1.0, J0=1.0):

        self.kf = kf
        self.J0 = J0
       
#### сгенерировать фоновую сетку треугольную 
    def generate_lattice(self, rows, cols, doping):
        points = []
        for i in range(rows):
            for j in range(cols):
                r = np.random.rand() 
                if r < doping:                      
                    x = j * 1 + (i % 2) * (1 / 2)
                    y = i * (1 * np.sqrt(3) / 2)
                    points.append((x, y))
        
        self.N = len(points)
        self.lattice_points=np.array(points)
        self.magnetic_moments = self.initialize_magnetic_moments() 
        self.interaction_matrix = self.compute_rkky_matrix()
    
    
    def initialize_magnetic_moments(self):

        # moments = np.random.uniform(-1, 1, (len(self.lattice_points), 3))
        # norms = np.linalg.norm(moments, axis=1, keepdims=True)
        # return moments / norms  
        return 2*(np.random.random(self.N) > 0.5)-1

    # for the same configuration/system this matrix suppose to stay const 
    def rkky_interaction_2d(self, r):
        if r == 0:
            return 0
        x = self.kf * r
        return -self.J0 * (j0(x) * y0(x) + j1(x) * y1(x))
    
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
        for i in range(self.N):
            for j in range(i + 1, self.N):
                energy += self.interaction_matrix[i, j] * self.magnetic_moments[i] * self.magnetic_moments[j]
        return energy
    
    #more optimised, computing only energy diff
    def metropolis_step(self):

        i = np.random.randint(0, self.N)
        old_moment = self.magnetic_moments[i]
        # new_moment = np.random.uniform(-1, 1, 3)
        # new_moment /= np.linalg.norm(new_moment)
        new_moment = -old_moment

        
        # Compute energy difference
        delta_energy = 0.0
        for j in range(len(self.lattice_points)):
            if j != i:
                # delta_energy += self.interaction_matrix[i, j] * np.dot(new_moment-old_moment, self.magnetic_moments[j])
                delta_energy += self.interaction_matrix[i, j] * (new_moment-old_moment) * self.magnetic_moments[j]
        
        # Accept or reject
        if delta_energy < 0 or np.exp(-delta_energy / self.T) > np.random.random():
            self.magnetic_moments[i] = new_moment
            self.E += delta_energy
            self.accept+=1
            
        self.M = np.mean(self.magnetic_moments, axis=0)


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
    
    def monte_carlo_loop(self, steps, warmup, T):
        self.T = T
        self.E = self.compute_energy()

        self.accept=0
        for _ in range(warmup):
            self.metropolis_step()

        self.accept=0
        self.energy=[]
        self.magnetization = []
        for _ in range(steps):
            self.metropolis_step()
            self.energy.append(self.E)
            self.magnetization.append(self.M)
            

        self.acceptance_rate=self.accept/steps

