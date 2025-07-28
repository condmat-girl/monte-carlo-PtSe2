import numpy as np
from tqdm import tqdm
from accumulator import Accumulator


class MonteCarlo:

    def __init__(self, lattice):
        self.lattice = lattice
        self.acc = Accumulator(lattice)         
        self.rng = np.random.default_rng()

        self.T = None
        self.E = self.acc.compute_energy()
        self.M = self.lattice.magnetic_moments.mean()
        self.accept = 0


    def compute_energy(self):
        return self.acc.compute_energy()

    def compute_pair_correlation(self):
        return self.acc.compute_pair_correlation()


    def metropolis_step(self):
        i = self.rng.integers(self.lattice.N)
        spins = self.lattice.magnetic_moments
        dE = -2 * spins[i] * (self.lattice.interaction_matrix[i] @ spins)
        if dE <= 0 or self.rng.random() < np.exp(-dE / self.T):
            spins[i] *= -1
            self.E += dE
            self.accept += 1
        self.M = spins.mean()


    def monte_carlo_loop(self, steps, warmup, T):
        self.T = T

        for _ in range(warmup):
            self.metropolis_step()

        self.accept = 0
        for _ in tqdm(range(steps), disable=False):
            self.metropolis_step()
            self.acc.sample()

        self.acceptance_rate = self.accept / steps
