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
        self.step = 0
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

        for _ in range(300):
            self.metropolis_step()
            self.acc.sample_production(self.E, self.M) 
                
        print("Starting warmup phase...")
        step = 0
        while True:
            self.metropolis_step()
            self.acc.sample_warmup(step, self.E, self.M)

            energy_tau = self.acc.energy_tau_int
            magnet_tau = self.acc.magnetization_tau_int
            tau = max(energy_tau, magnet_tau)

            if (step + 1) % 100 == 0:
                print(f"Warmup step {step + 1}")
                print(f"Energy τ_int = {energy_tau:.2f}, Magnetization τ_int = {magnet_tau:.2f}")

            step += 1
            if step > tau:
                print(f"Warmup finished at step {step} (τ_int = {tau:.2f})\n")
                break



        # print("Starting warmup phase...")
        # for step in tqdm(range(warmup), disable=False):
        #     self.metropolis_step()
        #     self.acc.sample_warmup(step, self.E, self.M)
            
        #     if (step + 1) % 100 == 0:
        #         print(f"Warmup step {step + 1}/{warmup}")
        #         print(f"Energy τ_int = {self.acc.energy_tau_int:.2f}")
        #         print(f"Magnetization τ_int = {self.acc.magnetization_tau_int:.2f}\n")
        
        print("Starting production phase...")
        self.accept = 0
        for _ in tqdm(range(steps), disable=False):
            self.metropolis_step()
            self.acc.sample_production(self.E, self.M) 
        
        self.acceptance_rate = self.accept / steps
