import numpy as np
from tqdm import tqdm
from accumulator import Accumulator


class MonteCarlo:

    def __init__(self, lattice):
        self.lattice = lattice
        self.acc = Accumulator(lattice)         
        self.rng = np.random.default_rng(seed=42)


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


    def run_loop(self, steps, T, method="metropolis"):
        self.T = T
        if method == "wolff":
            self.precompute_bond_probabilities()

        # # --- Pre-Warmup Phase (Burn-in) ---
        # burn_in_steps = self.acc.max_lag  # or any fixed number (e.g., 300)
        # print(f"Running {burn_in_steps} burn-in steps...")
        # for _ in range(burn_in_steps):
        #     if method == "metropolis":
        #         self.metropolis_step()
        #     elif method == "wolff":
        #         self.wolff_step()
        #     self.acc.sample_production(self.E, self.M)

        # --- Warmup Phase ---
        print("Starting warmup phase...")
        step = 0
        while True:
            if method == "metropolis":
                self.metropolis_step()
            elif method == "wolff":
                self.wolff_step()
            self.acc.sample_warmup(step, self.E, self.M)

            tau = max(self.acc.energy_tau_int, self.acc.magnetization_tau_int)

            if (step + 1) % 100 == 0:
                print(f"Warmup step {step + 1}")
                print(f"Energy τ_int = {self.acc.energy_tau_int:.2f}, Magnetization τ_int = {self.acc.magnetization_tau_int:.2f}")


     
            if step > tau + 100:
                print(f"Warmup finished at step {step} (τ_int = {tau:.2f})\n")
                
                #  to retain only relative data 
                self.acc.energy.clear()
                self.acc.magnetization.clear()
                break

            step += 1

        # --- Production Phase ---
        print("Starting production phase...")
        self.accept = 0

        for _ in tqdm(range(steps), disable=False):
            if method == "metropolis":
                self.metropolis_step()
            else:
                self.wolff_step()
            self.acc.sample_production(self.E, self.M)

        
        # self.acc.pair_correlation = self.acc.pair_correlation_accum / steps
        # Compute average pair correlation and bin
        # corr_avg = self.acc.pair_correlation_accum / steps
        # corr_sum, _ = np.histogram(self.lattice.r_ij, bins=self.lattice.bin_edges, weights=corr_avg)
        # self.acc.binned_pair_correlation = corr_sum / (self.lattice.hist + 1e-10)

        self.acc.process_pair_correlation(steps)

        self.acceptance_rate = self.accept / steps


 ### wolff functional 

    def precompute_bond_probabilities(self):
        J = self.lattice.interaction_matrix
        beta = 1.0 / self.T

        deltaE_same = -2 * J
        deltaE_opp = +2 * J

        self.padd_same = 1 - np.exp(np.minimum(0, -beta * deltaE_same))
        self.padd_opp = 1 - np.exp(np.minimum(0, -beta * deltaE_opp))


    def wolff_step(self):
        spins = self.lattice.magnetic_moments
        J = self.lattice.interaction_matrix
        N = self.lattice.N

        seed = self.rng.integers(N)
        cluster = set([seed])
        to_check = [seed]
        visited = np.zeros(N, dtype=bool)
        visited[seed] = True

        Si = spins[seed]

        while to_check:
            i = to_check.pop()
            Si = spins[i]
            neighbors = np.nonzero(J[i])[0]

            for j in neighbors:
                if visited[j]:
                    continue
                Sj = spins[j]
                padd = self.padd_same[i, j] if Si == Sj else self.padd_opp[i, j]
                if self.rng.random() < padd:
                    visited[j] = True
                    cluster.add(j)
                    to_check.append(j)

        # Flip cluster
        for idx in cluster:
            spins[idx] *= -1

        self.E = self.acc.compute_energy()
        self.M = spins.mean()
        self.accept += 1
