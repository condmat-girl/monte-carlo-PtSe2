import numpy as np
from tqdm import tqdm
from accumulator import Accumulator
from visualization import Visualization
# from visualization_ising import Visualization_Ising


class MonteCarlo:

    def __init__(self, lattice, progress=False):
        self.lattice = lattice
        self.acc = Accumulator(lattice)
        self.vis = Visualization(lattice, self.acc)
        # for tests:
        # self.vis_is = Visualization_Ising(lattice,self.acc)
        self.rng = np.random.default_rng(seed=42)
        self.progress = progress
        self.T = None
        self.E = self.acc.compute_energy()
        self.M = self.lattice.magnetic_moments.mean()
        self.M2 = None
        self.mabs = None
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
        self.M = np.mean(spins)  
        self.M2 = (self.M)**2
        self.chi = self.lattice.N * (1 - self.M**2) / self.T

    # def run_loop(self, warmup_steps, steps, T, method="metropolis",
    #             save_warmup=False , outdir="frames"):
        

    #     self.T = T

    #     if method == "wolff":
    #         self.precompute_bond_probabilities()

    #     if self.progress==True:
    #         print("Starting warmup phase...")
    #     for step in tqdm(range(warmup_steps), disable=not self.progress):
    #         if method == "metropolis":
    #             self.metropolis_step()
    #             if save_warmup and (step % 1_000 == 0):
    #                 self.vis.plot_coords_save(
    #                     int(step//1_00),
    #                     output_dir=outdir
    #                 )
    #         elif method == "wolff":
    #             if save_warmup:
    #                 (cluster_idx, edges_fm, edges_afm) = self.wolff_step(return_cluster=True)
    #                 self.vis.plot_cluster_save(cluster_idx, edges_fm, edges_afm, step, self.lattice.N, output_dir=outdir)
    #             else:
    #                 self.wolff_step(return_cluster=False)




    #     if save_warmup:
    #         self.vis.create_gif_from_frames(
    #             output_dir=outdir,
    #             output_file= str(method) + "_" + str(T) + "_warmup.gif",
    #             fps=self.vis.fps
    #         )

    #     if self.progress==True:
    #         print("Starting production phase...")
    #     self.accept = 0
    #     for _ in tqdm(range(steps), disable=not self.progress):
    #         if method == "metropolis":
    #             self.metropolis_step()
    #         else:
    #             self.wolff_step()
    #         self.acc.sample_production(self.E, self.M, self.chi,self.M2,self.mabs)

    #     self.acceptance_rate = self.accept / steps

    def run_loop(self, warmup_steps, steps, T, method="metropolis",
                save_warmup=False, outdir="frames"):

        self.T = T
        if method == "wolff":
            self.precompute_bond_probabilities()

        # очистить любые хвосты от прошлых запусков
        self.acc.energy.clear()
        self.acc.magnetization.clear()
        self.acc.susceptibility.clear()
        self.acc.m2_array.clear()
        self.acc.m_abs_array.clear()

        if self.progress:
            print("Starting warmup phase...")

        for step in tqdm(range(warmup_steps), disable=not self.progress):
            if method == "metropolis":
                self.metropolis_step()
                if save_warmup and (step % 1_000 == 0):
                    self.vis.plot_coords_save(step // 100, output_dir=outdir)
            else:  # wolff
                if save_warmup:
                    (cluster_idx, edges_fm, edges_afm) = self.wolff_step(return_cluster=True)
                    self.vis.plot_cluster_save(cluster_idx, edges_fm, edges_afm, step, self.lattice.N, output_dir=outdir)
                else:
                    self.wolff_step(return_cluster=False)

            # <<< ВАЖНО: собирать warmup-ряд для τ_int >>>
            self.acc.sample_warmup(step + 1, self.E, self.M)

        tauE = float(self.acc.energy_tau_int)
        tauM = float(self.acc.magnetization_tau_int)

        if save_warmup:
            self.vis.create_gif_from_frames(
                output_dir=outdir,
                output_file=f"{method}_{T}_warmup.gif",
                fps=self.vis.fps
            )

        if self.progress:
            print("Starting production phase...")

        # очистить серии, чтобы не смешивать с warmup
        self.acc.energy.clear()
        self.acc.magnetization.clear()
        self.acc.susceptibility.clear()
        self.acc.m2_array.clear()
        self.acc.m_abs_array.clear()

        self.accept = 0  # имеет смысл только для Metropolis

        for _ in tqdm(range(steps), disable=not self.progress):
            if method == "metropolis":
                self.metropolis_step()
            else:
                self.wolff_step()
            self.acc.sample_production(self.E, self.M, self.chi, self.M2, self.mabs)

        # для wolff лучше не интерпретировать acceptance_rate
        self.acceptance_rate = (self.accept / steps) if method == "metropolis" else float("nan")

        # по желанию возвращайте τ для расчёта ошибок снаружи
        return {"tau_E": tauE, "tau_M": tauM, "accept": self.acceptance_rate}

    def precompute_bond_probabilities(self):
        J = self.lattice.interaction_matrix
        beta = 1.0 / self.T

        deltaE_same = -2 * J
        deltaE_opp = +2 * J

        self.padd_same = 1 - np.exp(np.minimum(0, -beta * deltaE_same))
        self.padd_opp = 1 - np.exp(np.minimum(0, -beta * deltaE_opp))

   
    def wolff_step(self, return_cluster=False):
        spins = self.lattice.magnetic_moments
        J     = self.lattice.interaction_matrix
        N     = self.lattice.N
        seed  = int(self.rng.integers(N))

        cluster = {seed}
        to_check = [seed]
        visited  = np.zeros(N, dtype=bool)
        visited[seed] = True

        # edges_used = [] 
        edges_fm, edges_afm = [],[]

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
                    # edges_used.append((i, j))  # <— record the bond we actually used

                    if return_cluster:
                        if J[i,j]>0:
                            edges_fm.append((i,j))
                        else:
                            edges_afm.append((i,j))


        for idx in cluster:
            spins[idx] *= -1

        self.E = self.acc.compute_energy()
        self.M = np.mean(spins)
        self.mabs = np.abs(self.M)

        self.M2 = (self.M)**2
        self.E = self.compute_energy()  
        self.chi = self.lattice.N * (1 - self.M**2) / self.T

        if return_cluster:
            return (np.fromiter(cluster, dtype=int), edges_fm, edges_afm)              


