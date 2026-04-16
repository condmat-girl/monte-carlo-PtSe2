import numpy as np
from tqdm import tqdm
from accumulator import Accumulator
from visualization import Visualization
# from visualization_ising import Visualization_Ising


class MonteCarlo:

    def __init__(self, lattice):
        self.lattice = lattice
        self.acc = Accumulator(lattice)
        self.vis = Visualization(lattice, self.acc)
        # self.vis_is = Visualization_Ising(lattice,self.acc)
        self.rng = np.random.default_rng(seed=42)

        self.T    = None
        self.E    = self.acc.compute_energy()
        self.M    = self.lattice.magnetic_moments.mean()
        self.M2   = None
        self.M4   = None
        self.mabs = None
        self.chi  = None
        self._E2  = None
        self.step = 0
        self.accept = 0

    def compute_energy(self):
        return self.acc.compute_energy()

    def compute_pair_correlation(self):
        return self.acc.compute_pair_correlation()

    # ── derived observables ───────────────────────────────────────────────────
    def _update_observables(self):
        spins     = self.lattice.magnetic_moments
        self.M    = np.mean(spins)
        mabs      = np.abs(self.M)
        self.M2   = self.M ** 2
        self.M4   = self.M ** 4
        self.mabs = mabs
        self.chi  = self.lattice.N * (self.M2 - mabs ** 2) / self.T
        self._E2  = self.E ** 2

    # ── single-spin Metropolis ────────────────────────────────────────────────
    def metropolis_step(self):
        spins = self.lattice.magnetic_moments
        i     = self.rng.integers(self.lattice.N)
        dE    = -2 * spins[i] * (self.lattice.interaction_matrix[i] @ spins)
        if dE <= 0 or self.rng.random() < np.exp(-dE / self.T):
            spins[i] *= -1
            self.E   += dE
            self.accept += 1
        self._update_observables()

    # ── main loop ─────────────────────────────────────────────────────────────
    def run_loop(self, warmup_steps, steps, T, method="metropolis",
                save_warmup=False, outdir="frames"):

        self.T = T

        if method == "wolff":
            self.precompute_bond_probabilities()

        for step in tqdm(range(warmup_steps), disable=False):
            if method == "metropolis":
                self.metropolis_step()
                if save_warmup and (step % 1_000 == 0):
                    self.vis.plot_coords_save(
                        int(step//1_00),
                        output_dir=outdir
                    )
            elif method == "wolff":
                if save_warmup:
                    (cluster_idx, edges_fm, edges_afm) = self.wolff_step(return_cluster=True)
                    self.vis.plot_cluster_save(cluster_idx, edges_fm, edges_afm, step, self.lattice.N, output_dir=outdir)
                else:
                    self.wolff_step(return_cluster=False)

        if save_warmup:
            self.vis.create_gif_from_frames(
                output_dir=outdir,
                output_file=str(method) + "_" + str(T) + "_warmup.gif",
                fps=self.vis.fps
            )

        self.accept = 0
        for _ in tqdm(range(steps), disable=False):
            if method == "metropolis":
                self.metropolis_step()
            else:
                self.wolff_step()
            self.acc.sample_production(
                E    = self.E,
                M    = self.M,
                chi  = self.chi,
                m2   = self.M2,
                m4   = self.M4,
                mabs = self.mabs,
                e2   = self._E2,
            )

        self.acceptance_rate = self.accept / steps

    # ── Wolff bond probabilities ──────────────────────────────────────────────
    def precompute_bond_probabilities(self):
        J    = self.lattice.interaction_matrix
        beta = 1.0 / self.T

        self.padd_same = 1 - np.exp(np.minimum(0, -beta * (-2 * J)))
        self.padd_opp  = 1 - np.exp(np.minimum(0, -beta * (+2 * J)))

    # ── Wolff cluster ─────────────────────────────────────────────────────────
    def wolff_step(self, return_cluster=False):
        spins = self.lattice.magnetic_moments
        J     = self.lattice.interaction_matrix
        N     = self.lattice.N
        seed  = int(self.rng.integers(N))

        cluster  = {seed}
        to_check = [seed]
        visited  = np.zeros(N, dtype=bool)
        visited[seed] = True
        edges_fm, edges_afm = [], []

        while to_check:
            i  = to_check.pop()
            Si = spins[i]
            for j in np.nonzero(J[i])[0]:
                if visited[j]:
                    continue
                padd = self.padd_same[i, j] if Si == spins[j] else self.padd_opp[i, j]
                if self.rng.random() < padd:
                    visited[j] = True
                    cluster.add(j)
                    to_check.append(j)
                    if return_cluster:
                        (edges_fm if J[i, j] > 0 else edges_afm).append((i, j))

        for idx in cluster:
            spins[idx] *= -1

        self.E = self.acc.compute_energy()
        self._update_observables()

        if return_cluster:
            return (np.fromiter(cluster, dtype=int), edges_fm, edges_afm)
