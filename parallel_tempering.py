import copy
import numpy as np
from typing import Sequence, Dict, Any, List

from lattice import Lattice
from monte_carlo import MonteCarlo

class ParallelTempering:

    def __init__(
        self,
        base_lattice: Lattice,
        temperatures: Sequence[float],
        method: str = "metropolis",   # metropolis  wolff
        steps_per_exchange: int = 200,
        warmup_per_replica: int = 1_000,
        progress: bool = False,
        rng_seed: int = 1234,
    ):
        self.method = method
        self.steps_per_exchange = int(steps_per_exchange)
        self.warmup_per_replica = int(warmup_per_replica)
        self.progress = progress

        self.rng = np.random.default_rng(rng_seed)

        self.T: np.ndarray = np.asarray(temperatures, dtype=float)
        self.beta: np.ndarray = 1.0 / self.T
        self.n_rep = len(self.T)

        self.mcs: List[MonteCarlo] = []
        for k in range(self.n_rep):
            lat_k = copy.deepcopy(base_lattice)
            mc_k = MonteCarlo(lat_k, progress=False)
            mc_k.T = float(self.T[k])
            if self.method == "wolff":
                mc_k.precompute_bond_probabilities()  
            self.mcs.append(mc_k)

        self.energy_trace: List[List[float]] = [[] for _ in range(self.n_rep)]
        self.magn_trace:   List[List[float]] = [[] for _ in range(self.n_rep)]

        self.swap_proposals = np.zeros(self.n_rep - 1, dtype=int)
        self.swap_accepts   = np.zeros(self.n_rep - 1, dtype=int)

    def _sweep(self, mc: MonteCarlo, nsteps: int):
        if self.method == "metropolis":
            for _ in range(nsteps):
                mc.metropolis_step()
        elif self.method == "wolff":
            for _ in range(nsteps):
                mc.wolff_step()
        else:
            raise ValueError("method must be 'metropolis' or 'wolff'")

    def _attempt_swap(self, i: int):
        mc_i   = self.mcs[i]
        mc_ip1 = self.mcs[i + 1]
        Ei, Eip1 = float(mc_i.E), float(mc_ip1.E)
        bet_i, bet_ip1 = float(self.beta[i]), float(self.beta[i + 1])

        self.swap_proposals[i] += 1
        logA = (bet_i - bet_ip1) * (Eip1 - Ei)
        if logA >= 0 or self.rng.random() < np.exp(logA):
            self.mcs[i], self.mcs[i + 1] = self.mcs[i + 1], self.mcs[i]
            self.swap_accepts[i] += 1

            if self.method == "wolff":
                self.mcs[i].T     = float(self.T[i])
                self.mcs[i].precompute_bond_probabilities()
                self.mcs[i + 1].T = float(self.T[i + 1])
                self.mcs[i + 1].precompute_bond_probabilities()

    def _record_observables(self):
        for i, mc in enumerate(self.mcs):
            self.energy_trace[i].append(float(mc.E))
            self.magn_trace[i].append(float(mc.M))

    def run(self, n_exchange_epochs: int) -> Dict[str, Any]:

        for i, mc in enumerate(self.mcs):
            if self.method == "wolff":
                mc.T = float(self.T[i])
                mc.precompute_bond_probabilities()
            self._sweep(mc, self.warmup_per_replica)

        for epoch in range(int(n_exchange_epochs)):
            # эволюция всех реплик на своих T
            for mc in self.mcs:
                self._sweep(mc, self.steps_per_exchange)

            start = epoch % 2
            for i in range(start, self.n_rep - 1, 2):
                self._attempt_swap(i)

            self._record_observables()

            if self.progress and (epoch % 10 == 0):
                acc = np.divide(self.swap_accepts, np.maximum(1, self.swap_proposals))
                print(f"[epoch {epoch}] swap acc ~ {acc.mean():.3f}")

        swap_acc = np.divide(self.swap_accepts, np.maximum(1, self.swap_proposals))
        return {
            "T": self.T.copy(),
            "energy_trace": self.energy_trace,
            "magnetization_trace": self.magn_trace,
            "swap_proposals": self.swap_proposals.copy(),
            "swap_accepts": self.swap_accepts.copy(),
            "swap_accept_rate_by_pair": swap_acc,
            "swap_accept_rate_mean": float(np.mean(swap_acc)) if swap_acc.size else np.nan,
        }

