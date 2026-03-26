from typing import Dict, Set, Tuple, Optional, List, Any
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


class MonteCarlo:
    """
    Monte Carlo simulator for diluted magnetic systems with RKKY interactions.
    
    Implements Metropolis and Wolff cluster algorithms for equilibration.
    Tracks energy, magnetization, and susceptibility.
    
    Attributes:
        lattice: Lattice instance (system geometry + interaction matrix)
        acc: Accumulator instance (statistics)
        T: Current temperature (None until run_loop called)
        E: Current system energy
        M: Current total magnetization (mean of spins)
        M2: M² (for computing susceptibility)
        mabs: |M| (absolute magnetization)
        chi: Current magnetic susceptibility
        step: Current MC step number
        accept: Number of accepted Metropolis moves
    """
    
    def __init__(self, lattice, progress: bool = False) -> None:
        """
        Initialize Monte Carlo simulator.
        
        Args:
            lattice: Lattice instance with interaction_matrix computed
            progress: If True, show progress bars during simulation
        """
        self.lattice = lattice
        self.acc = self._create_accumulator()
        self.vis = None  # Optional visualization
        
        # RNG for MC decisions
        self.rng = np.random.default_rng(seed=42)
        self.progress = progress
        
        # Simulation state
        self.T: Optional[float] = None
        self.E = self.compute_energy()
        
        # Magnetization and susceptibility
        spins = self.lattice.magnetic_moments
        self.M = float(np.mean(spins))
        self.M2 = float(self.M ** 2)
        self.M4 = float(self.M2 ** 2)  # M^4 for Binder parameter
        self.mabs = float(np.abs(self.M))
        self.chi = 0.0  # Will be set first time through
        
        # Tracking
        self.step = 0
        self.accept = 0
        
        # Precomputed bond probabilities for Wolff (computed on demand)
        self.padd_same: Optional[NDArray[np.float64]] = None
        self.padd_opp: Optional[NDArray[np.float64]] = None

    def _create_accumulator(self):
        """Create accumulator instance."""
        from accumulator import Accumulator
        return Accumulator(self.lattice)

    def compute_energy(self) -> float:
        """
        Compute total system energy.
        
        E = s_i * J_ij * s_j (double counts, but diagonal is zero)
        
        Returns:
            Total energy
        """
        return self.acc.compute_energy()

    def metropolis_step(self) -> None:
        """
        Execute one Metropolis Monte Carlo step.
        
        - Select random spin i
        - Compute energy change ΔE for flipping s_i using ΔE = -2*s_i*(J_i·s)
        - Accept with probability min(1, exp(-βΔE))
        - Update energy and magnetization
        
        Physics: Detailed balance is preserved (reversible)
        
        Energy formula derivation:
            H = 0.5 * s^T J s
            When flipping spin i: s_i → -s_i
            ΔH = -2 * s_i * (J_i · s)  [derived from quadratic form expansion]
        """
        if self.T is None:
            raise RuntimeError("Set temperature (self.T) before calling metropolis_step()")
        
        i = self.rng.integers(self.lattice.N)
        spins = self.lattice.magnetic_moments
        
        # Energy change for flipping spin i: ΔE = -2 * s_i * (J_i · s)
        # This formula is derived from H = 0.5 * s^T J s (the true Hamiltonian)
        dE = -2.0 * spins[i] * float(self.lattice.interaction_matrix[i] @ spins)
        
        # Metropolis acceptance criterion
        if dE <= 0 or self.rng.random() < np.exp(-dE / self.T):
            spins[i] *= -1
            self.E += dE
            self.accept += 1
        
        # Update observables
        self.M = float(np.mean(spins))
        self.M2 = float(self.M ** 2)
        self.M4 = float(self.M2 ** 2)  # M^4 for Binder parameter
        self.mabs = float(np.abs(self.M))
        self.chi = float(self.lattice.N * (1 - self.M**2) / self.T)

    def precompute_bond_probabilities(self) -> None:
        """
        Precompute Wolff cluster addition probabilities.
        
        For efficiency, compute P_add(i,j | s_i = s_j) and P_add(i,j | s_i ≠ s_j)
        once at start of temperature point, reuse throughout.
        """
        if self.T is None:
            raise RuntimeError("Set temperature before calling precompute_bond_probabilities()")
        
        J = self.lattice.interaction_matrix
        beta = 1.0 / self.T

        # If spins are aligned (same), energy cost to add bond is -2J
        deltaE_same = -2.0 * J
        # If spins are antialigned (opposite), energy cost is +2J
        deltaE_opp = +2.0 * J

        # P_add = 1 - exp(-β min(0, ΔE))
        self.padd_same = 1.0 - np.exp(np.minimum(0, -beta * deltaE_same))
        self.padd_opp = 1.0 - np.exp(np.minimum(0, -beta * deltaE_opp))

    def wolff_step(
        self, 
        return_cluster: bool = False
    ) -> Optional[Tuple[NDArray[np.int32], List[Tuple[int, int]], List[Tuple[int, int]]]]:
        """
        Execute one Wolff cluster flip.
        
        - Define cluster as set of adjacent spins with same magnetic moment
        - Build cluster via BFS starting from random seed
        - Acceptance probability: P = 1 - exp(-β min(0, ΔE_bond))
        - Flip all spins in cluster
        - Recalculate energy
        
        Args:
            return_cluster: If True, return (cluster_indices, FM_edges, AFM_edges)
            
        Returns:
            If return_cluster=True: tuple of (cluster, FM_edges, AFM_edges)
            If return_cluster=False: None
            
        Notes:
            - Cluster flip has 100% acceptance in ensemble (but doesn't update M, chi explicitly)
            - Faster equilibration than Metropolis at low T
        """
        if self.T is None:
            raise RuntimeError("Set temperature before calling wolff_step()")
        
        if self.padd_same is None or self.padd_opp is None:
            self.precompute_bond_probabilities()
        
        spins = self.lattice.magnetic_moments
        J = self.lattice.interaction_matrix
        N = self.lattice.N
        
        # Start from random seed
        seed = int(self.rng.integers(N))
        cluster: Set[int] = {seed}
        to_check: List[int] = [seed]
        visited: NDArray[np.bool_] = np.zeros(N, dtype=bool)
        visited[seed] = True
        
        # Track edges for visualization
        edges_fm: List[Tuple[int, int]] = []
        edges_afm: List[Tuple[int, int]] = []
        
        # Build cluster via BFS
        while to_check:
            i = to_check.pop()
            Si = spins[i]
            
            # Find all neighbors (connected via non-zero J)
            neighbors = np.nonzero(J[i])[0]
            
            for j in neighbors:
                if visited[j]:
                    continue
                
                Sj = spins[j]
                
                # Determine addition probability based on spin alignment
                if Si == Sj:
                    padd = self.padd_same[i, j]
                else:
                    padd = self.padd_opp[i, j]
                
                # Add to cluster with probability padd
                if self.rng.random() < padd:
                    visited[j] = True
                    cluster.add(j)
                    to_check.append(j)
                    
                    # Track edge type for visualization
                    if return_cluster:
                        if J[i, j] > 0:
                            edges_fm.append((i, j))
                        else:
                            edges_afm.append((i, j))
        
        # Flip all spins in cluster
        for idx in cluster:
            spins[idx] *= -1
        
        # Recalculate energy (after cluster flip)
        self.E = self.compute_energy()
        
        # Update observables
        self.M = float(np.mean(spins))
        self.mabs = float(np.abs(self.M))
        self.M2 = float(self.M ** 2)
        self.M4 = float(self.M2 ** 2)  # M^4 for Binder parameter
        self.chi = float(self.lattice.N * (1 - self.M**2) / self.T)
        
        if return_cluster:
            return (np.array(sorted(cluster), dtype=np.int32), edges_fm, edges_afm)
        
        return None

    def run_loop(
        self,
        warmup_steps: int,
        steps: int,
        T: float,
        method: str = "metropolis",
        save_warmup: bool = False,
        outdir: str = "frames"
    ) -> Dict[str, float]:
        """
        Execute full simulation (warmup + production).
        
        Args:
            warmup_steps: Number of equilibration steps
            steps: Number of production steps (after warmup)
            T: Temperature
            method: "metropolis" or "wolff"
            save_warmup: If True, save snapshots during warmup
            outdir: Output directory for frames
            
        Returns:
            Dictionary with keys:
                - "tau_E": Integrated autocorrelation time for energy
                - "tau_M": Integrated autocorrelation time for magnetization
                - "accept": Acceptance rate (only for Metropolis)
                
        Notes:
            - Clears all previous data from self.acc
            - Computes autocorrelation times during warmup for efficiency estimate
        """
        self.T = T
        
        # Precompute probabilities if using Wolff
        if method == "wolff":
            self.precompute_bond_probabilities()
        
        # Clear previous data
        self.acc.energy.clear()
        self.acc.magnetization.clear()
        self.acc.susceptibility.clear()
        self.acc.m2_array.clear()
        self.acc.m4_array.clear()
        self.acc.m_abs_array.clear()
        
        if self.progress:
            print(f"Starting warmup phase ({warmup_steps} steps) at T={T}...")
        
        # ==================== WARMUP PHASE ====================
        for step in tqdm(range(warmup_steps), disable=not self.progress, desc="Warmup"):
            if method == "metropolis":
                self.metropolis_step()
                
                # Optional: save snapshots
                if save_warmup and (step % 1_000 == 0):
                    if hasattr(self, 'vis') and self.vis is not None:
                        self.vis.plot_coords_save(step // 100, output_dir=outdir)
            
            elif method == "wolff":
                if save_warmup:
                    cluster_idx, edges_fm, edges_afm = self.wolff_step(return_cluster=True)
                    if hasattr(self, 'vis') and self.vis is not None:
                        self.vis.plot_cluster_save(
                            cluster_idx, edges_fm, edges_afm, step,
                            self.lattice.N, output_dir=outdir
                        )
                else:
                    self.wolff_step(return_cluster=False)
            
            else:
                raise ValueError(f"Unknown method: {method}. Use 'metropolis' or 'wolff'")
            
            # Record warmup sample
            self.acc.sample_warmup(step + 1, self.E, self.M)
        
        # Extract autocorrelation times from warmup
        tauE = float(self.acc.energy_tau_int)
        tauM = float(self.acc.magnetization_tau_int)
        
        if self.progress:
            print(f"Warmup complete. τ_E={tauE:.2f}, τ_M={tauM:.2f}")
            print(f"Starting production phase ({steps} steps)...")
        
        # Create GIF from warmup frames if saved
        if save_warmup and hasattr(self, 'vis') and self.vis is not None:
            self.vis.create_gif_from_frames(
                output_dir=outdir,
                output_file=f"{method}_{T}_warmup.gif",
                fps=self.vis.fps
            )
        
        # ==================== PRODUCTION PHASE ====================
        # Clear warmup data
        self.acc.energy.clear()
        self.acc.magnetization.clear()
        self.acc.susceptibility.clear()
        self.acc.m2_array.clear()
        self.acc.m4_array.clear()
        self.acc.m_abs_array.clear()
        
        self.accept = 0  # Reset acceptance counter for Metropolis
        
        for _ in tqdm(range(steps), disable=not self.progress, desc="Production"):
            if method == "metropolis":
                self.metropolis_step()
            else:  # wolff
                self.wolff_step(return_cluster=False)
            
            # Record production sample
            self.acc.sample_production(self.E, self.M, self.chi, M2=self.M2, M4=self.M4)
        
        # Compute acceptance rate
        self.acceptance_rate = (self.accept / steps) if method == "metropolis" else float("nan")
        
        return {
            "tau_E": tauE,
            "tau_M": tauM,
            "accept": self.acceptance_rate
        }
