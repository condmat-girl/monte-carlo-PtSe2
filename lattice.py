from typing import Tuple, Optional
import numpy as np
from scipy.special import j0, j1, y0, y1
from numpy.typing import NDArray


class Lattice:
    """
    Triangular lattice with RKKY interactions and dilution.
    
    Attributes:
        rows: Number of lattice rows
        cols: Number of lattice columns  
        doping: Probability of site occupation (0 < doping ≤ 1)
        kf: Fermi wavevector (scales RKKY oscillation)
        J0: Overall RKKY interaction strength
        N: Number of occupied sites
        Lx: System size in x-direction
        Ly: System size in y-direction (accounting for triangular geometry)
        lattice_points: Coordinates of occupied sites, shape (N, 2)
        coords: Alias for lattice_points
        magnetic_moments: Spin values (±1), shape (N,)
        interaction_matrix: RKKY coupling matrix, shape (N, N)
    """
    
    rows: int
    cols: int
    doping: float
    kf: float
    J0: float
    N: int
    Lx: float
    Ly: float
    
    lattice_points: NDArray[np.float64]
    coords: NDArray[np.float64]
    magnetic_moments: NDArray[np.int8]
    interaction_matrix: NDArray[np.float64]
    distances: NDArray[np.float64]
    
    # Histogram data for pair correlation
    hist: NDArray[np.int64]
    bin_edges: NDArray[np.float64]
    bin_centers: NDArray[np.float64]
    
    # Upper triangular indices for efficient pairwise operations
    i_idx: NDArray[np.intp]
    j_idx: NDArray[np.intp]
    r_ij: NDArray[np.float64]

    def __init__(
        self, 
        rows: int, 
        cols: int, 
        doping: float,
        kf: float = 1.0, 
        J0: float = 1.0,
        seed: int = 12
    ) -> None:
        """
        Initialize triangular lattice with RKKY interactions.
        
        Args:
            rows: Number of lattice rows
            cols: Number of lattice columns
            doping: Occupation probability for each site (0, 1]
            kf: Fermi wavevector (default: 1.0)
            J0: RKKY interaction strength (default: 1.0)
            seed: Random seed for reproducibility (default: 12)
            
        Raises:
            ValueError: If doping ≤ 0 or no sites are occupied
        """
        if doping < 0 or doping > 1.0:
            raise ValueError(f"Doping must be in [0, 1], got {doping}")
        
        self.rng = np.random.default_rng(seed=seed)
        self.kf = kf
        self.J0 = J0
        
        self.rows = rows
        self.cols = cols
        self.Lx = float(cols)
        self.Ly = float(rows) * (np.sqrt(3) / 2)
        
        self.generate_lattice(rows, cols, doping)

    def generate_lattice(self, rows: int, cols: int, doping: float) -> None:
        """
        Generate random diluted triangular lattice.
        
        Uses minimum image convention for periodic boundary conditions.
        Empty sites are discarded (not represented as zeros in matrix).
        
        Args:
            rows: Number of lattice rows
            cols: Number of lattice columns
            doping: Occupation probability per site
            
        Raises:
            ValueError: If no sites are occupied after doping
        """
        self.rows = rows
        self.cols = cols
        self.Lx = float(cols)
        self.Ly = float(rows) * (np.sqrt(3) / 2)

        # Generate occupied sites
        points = []
        for i in range(rows):
            for j in range(cols):
                if self.rng.random() < doping:
                    x = float(j) + 0.5 * (i % 2)
                    y = float(i) * (np.sqrt(3) / 2)
                    points.append((x, y))

        if not points:
            raise ValueError(
                f"No occupied sites after doping={doping}. "
                f"Try increasing doping or lattice size."
            )

        self.N = len(points)
        self.lattice_points = np.array(points, dtype=np.float64)
        self.coords = self.lattice_points  # Alias
        
        self.magnetic_moments = self.initialize_magnetic_moments()
        self.interaction_matrix = self.compute_rkky_matrix()

    def initialize_magnetic_moments(self) -> NDArray[np.int8]:
        """
        Initialize random spins (±1) for all occupied sites.
        
        Returns:
            Array of spins, shape (N,)
        """
        return (2 * (self.rng.random(self.N) > 0.5) - 1).astype(np.int8)

    def rkky_interaction_2d(self, r: float) -> float:
        """
        RKKY interaction strength at distance r.
        
        Formula: J(r) = -J₀ [J₀(kf*r)*Y₀(kf*r) + J₁(kf*r)*Y₁(kf*r)]
        
        Args:
            r: Pairwise distance
            
        Returns:
            Interaction strength J(r)
        """
        if r == 0:
            return 0.0
        
        x = self.kf * r
        interaction = -self.J0 * (
            j0(x) * y0(x) + j1(x) * y1(x)
        )
        return float(interaction)

    def distance(
        self, 
        r1: Tuple[float, float], 
        r2: Tuple[float, float]
    ) -> float:
        """
        Euclidean distance with periodic boundary conditions.
        
        Uses minimum image convention: accounts for lattice wrap-around.
        
        Args:
            r1: First position (x, y)
            r2: Second position (x, y)
            
        Returns:
            Minimum image distance
        """
        dx = r1[0] - r2[0]
        dy = r1[1] - r2[1]
        
        # Minimum image convention
        dx -= self.Lx * np.round(dx / self.Lx)
        dy -= self.Ly * np.round(dy / self.Ly)
        
        return float(np.hypot(dx, dy))

    def compute_rkky_matrix(self) -> NDArray[np.float64]:
        """
        Compute RKKY interaction matrix for all pairs.
        
        Returns:
            Symmetric (N×N) matrix with J[i,j] = J[j,i], diagonal zero
            
        Notes:
            - Sets diagonal to zero (no self-interaction)
            - Computes upper triangle indices for efficient pairwise operations
            - Generates histogram of pairwise distances for analysis
        """
        # Compute pairwise distances (O(N²) but necessary)
        distances = np.zeros((self.N, self.N), dtype=np.float64)
        for i in range(self.N):
            for j in range(self.N):
                distances[i, j] = self.distance(
                    self.lattice_points[i],
                    self.lattice_points[j]
                )

        # Vectorized RKKY calculation
        x = self.kf * distances
        interaction_matrix = -self.J0 * (
            j0(x) * y0(x) +
            j1(x) * y1(x)
        )
        
        # Remove self-interaction
        np.fill_diagonal(interaction_matrix, 0)

        self.distances = distances
        
        # Pre-compute upper triangle indices for pair operations
        self.i_idx, self.j_idx = np.triu_indices(self.N, k=1)
        self.r_ij = distances[self.i_idx, self.j_idx]
        
        # Histogram for pair correlation analysis
        self.hist, self.bin_edges = np.histogram(self.r_ij, bins="fd")
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        return interaction_matrix

    def compute_pair_correlation(self) -> NDArray[np.float64]:
        """
        Compute spin-spin correlation for all pairs in upper triangle.
        
        Returns:
            Array of S_i * S_j products, shape (M,) where M = N(N-1)/2
        """
        s = self.magnetic_moments
        return s[self.i_idx] * s[self.j_idx]
