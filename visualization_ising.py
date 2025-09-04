# visualization.py  – all plotting, no stored data duplicates
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import imageio.v2 as imageio
import glob
# visualization.py (top)
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection

# Define once (module-level)
SPIN_CMAP = ListedColormap(["#094B55", "#88c2ca"]) 



class Visualization_Ising:

    def __init__(self, lattice, accumulator):
        self.lattice = lattice
        self.acc     = accumulator
        self.outdir    = 'pic'
        self.gif_file  = 'evolv'
        self.fps       = 12
        os.makedirs(self.outdir, exist_ok=True)


    def plot_lattice(self):
        rows, cols = self.lattice.rows, self.lattice.cols
        full_points = np.array(
            [(j + 0.5 * (i % 2), i * (np.sqrt(3) / 2))
             for i in range(rows) for j in range(cols)]
        )

        triang = tri.Triangulation(full_points[:, 0], full_points[:, 1])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.triplot(triang, color="lightgray", linewidth=0.5)
        ax.scatter(self.lattice.coords[:, 0],
                   self.lattice.coords[:, 1],
                   c=self.lattice.magnetic_moments, cmap="bwr", s=50)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show()

    def plot_magnetization(self):
        if not self.acc.magnetization:
            raise ValueError("Run monte_carlo_loop first!")

        m   = np.asarray(self.acc.magnetization)
        acf = self.acc.incremental_autocorrelation(
            m, m.mean(), m.var() if m.var() > 0 else 1e-12
        )
        tau = self.acc.calculate_autocorrelation_time(acf)
        err = self.acc.calculate_error(m, tau)

        steps = np.arange(len(m))
        plt.figure(figsize=(5, 4))
        plt.errorbar(steps, m, yerr=err, fmt="-",
                     label=f"τ = {tau:.2f}\n⟨M⟩ = {m.mean():.3f}")
        plt.ylabel("Magnetization")
        plt.legend()
        plt.show()

    def plot_pair_correlation(self):
        r = self.lattice.r_ij
        corr = self.acc.compute_pair_correlation()

        plt.figure(figsize=(6, 4))
        plt.plot(r, corr, ".", alpha=0.6)
        plt.xlabel(" r")
        plt.ylabel("⟨Sᵢ Sⱼ⟩(r)")
        plt.title("Pair Correlation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_ft_pair_correlation(self):
        corr = self.acc.compute_pair_correlation()
        r    = self.lattice.r_ij
        q_array = np.linspace(0, 2 * np.pi, 151)
        ft = np.array([np.sum(corr * np.exp(1j * r * q))
                       for q in q_array]) / len(corr)

        plt.figure(figsize=(6, 4))
        plt.plot(q_array, ft.real, label="real")
        plt.plot(q_array, ft.imag, label="imag")
        plt.xlabel("q")
        plt.ylabel("⟨S_i S_j⟩(q)")
        plt.legend()
        plt.show()



    def plot_autocorrelation(self):
        """
        Plot the radially averaged spin–spin autocorrelation ⟨SᵢSⱼ⟩(r).
        """
        r = self.acc.bin_centers
        corr = self.acc.binned_pair_correlation

        plt.figure(figsize=(6, 4))
        plt.plot(r, corr, "-", lw=2)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Distance r")
        plt.ylabel("⟨Sᵢ Sⱼ⟩(r)")
        plt.title("Spin Autocorrelation Function")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def plot_lattice(spins, Lx, Ly, step=None):
        """
        Plot the current 2D Ising lattice configuration.
        Spins are expected as a flat array of shape (Lx * Ly,)
        """
        lattice_2D = spins.reshape((Ly, Lx))
        plt.imshow(lattice_2D, cmap='bwr', vmin=-1, vmax=1)
        plt.axis('off')
        if step is not None:
            plt.title(f"Step {step}")
        plt.show()

    @staticmethod
    def plot_lattice(spins, Lx, Ly, step=None):
        """
        Plot the current 2D Ising lattice configuration.
        Spins are expected as a flat array of shape (Lx * Ly,)
        """
        lattice_2D = spins.reshape((Ly, Lx))
        plt.imshow(
            lattice_2D,
            cmap=SPIN_CMAP,
            vmin=-1, vmax=1,
            interpolation="nearest",
            aspect="equal"
        )
        plt.axis('off')
        if step is not None:
            plt.title(f"Step {step}")
        plt.show()

    @staticmethod
    def plot_lattice_save(spins, Lx, Ly, step, output_dir="frames"):
        """
        Save the current 2D Ising lattice configuration as a PNG.
        Creates the output directory if it doesn't exist.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        lattice_2D = spins.reshape((Ly, Lx))
        plt.imshow(
            lattice_2D,
            cmap=SPIN_CMAP,
            vmin=-1, vmax=1,
            interpolation="nearest",
            aspect="equal"
        )
        plt.axis('off')
        plt.title(f"Step {step}")
        plt.savefig(os.path.join(output_dir, f"step_{step:04d}.png"), dpi=150)
        plt.close()

    @staticmethod
    def create_gif_from_frames(output_dir="frames", output_file="ising_simulation.gif", fps=12):
        """
        Create an animated GIF from saved lattice images.
        """
        images = [imageio.imread(fname) for fname in sorted(glob.glob(f"{output_dir}/step_*.png"))]
        imageio.mimsave(output_file, images, fps=fps)
        print(f"GIF saved to {output_file}")


    ### boundary segments 

    def _square_boundary_segments(self, cluster_idx):

        Lx, Ly = self.lattice.Lx, self.lattice.Ly
        mask1d = np.zeros(self.lattice.N, dtype=bool)
        mask1d[cluster_idx] = True
        mask = mask1d.reshape((Ly, Lx))  # rows=y, cols=x

        segs = []
        for y in range(Ly):
            for x in range(Lx):
                if not mask[y, x]:
                    continue

                # left neighbor
                nx = (x - 1) % Lx
                if not mask[y, nx]:
                    segs.append([(x - 0.5, y - 0.5), (x - 0.5, y + 0.5)])

                # right neighbor
                nx = (x + 1) % Lx
                if not mask[y, nx]:
                    segs.append([(x + 0.5, y - 0.5), (x + 0.5, y + 0.5)])

                # bottom neighbor
                ny = (y - 1) % Ly
                if not mask[ny, x]:
                    segs.append([(x - 0.5, y - 0.5), (x + 0.5, y - 0.5)])

                # top neighbor
                ny = (y + 1) % Ly
                if not mask[ny, x]:
                    segs.append([(x - 0.5, y + 0.5), (x + 0.5, y + 0.5)])

        return segs



    def _cluster_boundary_segments(self, cluster_mask):
        """
        Build line segments along the boundary between cluster and non-cluster nodes
        using the triangulation of all lattice points.
        """
        coords = self.lattice.coords
        triang = tri.Triangulation(coords[:, 0], coords[:, 1])

        segments = []
        tri_points = triang.triangles 

        seen = set()
        edges = [(0,1), (1,2), (2,0)]
        for t in tri_points:
            for a, b in edges:
                i, j = t[a], t[b]
                if cluster_mask[i] ^ cluster_mask[j]:
                    # undirected edge key
                    key = (i, j) if i < j else (j, i)
                    if key in seen:
                        continue
                    seen.add(key)
                    segments.append([coords[i], coords[j]])
        return segments

    def plot_lattice_with_cluster(self, cluster_idx, step=None):
        """
        Scatter spins by sign; overlay the boundary of the Wolff cluster.
        """
        spins  = self.lattice.magnetic_moments
        coords = self.lattice.coords
        cluster_mask = np.zeros(len(spins), dtype=bool)
        cluster_mask[cluster_idx] = True

        # base scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(coords[:, 0], coords[:, 1],
                    c=spins, cmap=SPIN_CMAP, vmin=-1, vmax=1,
                    s=50, linewidths=0, zorder=1)

        # draw faint mesh (optional)
        triang = tri.Triangulation(coords[:, 0], coords[:, 1])
        plt.triplot(triang, color="#d1d5db", linewidth=0.5, zorder=0)

        # boundary segments
        segments = self._square_boundary_segments(cluster_mask)
        if segments:
            lc = LineCollection(segments, linewidths=2.0, colors="#a99f17", zorder=2)  # warm yellow
            plt.gca().add_collection(lc)

        if step is not None:
            plt.title(f"Wolff step {step} — cluster size {cluster_mask.sum()}")
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _save_current_figure(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_lattice_with_cluster_save(self, cluster_idx, step, output_dir="frames"):
        self.plot_lattice_with_cluster(cluster_idx=cluster_idx, step=step)
        self._save_current_figure(os.path.join(output_dir, f"step_{step:04d}.png"))

    def plot_grid_with_cluster_save(self, cluster_idx, step, output_dir="frames"):
        """
        For Ising grids: save a frame with the Wolff cluster outlined.
        Uses Lx, Ly and flat spins array; no coords needed.
        """
        Lx, Ly = self.lattice.Lx, self.lattice.Ly
        spins = self.lattice.magnetic_moments

        mask = np.zeros(spins.size, dtype=bool)
        mask[cluster_idx] = True

        spins2d = spins.reshape(Ly, Lx)
        mask2d  = mask.reshape(Ly, Lx)

        plt.figure(figsize=(6, 6))
        plt.imshow(spins2d, cmap=SPIN_CMAP, vmin=-1, vmax=1,
                interpolation="nearest", aspect="equal")
        plt.axis("off")

        # optional soft glow then crisp outline
        plt.contour(mask2d.astype(float), levels=[0.5],
                    linewidths=4.0, colors="black", alpha=0.25)
        plt.contour(mask2d.astype(float), levels=[0.5],
                    linewidths=2.0, colors="#a99f17")

        plt.title(f"Wolff step {step} — cluster size {mask.sum()}")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"step_{step:04d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
