# visualization.py — coords-based plotting for disordered lattice
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio


PALETTE = [
    "#C87568",  # терракота
    "#D9A79A",  # светлый персиковый
    "#D5963C",  # охра/горчичный
    "#2F365A",  # темный индиго/нави
    "#7B6886",  # приглушенный сиренево-серый
]

# PALETTE_ALT = ["#C56E61", "#DEB0A7", "#D39233", "#293155", "#746180"]
# MY_CMAP_ALT = ListedColormap(PALETTE_ALT, name="custom5b")



MESH_COLOR    = PALETTE[2]
BOND_FM_COLOR = PALETTE[1]
BOND_AF_COLOR = PALETTE[4]

SPIN_CMAP = ListedColormap([PALETTE[0], PALETTE[3]], name="spin2")



class Visualization:
    def __init__(self, lattice, accumulator,
                 outdir="frames", gif_file="lattice.gif", fps=6):
        self.lattice  = lattice
        self.acc      = accumulator
        self.outdir   = outdir
        self.gif_file = gif_file if gif_file.endswith(".gif") else gif_file + ".gif"
        self.fps      = fps
        os.makedirs(self.outdir, exist_ok=True)

    # ---------- BASIC SNAPSHOTS ----------
    def plot_coords(self, show_mesh=True, s=28):
        """Scatter spins on triangulation."""
        coords = self.lattice.coords
        spins  = self.lattice.magnetic_moments
        plt.figure(figsize=(6, 6))
        if show_mesh:
            triang = tri.Triangulation(coords[:, 0], coords[:, 1])
            plt.triplot(triang, color=MESH_COLOR, linewidth=0.5, zorder=0)
        plt.scatter(coords[:, 0], coords[:, 1],
                    c=spins, cmap=SPIN_CMAP, vmin=-1, vmax=1,
                    s=s, linewidths=0, zorder=1)
        plt.axis("equal"); plt.axis("off")
        plt.tight_layout();# plt.show()

    def plot_coords_save(self, step, show_mesh=True, s=70, output_dir=None):
        """Save a plain snapshot for the disordered lattice."""
        out = output_dir or self.outdir
        coords = self.lattice.coords
        spins  = self.lattice.magnetic_moments
        plt.figure(figsize=(6, 6))
        if show_mesh:
            triang = tri.Triangulation(coords[:, 0], coords[:, 1])
            plt.triplot(triang, color=MESH_COLOR, linewidth=0.5, zorder=0)
        plt.scatter(coords[:, 0], coords[:, 1],
                    c=spins, cmap=SPIN_CMAP, vmin=-1, vmax=1,
                    s=s, linewidths=0, zorder=1)
        plt.axis("equal"); plt.axis("off")
        plt.title(f"Step {step}")
        os.makedirs(out, exist_ok=True)
        plt.savefig(os.path.join(out, f"step_{step:04d}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def create_gif_from_frames(self, output_dir=None, output_file=None, fps=None):
        outdir = output_dir or self.outdir
        outfile = output_file or self.gif_file
        _fps = fps or self.fps
        files = sorted(glob.glob(os.path.join(outdir, "step_*.png")))
        images = [imageio.imread(f) for f in files]
        imageio.mimsave(outfile, images, fps=_fps)
        print(f"GIF saved to {outfile}")

    # ---------- CLUSTER BORDER (irregular mesh, PBC-safe) ----------
    def _cluster_boundary_segments(self, cluster_mask):
        """
        Build short PBC-aware line segments along boundaries where cluster vs non-cluster
        meet, using the triangulation graph.
        """
        coords = self.lattice.coords
        Lx, Ly = self.lattice.Lx, self.lattice.Ly
        triang = tri.Triangulation(coords[:, 0], coords[:, 1])

        def short_segment(i, j):
            xi, yi = coords[i]
            xj, yj = coords[j]
            dx = xj - xi; dy = yj - yi
            # minimum-image convention
            if dx >  Lx/2:  xj -= Lx
            if dx < -Lx/2:  xj += Lx
            if dy >  Ly/2:  yj -= Ly
            if dy < -Ly/2:  yj += Ly
            return [(xi, yi), (xj, yj)]

        segments, seen = [], set()
        for tri_ in triang.triangles:
            for a, b in ((0,1),(1,2),(2,0)):
                i, j = tri_[a], tri_[b]
                if cluster_mask[i] ^ cluster_mask[j]:
                    key = (i, j) if i < j else (j, i)
                    if key in seen: continue
                    seen.add(key)
                    segments.append(short_segment(i, j))
        return segments

    def plot_lattice_with_cluster(self, cluster_idx, step=None, s=28):
        """Scatter spins and overlay the cluster boundary (yellow), PBC-safe."""
        spins  = self.lattice.magnetic_moments
        coords = self.lattice.coords
        cmask  = np.zeros(len(spins), dtype=bool)
        cmask[np.asarray(list(cluster_idx), dtype=int)] = True

        plt.figure(figsize=(6, 6))
        triang = tri.Triangulation(coords[:, 0], coords[:, 1])
        plt.triplot(triang, color=MESH_COLOR, linewidth=0.5, zorder=0)
        plt.scatter(coords[:, 0], coords[:, 1],
                    c=spins, cmap=SPIN_CMAP, vmin=-1, vmax=1,
                    s=s, linewidths=0, zorder=1)

        segs = self._cluster_boundary_segments(cmask)
        if segs:
            lc = LineCollection(segs, linewidths=2.0, colors=BORDER_COLOR, zorder=2)
            plt.gca().add_collection(lc)

        if step is not None:
            plt.title(f"Wolff step {step} — cluster size {cmask.sum()}")
        plt.axis("equal"); plt.axis("off"); plt.tight_layout()#; plt.show()

    def plot_lattice_with_cluster_save(self, cluster_idx, step, s=28, output_dir=None):
        self.plot_lattice_with_cluster(cluster_idx, step, s=s)
        out = output_dir or self.outdir
        os.makedirs(out, exist_ok=True)
        plt.savefig(os.path.join(out, f"step_{step:04d}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ---------- OPTIONAL: BOND MAP (top-|J| or r cutoff) ----------
    def plot_bonds(self, k=500, r_max=None, lw=1.5, alpha=0.5):
        """
        Draw strongest bonds: either top-k by |J| (upper triangle) or all with r < r_max.
        Green = ferro (J>0), Red = antiferro (J<0). PBC-short segments.
        """
        coords = self.lattice.coords
        Lx, Ly = self.lattice.Lx, self.lattice.Ly
        J = self.lattice.interaction_matrix
        i_idx, j_idx = np.triu_indices(self.lattice.N, k=1)

        if r_max is not None:
            mask = (self.lattice.distances[i_idx, j_idx] <= r_max)
        else:
            flat = np.abs(J[i_idx, j_idx])
            if k is None or k >= flat.size: mask = np.ones_like(flat, dtype=bool)
            else:
                thresh = np.partition(flat, -k)[-k]
                mask = flat >= thresh

        ii = i_idx[mask]; jj = j_idx[mask]
        segs_fm, segs_af = [], []

        def short_seg(i, j):
            xi, yi = coords[i]; xj, yj = coords[j]
            dx = xj - xi; dy = yj - yi
            if dx >  Lx/2:  xj -= Lx
            if dx < -Lx/2:  xj += Lx
            if dy >  Ly/2:  yj -= Ly
            if dy < -Ly/2:  yj += Ly
            return [(xi, yi), (xj, yj)]

        for a, b in zip(ii, jj):
            seg = short_seg(a, b)
            if J[a, b] >= 0:
                segs_fm.append(seg)
            else:
                segs_af.append(seg)

        plt.figure(figsize=(6, 6))
        plt.scatter(coords[:,0], coords[:,1], c=self.lattice.magnetic_moments,
                    cmap=SPIN_CMAP, vmin=-1, vmax=1, s=18, linewidths=0, zorder=1)
        if segs_fm:
            plt.gca().add_collection(LineCollection(segs_fm, colors=BOND_FM_COLOR, lw=lw, alpha=alpha, zorder=0))
        if segs_af:
            plt.gca().add_collection(LineCollection(segs_af, colors=BOND_AF_COLOR, lw=lw, alpha=alpha, zorder=0))
        plt.axis("equal"); plt.axis("off"); plt.tight_layout()#; plt.show()

    # ---------- STRUCTURE FACTOR S(q) ----------
    def compute_structure_factor(self, qn=64):
        """
        Return (qx, qy, S) on a q-grid; qx, qy are 1D arrays (radians).
        """
        x = self.lattice.coords[:, 0]; y = self.lattice.coords[:, 1]
        s = self.lattice.magnetic_moments.astype(float)
        Lx, Ly = self.lattice.Lx, self.lattice.Ly

        qx = 2*np.pi * np.linspace(-qn//2, qn//2-1, qn) / Lx
        qy = 2*np.pi * np.linspace(-qn//2, qn//2-1, qn) / Ly
        Qx, Qy = np.meshgrid(qx, qy, indexing='ij')

        # broadcast: sum_i s_i exp(-i(qx x_i + qy y_i))
        phase = x[:, None, None] * Qx[None, :, :] + y[:, None, None] * Qy[None, :, :]
        Mq    = (s[:, None, None] * np.exp(-1j * phase)).sum(axis=0)
        S     = (np.abs(Mq) ** 2) / s.size
        return qx, qy, S

    def plot_structure_factor(self, qn=64):
        qx, qy, S = self.compute_structure_factor(qn=qn)
        extent = [qx.min(), qx.max(), qy.min(), qy.max()]
        plt.figure(figsize=(6, 5))
        plt.imshow(np.fft.fftshift(S.T), origin="lower", extent=extent, aspect="equal")
        plt.colorbar(label="S(q)")
        plt.xlabel("qx"); plt.ylabel("qy"); plt.title("Structure factor S(q)")
        plt.tight_layout()#; plt.show()

    # ---------- BINNED PAIR CORRELATION ----------
    def plot_pair_correlation_binned(self, nbins="fd"):
        """
        Radial ⟨S_i S_j⟩(r) using existing distances and pair products.
        """
        sijs = self.lattice.compute_pair_correlation()      # shape: M= N(N-1)/2
        r    = self.lattice.r_ij
        if nbins == "fd":
            bins = self.lattice.bin_edges
        else:
            bins = np.linspace(0, min(self.lattice.Lx, self.lattice.Ly)/2, nbins+1)

        idx = np.digitize(r, bins) - 1
        nb  = bins.size - 1
        corr = np.zeros(nb); count = np.zeros(nb)
        for k in range(nb):
            mask = (idx == k)
            if np.any(mask):
                corr[k]  = sijs[mask].mean()
                count[k] = mask.sum()
        centers = 0.5 * (bins[:-1] + bins[1:])

        plt.figure(figsize=(6,4))
        plt.plot(centers, corr, "-o", ms=3)
        plt.axhline(0, color="gray", lw=1, ls="--")
        plt.xlabel("r"); plt.ylabel("⟨S_i S_j⟩(r)")
        plt.title("Radial pair correlation")
        plt.grid(True, alpha=0.3); plt.tight_layout()#; plt.show()


    def _save_frame(self, step, cluster_idx=None, outdir="frames"):
        if cluster_idx is not None:
            if hasattr(self.lattice, "coords"):   # irregular
                self.plot_lattice_with_cluster_save(cluster_idx, step, output_dir=outdir)
            else:                                  # regular grid
                self.plot_grid_with_cluster_save(cluster_idx, step, output_dir=outdir)
        else:
            if hasattr(self.lattice, "coords"):
                self.plot_coords_save(step, output_dir=outdir)
            else:
                self.plot_lattice_save(self.lattice.magnetic_moments,
                                        self.lattice.Lx, self.lattice.Ly,
                                        step, output_dir=outdir)


    def plot_cluster_save(self, cluster_idx, edges_fm, edges_afm, step, N, output_dir=None,
                        node_s=70, color_fm=BOND_FM_COLOR, color_afm=BOND_AF_COLOR, edge_lw=1.):
        import os
        from matplotlib.collections import LineCollection
        import matplotlib.tri as tri
        out = output_dir or self.outdir
        os.makedirs(out, exist_ok=True)

        coords = self.lattice.coords
        spins  = self.lattice.magnetic_moments
        Lx, Ly = self.lattice.Lx, self.lattice.Ly

        def short_seg(i, j):
            xi, yi = coords[i]; xj, yj = coords[j]
            dx = xj - xi; dy = yj - yi
            if dx >  Lx/2: xj -= Lx
            if dx < -Lx/2: xj += Lx
            if dy >  Ly/2: yj -= Ly
            if dy < -Ly/2: yj += Ly
            return [(xi, yi), (xj, yj)]
        segs_fm =  [short_seg(i, j) for (i, j) in edges_fm]
        segs_afm = [short_seg(i, j) for (i, j) in edges_afm]

        fig, ax = plt.subplots(figsize=(6,6))
        triang = tri.Triangulation(coords[:,0], coords[:,1])
        ax.triplot(triang, color="#c6c7c9", linewidth=0.5, zorder=0)
        ax.scatter(coords[:,0], coords[:,1], c=spins, cmap=SPIN_CMAP, vmin=-1, vmax=1,
                s=node_s, linewidths=0, zorder=5)                                  # atoms
                      # grid
        if segs_fm:
            ax.add_collection(LineCollection(segs_fm, colors=color_fm,
                                            linewidths=edge_lw, zorder=2))        
        if segs_afm:
            ax.add_collection(LineCollection(segs_afm, colors=color_afm,
                                            linewidths=edge_lw, zorder=2))        
        # ax.scatter(coords[:,0], coords[:,1], c=spins, cmap=SPIN_CMAP, vmin=-1, vmax=1,
        #         s=node_s, linewidths=0, zorder=1)                                  # atoms
     
        ax.set_axis_off(); ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Wolff step {step} — cluster size {len(cluster_idx)}/{N}")
        fig.savefig(os.path.join(out, f"step_{step:04d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
