"""
Comprehensive visualization suite for Monte Carlo simulation.

Generates:
1. Lattice visualizations for different dopings
2. Phase diagrams: |M| vs T (varying kF and doping)
3. Phase diagrams: χ vs T (varying kF and doping)
4. GIFs of system evolution during warmup (Metropolis & Wolff)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap
import imageio
import glob
from tqdm import tqdm

from lattice import Lattice
from monte_carlo import MonteCarlo
from visualization import Visualization

plt.style.use('seaborn-v0_8-darkgrid')

# Color palette
PALETTE = ["#C87568", "#D9A79A", "#D5963C", "#2F365A", "#7B6886"]
SPIN_CMAP = ListedColormap([PALETTE[0], PALETTE[3]], name="spin2")


# ============================================================================
# PART 1: LATTICE VISUALIZATIONS FOR DIFFERENT DOPINGS
# ============================================================================

def visualize_lattices_by_doping(dopings=[0.005, 0.01, 0.05], rows=20, cols=20, 
                                 outdir="plots", dpi=150):
    """
    Create side-by-side lattice visualizations for different doping levels.
    
    Args:
        dopings: List of doping fractions to visualize
        rows, cols: Lattice size
        outdir: Output directory
        dpi: Figure resolution
    """
    os.makedirs(outdir, exist_ok=True)
    
    n_dopings = len(dopings)
    fig, axes = plt.subplots(1, n_dopings, figsize=(5*n_dopings, 5))
    
    if n_dopings == 1:
        axes = [axes]
    
    for ax, doping in zip(axes, dopings):
        # Create lattice
        np.random.seed(42)
        lat = Lattice(rows=rows, cols=cols, doping=doping, kf=1.0, J0=1.0)
        
        coords = lat.lattice_points
        spins = lat.magnetic_moments
        
        # Create triangulation
        triang = tri.Triangulation(coords[:, 0], coords[:, 1])
        
        # Plot
        ax.triplot(triang, color=PALETTE[2], linewidth=0.5, zorder=0)
        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                            c=spins, cmap=SPIN_CMAP, vmin=-1, vmax=1,
                            s=50, linewidths=0, zorder=1)
        
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(f'Doping = {doping*100:.1f}% (N={lat.N})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "01_lattices_dopings.png"), dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {outdir}/01_lattices_dopings.png")
    plt.close()


# ============================================================================
# PART 2: PHASE DIAGRAMS - |M| vs T with varying kF
# ============================================================================

def phase_diagram_M_vs_T_varying_kf(temperatures=None, kf_values=[0.5, 1.0, 1.5, 2.0],
                                     rows=15, cols=15, doping=0.5,
                                     warmup_steps=200, prod_steps=500,
                                     outdir="plots", dpi=150):
    """
    Create phase diagram: |M| vs T for different kF values.
    
    Args:
        temperatures: List of T values to sample
        kf_values: Fermi wavevector values
        rows, cols: Lattice size
        doping: Doping concentration
        warmup_steps: Equilibration steps
        prod_steps: Production steps
        outdir: Output directory
        dpi: Figure resolution
    """
    if temperatures is None:
        temperatures = np.linspace(0.2, 3.0, 12)
    
    os.makedirs(outdir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for kf in kf_values:
        M_abs_list = []
        
        print(f"\nMeasuring |M| vs T for kF={kf}...")
        for T in tqdm(temperatures):
            np.random.seed(42)
            lat = Lattice(rows=rows, cols=cols, doping=doping, kf=kf, J0=1.0)
            mc = MonteCarlo(lat, progress=False)
            
            # Run simulation
            mc.run_loop(warmup_steps=warmup_steps, steps=prod_steps, T=T, 
                       method="metropolis", save_warmup=False)
            
            # Extract |M| average
            M_abs = np.mean(np.abs(mc.acc.magnetization))
            M_abs_list.append(M_abs)
        
        ax.plot(temperatures, M_abs_list, 'o-', label=f'$k_F$ = {kf}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Temperature T', fontsize=12, fontweight='bold')
    ax.set_ylabel('|Magnetization| |M|', fontsize=12, fontweight='bold')
    ax.set_title(f'Phase Diagram: |M| vs T (Varying $k_F$)\nDoping = {doping*100:.1f}%', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02_phase_diagram_M_vs_T_kf.png"), dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {outdir}/02_phase_diagram_M_vs_T_kf.png")
    plt.close()


# ============================================================================
# PART 3: PHASE DIAGRAMS - |M| vs T with varying doping
# ============================================================================

def phase_diagram_M_vs_T_varying_doping(temperatures=None, dopings=[0.3, 0.5, 0.7, 0.9],
                                       rows=15, cols=15, kf=1.0,
                                       warmup_steps=200, prod_steps=500,
                                       outdir="plots", dpi=150):
    """
    Create phase diagram: |M| vs T for different doping levels.
    """
    if temperatures is None:
        temperatures = np.linspace(0.2, 3.0, 12)
    
    os.makedirs(outdir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for doping in dopings:
        M_abs_list = []
        
        print(f"\nMeasuring |M| vs T for doping={doping}...")
        for T in tqdm(temperatures):
            np.random.seed(42)
            lat = Lattice(rows=rows, cols=cols, doping=doping, kf=kf, J0=1.0)
            mc = MonteCarlo(lat, progress=False)
            
            # Run simulation
            mc.run_loop(warmup_steps=warmup_steps, steps=prod_steps, T=T,
                       method="metropolis", save_warmup=False)
            
            # Extract |M| average
            M_abs = np.mean(np.abs(mc.acc.magnetization))
            M_abs_list.append(M_abs)
        
        ax.plot(temperatures, M_abs_list, 'o-', label=f'δ = {doping*100:.0f}%', 
               linewidth=2, markersize=6)
    
    ax.set_xlabel('Temperature T', fontsize=12, fontweight='bold')
    ax.set_ylabel('|Magnetization| |M|', fontsize=12, fontweight='bold')
    ax.set_title(f'Phase Diagram: |M| vs T (Varying Doping)\n$k_F$ = {kf}', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03_phase_diagram_M_vs_T_doping.png"), dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {outdir}/03_phase_diagram_M_vs_T_doping.png")
    plt.close()


# ============================================================================
# PART 4: SUSCEPTIBILITY vs T with varying kF
# ============================================================================

def phase_diagram_chi_vs_T_varying_kf(temperatures=None, kf_values=[0.5, 1.0, 1.5, 2.0],
                                      rows=15, cols=15, doping=0.5,
                                      warmup_steps=200, prod_steps=500,
                                      outdir="plots", dpi=150):
    """
    Create phase diagram: χ vs T for different kF values.
    """
    if temperatures is None:
        temperatures = np.linspace(0.2, 3.0, 12)
    
    os.makedirs(outdir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for kf in kf_values:
        chi_list = []
        
        print(f"\nMeasuring χ vs T for kF={kf}...")
        for T in tqdm(temperatures):
            np.random.seed(42)
            lat = Lattice(rows=rows, cols=cols, doping=doping, kf=kf, J0=1.0)
            mc = MonteCarlo(lat, progress=False)
            
            # Run simulation
            mc.run_loop(warmup_steps=warmup_steps, steps=prod_steps, T=T,
                       method="metropolis", save_warmup=False)
            
            # Extract χ average
            chi = np.mean(mc.acc.susceptibility)
            chi_list.append(chi)
        
        ax.plot(temperatures, chi_list, 's-', label=f'$k_F$ = {kf}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Temperature T', fontsize=12, fontweight='bold')
    ax.set_ylabel('Magnetic Susceptibility χ', fontsize=12, fontweight='bold')
    ax.set_title(f'Phase Diagram: χ vs T (Varying $k_F$)\nDoping = {doping*100:.1f}%', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "04_phase_diagram_chi_vs_T_kf.png"), dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {outdir}/04_phase_diagram_chi_vs_T_kf.png")
    plt.close()


# ============================================================================
# PART 5: SUSCEPTIBILITY vs T with varying doping
# ============================================================================

def phase_diagram_chi_vs_T_varying_doping(temperatures=None, dopings=[0.3, 0.5, 0.7, 0.9],
                                         rows=15, cols=15, kf=1.0,
                                         warmup_steps=200, prod_steps=500,
                                         outdir="plots", dpi=150):
    """
    Create phase diagram: χ vs T for different doping levels.
    """
    if temperatures is None:
        temperatures = np.linspace(0.2, 3.0, 12)
    
    os.makedirs(outdir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for doping in dopings:
        chi_list = []
        
        print(f"\nMeasuring χ vs T for doping={doping}...")
        for T in tqdm(temperatures):
            np.random.seed(42)
            lat = Lattice(rows=rows, cols=cols, doping=doping, kf=kf, J0=1.0)
            mc = MonteCarlo(lat, progress=False)
            
            # Run simulation
            mc.run_loop(warmup_steps=warmup_steps, steps=prod_steps, T=T,
                       method="metropolis", save_warmup=False)
            
            # Extract χ average
            chi = np.mean(mc.acc.susceptibility)
            chi_list.append(chi)
        
        ax.plot(temperatures, chi_list, 's-', label=f'δ = {doping*100:.0f}%', 
               linewidth=2, markersize=6)
    
    ax.set_xlabel('Temperature T', fontsize=12, fontweight='bold')
    ax.set_ylabel('Magnetic Susceptibility χ', fontsize=12, fontweight='bold')
    ax.set_title(f'Phase Diagram: χ vs T (Varying Doping)\n$k_F$ = {kf}', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "05_phase_diagram_chi_vs_T_doping.png"), dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {outdir}/05_phase_diagram_chi_vs_T_doping.png")
    plt.close()


# ============================================================================
# PART 6: GIF GENERATION - METROPOLIS WARMUP
# ============================================================================

def generate_gif_metropolis_warmup(rows=15, cols=15, doping=0.5, T=1.0, kf=1.0,
                                  warmup_steps=500, save_interval=10,
                                  outdir="frames", gif_file="metropolis_warmup.gif", dpi=150):
    """
    Generate GIF showing system evolution during Metropolis warmup.
    
    Args:
        rows, cols: Lattice size
        doping: Doping concentration
        T: Temperature
        kf: Fermi wavevector
        warmup_steps: Total warmup steps
        save_interval: Save every Nth step
        outdir: Directory for frames
        gif_file: Output GIF filename
        dpi: Image resolution
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Create lattice and Monte Carlo
    np.random.seed(42)
    lat = Lattice(rows=rows, cols=cols, doping=doping, kf=kf, J0=1.0)
    mc = MonteCarlo(lat, progress=False)
    mc.T = T
    
    coords = lat.lattice_points
    Lx, Ly = lat.Lx, lat.Ly
    
    print(f"\nGenerating Metropolis warmup GIF (T={T}, doping={doping*100:.1f}%)...")
    
    # Clean old frames
    for f in glob.glob(os.path.join(outdir, "metro_*.png")):
        os.remove(f)
    
    frame_count = 0
    for step in tqdm(range(warmup_steps)):
        mc.metropolis_step()
        
        if step % save_interval == 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Plot
            triang = tri.Triangulation(coords[:, 0], coords[:, 1])
            ax.triplot(triang, color=PALETTE[2], linewidth=0.5, zorder=0)
            ax.scatter(coords[:, 0], coords[:, 1],
                      c=lat.magnetic_moments, cmap=SPIN_CMAP, vmin=-1, vmax=1,
                      s=50, linewidths=0, zorder=1)
            
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.set_title(f'Metropolis Warmup: Step {step:04d} | T={T:.2f} | M={mc.M:.3f} | E={mc.E:.3f}',
                        fontsize=11, fontweight='bold')
            
            # Save frame
            frame_path = os.path.join(outdir, f"metro_{frame_count:04d}.png")
            plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            frame_count += 1
    
    # Create GIF with consistent image format
    frames = sorted(glob.glob(os.path.join(outdir, "metro_*.png")))
    images = []
    
    for f in frames:
        from PIL import Image
        # Read with PIL to ensure consistent format
        img = Image.open(f).convert('RGB')
        images.append(np.array(img))
    
    # Ensure all images have the same shape
    if images:
        shapes = [img.shape for img in images]
        if len(set(shapes)) > 1:
            # Resize all to match first image
            target_shape = images[0].shape
            images = [np.array(Image.fromarray(img).resize((target_shape[1], target_shape[0]))) if img.shape != target_shape else img for img in images]
        
        imageio.mimsave(os.path.join(outdir, gif_file), images, fps=10)
    
    print(f"✓ Saved: {outdir}/{gif_file}")
    
    # Clean up frames
    for f in frames:
        os.remove(f)


# ============================================================================
# PART 7: GIF GENERATION - WOLFF WARMUP
# ============================================================================

def generate_gif_wolff_warmup(rows=15, cols=15, doping=0.5, T=1.0, kf=1.0,
                             warmup_steps=200, save_interval=5,
                             outdir="frames", gif_file="wolff_warmup.gif", dpi=150):
    """
    Generate GIF showing system evolution during Wolff warmup.
    
    Args:
        rows, cols: Lattice size
        doping: Doping concentration
        T: Temperature
        kf: Fermi wavevector
        warmup_steps: Total warmup steps
        save_interval: Save every Nth step
        outdir: Directory for frames
        gif_file: Output GIF filename
        dpi: Image resolution
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Create lattice and Monte Carlo
    np.random.seed(42)
    lat = Lattice(rows=rows, cols=cols, doping=doping, kf=kf, J0=1.0)
    mc = MonteCarlo(lat, progress=False)
    mc.T = T
    mc.precompute_bond_probabilities()
    
    coords = lat.lattice_points
    Lx, Ly = lat.Lx, lat.Ly
    
    print(f"\nGenerating Wolff warmup GIF (T={T}, doping={doping*100:.1f}%)...")
    
    # Clean old frames
    for f in glob.glob(os.path.join(outdir, "wolff_*.png")):
        os.remove(f)
    
    frame_count = 0
    for step in tqdm(range(warmup_steps)):
        cluster, edges_fm, edges_afm = mc.wolff_step(return_cluster=True)
        
        if step % save_interval == 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Plot lattice
            triang = tri.Triangulation(coords[:, 0], coords[:, 1])
            ax.triplot(triang, color=PALETTE[2], linewidth=0.5, zorder=0)
            ax.scatter(coords[:, 0], coords[:, 1],
                      c=lat.magnetic_moments, cmap=SPIN_CMAP, vmin=-1, vmax=1,
                      s=50, linewidths=0, zorder=1)
            
            # Highlight cluster if exists
            if len(cluster) > 0:
                cluster_coords = coords[cluster]
                ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                          facecolors='none', edgecolors='yellow', linewidths=2, s=100, zorder=2)
            
            ax.set_aspect('equal')
            ax.set_axis_off()
            ax.set_title(f'Wolff Warmup: Step {step:04d} | T={T:.2f} | Cluster Size {len(cluster):3d} | M={mc.M:.3f} | E={mc.E:.3f}',
                        fontsize=11, fontweight='bold')
            
            # Save frame
            frame_path = os.path.join(outdir, f"wolff_{frame_count:04d}.png")
            plt.savefig(frame_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            frame_count += 1
    
    # Create GIF with consistent image format
    frames = sorted(glob.glob(os.path.join(outdir, "wolff_*.png")))
    images = []
    
    for f in frames:
        from PIL import Image
        # Read with PIL to ensure consistent format
        img = Image.open(f).convert('RGB')
        images.append(np.array(img))
    
    # Ensure all images have the same shape
    if images:
        shapes = [img.shape for img in images]
        if len(set(shapes)) > 1:
            # Resize all to match first image
            target_shape = images[0].shape
            images = [np.array(Image.fromarray(img).resize((target_shape[1], target_shape[0]))) if img.shape != target_shape else img for img in images]
        
        imageio.mimsave(os.path.join(outdir, gif_file), images, fps=10)
    
    print(f"✓ Saved: {outdir}/{gif_file}")
    
    # Clean up frames
    for f in frames:
        os.remove(f)


# ============================================================================
# MAIN: RUN ALL VISUALIZATIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE VISUALIZATION SUITE")
    print("=" * 70)
    
    # Configuration
    PLOTS_DIR = "plots"
    FRAMES_DIR = "frames"
    
    # Part 1: Lattice visualizations for different dopings
    print("\n[1/7] Visualizing lattices for different dopings...")
    visualize_lattices_by_doping(dopings=[0.005, 0.01, 0.05], rows=20, cols=20, outdir=PLOTS_DIR)
    
    # Part 2: |M| vs T with varying kF
    print("\n[2/7] Phase diagram: |M| vs T (varying kF)...")
    phase_diagram_M_vs_T_varying_kf(
        temperatures=np.linspace(0.2, 3.0, 10),
        kf_values=[0.5, 1.0, 1.5, 2.0],
        rows=12, cols=12, doping=0.5,
        warmup_steps=150, prod_steps=300,
        outdir=PLOTS_DIR
    )
    
    # Part 3: |M| vs T with varying doping
    print("\n[3/7] Phase diagram: |M| vs T (varying doping)...")
    phase_diagram_M_vs_T_varying_doping(
        temperatures=np.linspace(0.2, 3.0, 10),
        dopings=[0.3, 0.5, 0.7, 0.9],
        rows=12, cols=12, kf=1.0,
        warmup_steps=150, prod_steps=300,
        outdir=PLOTS_DIR
    )
    
    # Part 4: χ vs T with varying kF
    print("\n[4/7] Phase diagram: χ vs T (varying kF)...")
    phase_diagram_chi_vs_T_varying_kf(
        temperatures=np.linspace(0.2, 3.0, 10),
        kf_values=[0.5, 1.0, 1.5, 2.0],
        rows=12, cols=12, doping=0.5,
        warmup_steps=150, prod_steps=300,
        outdir=PLOTS_DIR
    )
    
    # Part 5: χ vs T with varying doping
    print("\n[5/7] Phase diagram: χ vs T (varying doping)...")
    phase_diagram_chi_vs_T_varying_doping(
        temperatures=np.linspace(0.2, 3.0, 10),
        dopings=[0.3, 0.5, 0.7, 0.9],
        rows=12, cols=12, kf=1.0,
        warmup_steps=150, prod_steps=300,
        outdir=PLOTS_DIR
    )
    
    # Part 6: Metropolis warmup GIF
    print("\n[6/7] Generating Metropolis warmup GIF...")
    generate_gif_metropolis_warmup(
        rows=12, cols=12, doping=0.5, T=1.0, kf=1.0,
        warmup_steps=300, save_interval=10,
        outdir=FRAMES_DIR, gif_file="metropolis_warmup.gif"
    )
    
    # Part 7: Wolff warmup GIF
    print("\n[7/7] Generating Wolff warmup GIF...")
    generate_gif_wolff_warmup(
        rows=12, cols=12, doping=0.5, T=1.0, kf=1.0,
        warmup_steps=100, save_interval=5,
        outdir=FRAMES_DIR, gif_file="wolff_warmup.gif"
    )
    
    print("\n" + "=" * 70)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to:")
    print(f"  - Phase diagrams and lattice plots: {PLOTS_DIR}/")
    print(f"  - GIFs: {FRAMES_DIR}/")
    print("\nGenerated files:")
    print("  1. 01_lattices_dopings.png - Lattice visualizations")
    print("  2. 02_phase_diagram_M_vs_T_kf.png - |M| vs T (varying kF)")
    print("  3. 03_phase_diagram_M_vs_T_doping.png - |M| vs T (varying doping)")
    print("  4. 04_phase_diagram_chi_vs_T_kf.png - χ vs T (varying kF)")
    print("  5. 05_phase_diagram_chi_vs_T_doping.png - χ vs T (varying doping)")
    print("  6. metropolis_warmup.gif - System evolution (Metropolis)")
    print("  7. wolff_warmup.gif - System evolution (Wolff)")
