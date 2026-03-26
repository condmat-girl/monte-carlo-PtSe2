"""
Visualize RKKY potential V_RKKY(r*k_F) for different Fermi wavevector values.

The RKKY interaction is:
    J(r) = -J₀ [J₀(k_F*r)*Y₀(k_F*r) + J₁(k_F*r)*Y₁(k_F*r)]

Plotting as a function of the dimensionless parameter (r*k_F) shows the
universal oscillation pattern and how k_F affects the wavelength and
amplitude of the interaction.

Physical interpretation:
- The oscillating RKKY interaction is the basis for magnetic ordering
- The Fermi wavevector k_F depends on the carrier density: k_F ~ sqrt(n)
- Different k_F values represent different doping levels or materials
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1, y0, y1


def rkky_interaction(r, kf, J0=1.0):
    """
    RKKY interaction strength at distance r.
    
    Args:
        r: Pairwise distance(s)
        kf: Fermi wavevector
        J0: Overall interaction strength (default: 1.0)
        
    Returns:
        Interaction strength J(r)
    """
    x = kf * r
    # Avoid r=0 singularity
    x = np.where(x == 0, 1e-10, x)
    interaction = -J0 * (j0(x) * y0(x) + j1(x) * y1(x))
    return interaction


def plot_rkky_potential():
    """Plot RKKY potential for multiple k_F values."""
    
    # Extended range in real space
    r_max = 50
    r = np.linspace(0.1, r_max, 2000)
    
    # Multiple k_F values spanning typical experimental ranges
    # For a 2D system with carrier density n:
    #   k_F = sqrt(2*pi*n) in SI units
    # Typical ranges: k_F ~ 0.1 - 2.0 Angstrom^-1
    kf_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RKKY Potential $V_{RKKY}(r k_F)$', fontsize=16, fontweight='bold')
    
    # ========== Panel 1: V vs r (rescaled by k_F) ==========
    ax = axes[0, 0]
    for kf in kf_values:
        potential = rkky_interaction(r, kf)
        # Plot as function of dimensionless parameter r*k_F
        x_scaled = r * kf
        ax.plot(x_scaled, potential, linewidth=2, label=f'$k_F = {kf}$ Ų⁻¹', 
                alpha=0.8)
    
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('$r k_F$ (dimensionless)', fontsize=12)
    ax.set_ylabel('$V_{RKKY}(r k_F)$ (units of $J_0$)', fontsize=12)
    ax.set_title('Universal RKKY Oscillation Pattern', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 20])
    
    # ========== Panel 2: V vs r for each k_F individually ==========
    ax = axes[0, 1]
    for kf in kf_values:
        potential = rkky_interaction(r, kf)
        ax.plot(r, potential, linewidth=2, label=f'$k_F = {kf}$ Ų⁻¹', alpha=0.8)
    
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Distance $r$ (lattice units)', fontsize=12)
    ax.set_ylabel('$V_{RKKY}(r)$ (units of $J_0$)', fontsize=12)
    ax.set_title('Real-Space RKKY Potential', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 20])
    
    # ========== Panel 3: Oscillation wavelength analysis ==========
    ax = axes[1, 0]
    # The RKKY oscillation has period ~ 2π / (2*k_F) in real space
    wavelengths = []
    for kf in kf_values:
        lambda_rkky = 2 * np.pi / (2 * kf)  # Simplified estimate
        wavelengths.append(lambda_rkky)
    
    ax.bar(range(len(kf_values)), wavelengths, color='steelblue', alpha=0.7, 
           edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(kf_values)))
    ax.set_xticklabels([f'{kf}' for kf in kf_values])
    ax.set_xlabel('Fermi Wavevector $k_F$ (Ų⁻¹)', fontsize=12)
    ax.set_ylabel('RKKY Wavelength $\lambda$ (lattice units)', fontsize=12)
    ax.set_title('Oscillation Wavelength vs $k_F$', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========== Panel 4: First minima and maxima locations ==========
    ax = axes[1, 1]
    first_periods = []
    first_minima = []
    first_maxima = []
    
    for kf in kf_values:
        # Find zero crossings and extrema
        x_scaled = r * kf
        potential = rkky_interaction(r, kf)
        
        # Simple approach: find first minimum and maximum
        # For RKKY, first maximum is around π/k_F, first minimum around 2π/k_F
        first_max_approx = np.pi / kf
        first_min_approx = 2 * np.pi / kf
        
        first_minima.append(first_min_approx)
        first_maxima.append(first_max_approx)
        first_periods.append(first_min_approx)
    
    x_pos = np.arange(len(kf_values))
    width = 0.35
    
    ax.bar(x_pos - width/2, first_maxima, width, label='First Maximum', 
           color='coral', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x_pos + width/2, first_minima, width, label='First Minimum', 
           color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{kf}' for kf in kf_values])
    ax.set_xlabel('Fermi Wavevector $k_F$ (Ų⁻¹)', fontsize=12)
    ax.set_ylabel('Distance $r$ (lattice units)', fontsize=12)
    ax.set_title('Location of Extrema', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def print_rkky_properties(kf_values):
    """Print physical properties of RKKY potential for different k_F."""
    print("\n" + "="*70)
    print("RKKY POTENTIAL PROPERTIES")
    print("="*70)
    
    for kf in kf_values:
        print(f"\nk_F = {kf} Ų⁻¹:")
        print(f"  Oscillation wavelength λ ≈ π/k_F = {np.pi/kf:.3f} lattice units")
        print(f"  First extrema at r ≈ {np.pi/(2*kf):.3f} (max) and {np.pi/kf:.3f} (min)")
        print(f"  Characteristic range ≈ 2π/k_F = {2*np.pi/kf:.3f} lattice units")
        
        # Evaluate at specific distances
        r_test = [1.0, 2.0, np.pi/kf, 2*np.pi/kf]
        print(f"  V_RKKY values at characteristic distances:")
        for r in r_test:
            v = rkky_interaction(r, kf)
            print(f"    r = {r:.3f}: V = {v:+.4f} J₀")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("RKKY POTENTIAL ANALYSIS")
    print("="*70)
    
    kf_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    # Print properties
    print_rkky_properties(kf_values)
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = plot_rkky_potential()
    
    # Save figure
    output_path = "plots/rkky_potential.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
