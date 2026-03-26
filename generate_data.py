"""
Generate magnetization and susceptibility data for Metropolis vs Wolff comparison.
Universal function for use in Jupyter notebooks with customizable parameters.
"""

import numpy as np
import json
import os
from tqdm import tqdm
from lattice import Lattice
from monte_carlo import MonteCarlo


def run_simulation(algo_name, rows, cols, doping, kf, T, warmup_steps, prod_steps):
    """
    Run single simulation and return mean and std of magnetization and susceptibility.
    
    Args:
        algo_name: 'metropolis' or 'wolff'
        rows, cols: Lattice dimensions
        doping: Doping concentration
        kf: Fermi wavevector
        T: Temperature
        warmup_steps: Warmup iterations
        prod_steps: Production iterations
    
    Returns:
        dict with keys: mag_mean, mag_err, chi_mean, chi_err
    """
    np.random.seed(None)  # Use random seed each time
    
    lat = Lattice(rows=rows, cols=cols, doping=doping, kf=kf, J0=1.0)
    mc = MonteCarlo(lat, progress=False)
    mc.T = T
    
    # Warmup phase
    if algo_name == 'metropolis':
        for _ in range(warmup_steps):
            mc.metropolis_step()
    else:  # wolff
        mc.precompute_bond_probabilities()
        for _ in range(warmup_steps):
            mc.wolff_step()
    
    # Production phase - collect statistics
    mag_samples = []
    chi_samples = []
    save_interval = max(1, prod_steps // 100)  # Collect ~100 samples
    
    for step in range(prod_steps):
        if algo_name == 'metropolis':
            mc.metropolis_step()
        else:
            mc.wolff_step()
        
        if step % save_interval == 0:
            mag_samples.append(np.abs(mc.M))
            chi_samples.append(mc.chi)
    
    mag_mean = np.mean(mag_samples)
    mag_std = np.std(mag_samples) / np.sqrt(len(mag_samples))  # Standard error
    chi_mean = np.mean(chi_samples)
    chi_std = np.std(chi_samples) / np.sqrt(len(chi_samples))  # Standard error
    
    return {
        "mag_mean": mag_mean,
        "mag_err": mag_std,
        "chi_mean": chi_mean,
        "chi_err": chi_std
    }


def generate_comparison_data(
    kf_values,
    doping_values,
    T_grid,
    rows=15,
    cols=15,
    metro_warmup=500,
    metro_prod=1000,
    wolff_warmup=100,
    wolff_prod=200,
    data_dir="data",
    filename="mag_chi_comparison.json",
    verbose=True
):
    """
    Generate universal comparison data for Metropolis vs Wolff algorithms.
    
    Args:
        kf_values: Array of Fermi wavevector values
        doping_values: Array of doping concentrations
        T_grid: Array of temperature values
        rows, cols: Lattice dimensions
        metro_warmup: Metropolis warmup steps
        metro_prod: Metropolis production steps
        wolff_warmup: Wolff warmup steps (typically much smaller)
        wolff_prod: Wolff production steps (typically much smaller)
        data_dir: Directory to save JSON
        filename: Output JSON filename
        verbose: Print progress information
    
    Returns:
        dict with structure:
        {
            "T_grid": [...],
            "kf_values": [...],
            "doping_values": [...],
            "metropolis": [
                {"kf": X, "doping": Y, "mag_mean": [...], "mag_err": [...], ...},
                ...
            ],
            "wolff": [
                {"kf": X, "doping": Y, "mag_mean": [...], "mag_err": [...], ...},
                ...
            ]
        }
    """
    os.makedirs(data_dir, exist_ok=True)
    
    kf_values = np.asarray(kf_values)
    doping_values = np.asarray(doping_values)
    T_grid = np.asarray(T_grid)
    
    data = {
        "T_grid": T_grid.tolist(),
        "kf_values": kf_values.tolist(),
        "doping_values": doping_values.tolist(),
        "metropolis": [],
        "wolff": []
    }
    
    if verbose:
        print("=" * 70)
        print("METROPOLIS vs WOLFF COMPARISON DATA GENERATION")
        print("=" * 70)
        print(f"Lattice: {rows}×{cols}")
        print(f"kF values: {kf_values}")
        print(f"Doping values: {doping_values}")
        print(f"Temperature grid: {len(T_grid)} points from {T_grid[0]:.2f} to {T_grid[-1]:.2f}")
        print(f"\nMetropolis:  warmup={metro_warmup}, prod={metro_prod}")
        print(f"Wolff:       warmup={wolff_warmup}, prod={wolff_prod}")
        print()
    
    total_sims = len(kf_values) * len(doping_values) * len(T_grid) * 2
    
    # Metropolis simulations
    if verbose:
        print(f"[Metropolis Algorithm - {len(kf_values)*len(doping_values)*len(T_grid)} sims]")
    
    for doping in doping_values:
        for kf in kf_values:
            entry = {
                "kf": float(kf),
                "doping": float(doping),
                "mag_mean": [],
                "mag_err": [],
                "chi_mean": [],
                "chi_err": []
            }
            
            pbar = tqdm(T_grid, desc=f"Metro kf={kf:.2f} dop={doping:.2f}", 
                       unit="T", disable=not verbose)
            for T in pbar:
                result = run_simulation("metropolis", rows, cols, doping, kf, T,
                                       metro_warmup, metro_prod)
                entry["mag_mean"].append(result["mag_mean"])
                entry["mag_err"].append(result["mag_err"])
                entry["chi_mean"].append(result["chi_mean"])
                entry["chi_err"].append(result["chi_err"])
            
            data["metropolis"].append(entry)
    
    # Wolff simulations
    if verbose:
        print(f"\n[Wolff Algorithm - {len(kf_values)*len(doping_values)*len(T_grid)} sims]")
    
    for doping in doping_values:
        for kf in kf_values:
            entry = {
                "kf": float(kf),
                "doping": float(doping),
                "mag_mean": [],
                "mag_err": [],
                "chi_mean": [],
                "chi_err": []
            }
            
            pbar = tqdm(T_grid, desc=f"Wolff kf={kf:.2f} dop={doping:.2f}", 
                       unit="T", disable=not verbose)
            for T in pbar:
                result = run_simulation("wolff", rows, cols, doping, kf, T,
                                       wolff_warmup, wolff_prod)
                entry["mag_mean"].append(result["mag_mean"])
                entry["mag_err"].append(result["mag_err"])
                entry["chi_mean"].append(result["chi_mean"])
                entry["chi_err"].append(result["chi_err"])
            
            data["wolff"].append(entry)
    
    # Save to JSON
    output_file = os.path.join(data_dir, filename)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    if verbose:
        print(f"\n✓ Data saved to {output_file}")
        print(f"✓ Total simulations: {total_sims}")
    
    return data


# Example usage for command line
if __name__ == "__main__":
    # Default configuration
    kf_values = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    doping_values = np.array([0.5])
    T_grid = np.linspace(0.2, 3.0, 15)
    
    data = generate_comparison_data(
        kf_values=kf_values,
        doping_values=doping_values,
        T_grid=T_grid,
        rows=15,
        cols=15,
        metro_warmup=500,
        metro_prod=1000,
        wolff_warmup=100,
        wolff_prod=200,
        filename="mag_chi_vs_kf.json",
        verbose=True
    )
