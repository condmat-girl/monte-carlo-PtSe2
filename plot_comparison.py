"""
Plot Metropolis vs Wolff comparison from saved JSON data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("data/mag_chi_vs_kf.json", "r") as f:
    data = json.load(f)

T_grid = np.array(data["T_grid"])
kf_values = np.array(data["kf_values"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Susceptibility vs Temperature
ax = axes[0]
for entry in data["metropolis"]:
    kf = entry["kf"]
    chi = np.array(entry["chi_mean"])
    chi_err = np.array(entry["chi_err"])
    ax.errorbar(T_grid, chi, yerr=chi_err, label=f"Metro kf={kf:.2f}", 
                marker='o', capsize=4, linestyle='-', linewidth=1.5)

for entry in data["wolff"]:
    kf = entry["kf"]
    chi = np.array(entry["chi_mean"])
    chi_err = np.array(entry["chi_err"])
    ax.errorbar(T_grid, chi, yerr=chi_err, label=f"Wolff kf={kf:.2f}", 
                marker='s', capsize=4, linestyle='--', linewidth=1.5, alpha=0.8)

ax.set_xlabel("T", fontsize=12)
ax.set_ylabel(r"$\chi(T)$", fontsize=12)
ax.legend(fontsize=8, ncol=2, loc='best')
ax.grid(True, alpha=0.3)
ax.set_title("Magnetic Susceptibility vs Temperature", fontsize=13, fontweight='bold')

# Plot 2: Magnetization vs Temperature
ax = axes[1]
for entry in data["metropolis"]:
    kf = entry["kf"]
    mag = np.array(entry["mag_mean"])
    mag_err = np.array(entry["mag_err"])
    ax.errorbar(T_grid, mag, yerr=mag_err, label=f"Metro kf={kf:.2f}", 
                marker='o', capsize=4, linestyle='-', linewidth=1.5)

for entry in data["wolff"]:
    kf = entry["kf"]
    mag = np.array(entry["mag_mean"])
    mag_err = np.array(entry["mag_err"])
    ax.errorbar(T_grid, mag, yerr=mag_err, label=f"Wolff kf={kf:.2f}", 
                marker='s', capsize=4, linestyle='--', linewidth=1.5, alpha=0.8)

ax.set_xlabel("T", fontsize=12)
ax.set_ylabel(r"$|M|(T)$", fontsize=12)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, ncol=2, loc='best')
ax.grid(True, alpha=0.3)
ax.set_title("Magnetization vs Temperature", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("plots/mag_chi_comparison.png", dpi=180, bbox_inches='tight')
print("✓ Saved: plots/mag_chi_comparison.png")
plt.show()
