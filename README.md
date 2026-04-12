
# Monte Carlo Simulation of RKKY-Coupled Magnetic Impurities in Two Dimensions

This repository provides a Python-based implementation of Monte Carlo simulations for a two-dimensional lattice of magnetic impurities coupled via long-range RKKY interactions. The system is modeled as classical Ising spins randomly distributed on a triangular lattice, and simulations are performed using both the Metropolis algorithm and a modified Wolff cluster update algorithm that respects the oscillatory nature of RKKY coupling.

---


## Features

- **Customizable lattice parameters**: triangular geometry, doping level, interaction constants.
- **Long-range oscillatory interaction**: modeled via 2D RKKY theory with Bessel functions.
- **Two update algorithms**:
  - Local updates via the Metropolis-Hastings algorithm.
  - Global cluster updates via a detailed-balance-preserving Wolff algorithm (supports AFM/FM bonds).
- **Autocorrelation-aware statistics**: integrated autocorrelation time used to adaptively determine thermalization.
- **Visualization tools**: includes real-space spin plots, correlation functions ⟨SᵢSⱼ⟩(r), and their Fourier transforms.

---

## Installation

Python 3.8+ is recommended. To install the required dependencies:

```bash
pip install numpy scipy matplotlib tqdm
````

---

## Usage

Clone the repository and run the simulation using:

```bash
python main.py
```

The main entry point initializes the lattice, runs the simulation, and produces the visualizations.

Simulation parameters can be modified in `main.py`:

```python
rows = 20         # lattice height
cols = 20         # lattice width
doping = 0.3      # fraction of occupied lattice sites
kf = 1.0          # Fermi wavevector
J0 = 1.0          # coupling constant
T = 1.5           # temperature
steps = 1000      # number of Monte Carlo steps
```

To choose the simulation method:

```python
mc.run_loop(steps=steps, T=T, method="wolff")       # Wolff cluster algorithm
mc.run_loop(steps=steps, T=T, method="metropolis")  # Metropolis local updates
```

---

## Project Structure

```
monte-carlo-PtSe2/
│
├── example.ipynb             # Example simulation script
├── lattice.py          # Lattice construction and RKKY interaction matrix
├── monte_carlo.py      # Metropolis and Wolff simulation engines
├── accumulator.py      # Observable tracking and autocorrelation estimation
├── visualization.py    # Plotting tools for lattice and correlation functions
```

---

## Output

After a successful run, the following plots are generated:

* **Lattice plot** with spin orientation (color-coded)
* **Magnetization vs. MC step**, with error bars based on autocorrelation time
* **Pair correlation function** ⟨SᵢSⱼ⟩(r)

These provide insights into the onset of magnetic order and correlation length in the system.

---
