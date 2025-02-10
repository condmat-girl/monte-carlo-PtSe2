# Triangular Lattice Monte Carlo Simulation with RKKY Interaction

## Introduction
This documentation provides an overview of the implementation of a Monte Carlo simulation for a 2D triangular lattice where atoms interact via the Ruderman-Kittel-Kasuya-Yosida (RKKY) interaction. The system evolves through the Monte Carlo Metropolis algorithm, which updates the orientation of magnetic moments and evaluates equilibrium configurations at different temperatures.

## How to use

### **Installation**
Ensure you have Python 3 and the necessary dependencies installed:
```sh
pip install numpy matplotlib scipy
```

### **Running the Simulation**
To execute the simulation, run:
```sh
python test.py
```

## Parameters
The following parameters can be modified in `simulation.py`:

- `rows`, `cols`: Dimensions of the lattice.
- `spacing`: Distance between lattice points.
- `k_f`, `U0`: Parameters defining the RKKY interaction.
- `steps`: Number of Monte Carlo steps per temperature.
- `temperatures`: List of temperatures to simulate.

## Output
- A 3D visualization of the lattice and magnetic moments.
- A plot showing the variation of magnetization (m_x) with temperature.


