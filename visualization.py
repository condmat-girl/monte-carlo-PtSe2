# visualization.py  – all plotting, no stored data duplicates
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


class Visualization:

    def __init__(self, lattice, accumulator):
        self.lattice = lattice
        self.acc     = accumulator

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
        acf = self.acc.autocorrelation(m)
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
        corr = self.acc.compute_pair_correlation()
        r    = self.lattice.r_ij
        plt.figure(figsize=(6, 4))
        plt.plot(r, corr, ".")
        plt.xlabel("r")
        plt.ylabel("⟨S_i S_j⟩(r)")
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
