import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from lattice import Lattice
from monte_carlo import MonteCarlo
from parallel_tempering import ParallelTempering


def run_mc_for_temps(base_lat, temps, warmup, steps, method="metropolis", progress=False):
    chis = []
    times = []
    for T in temps:
        lat = copy.deepcopy(base_lat)
        mc = MonteCarlo(lat, progress=progress)
        t0 = time.time()
        mc.run_loop(warmup, steps, T, method=method, save_warmup=False)
        t1 = time.time()
        times.append(t1 - t0)
        # compute susceptibility from magnetization samples (variance-based),
        # so it's consistent with ParallelTempering.compute_susceptibility
        try:
            chi = mc.acc.compute_susceptibility(mc.acc.magnetization, T, mc.lattice.N)
        except Exception:
            chi = float('nan')
        chis.append(float(chi))
    return np.asarray(chis), np.asarray(times)


def run_pt(base_lat, temps, method="metropolis", steps_per_exchange=200, warmup_per_replica=1000, n_epochs=500, rng_seed=1234, progress=False):
    pt = ParallelTempering(base_lat, temperatures=temps, method=method, steps_per_exchange=steps_per_exchange, warmup_per_replica=warmup_per_replica, progress=progress, rng_seed=rng_seed)
    out = pt.run(n_epochs)
    mags = out["magnetization_trace"]  # list per replica
    chis = []
    N = base_lat.N
    for i, Mtrace in enumerate(mags):
        M = np.asarray(Mtrace)
        if M.size:
            mean_M = M.mean()
            mean_M2 = (M**2).mean()
            chi = (N / temps[i]) * (mean_M2 - mean_M**2)
        else:
            chi = float('nan')
        chis.append(chi)
    return np.asarray(chis), out


def plot_results(temps, chi_mc, chi_pt, out_file="plots/pt_vs_mc_chi.png"):
    diff = chi_pt - chi_mc
    fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    ax[0].plot(temps, chi_mc, 'o-', label='MC')
    ax[0].plot(temps, chi_pt, 's--', label='Parallel Tempering')
    ax[0].set_ylabel('Susceptibility')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(temps, diff, 'k.-')
    ax[1].axhline(0, color='gray', linewidth=0.7)
    ax[1].set_ylabel('PT - MC')
    ax[1].set_xlabel('Temperature')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot to {out_file}")


def main():
    # parameters — tweak for speed/accuracy
    rows, cols = 6, 6
    doping = 0.6
    kf = 1.0
    J0 = 1.0

    temps = np.linspace(0.8, 3.0, 8)

    # create base lattice
    base_lat = Lattice(rows, cols, doping, kf=kf, J0=J0)

    # Monte Carlo settings — recorded samples = n_epochs to compare with PT records
    warmup_mc = 1000
    steps_mc = 500  # number of recorded production samples

    # Parallel tempering settings
    steps_per_exchange = 1000
    warmup_per_replica = 10000
    n_epochs = steps_mc  # one magnetization recorded per epoch in PT

    print('Running plain Monte Carlo for each T (this may take a while)...')
    chi_mc, times_mc = run_mc_for_temps(base_lat, temps, warmup_mc, steps_mc)

    print('Running Parallel Tempering...')
    chi_pt, pt_out = run_pt(copy.deepcopy(base_lat), temps, method='metropolis', steps_per_exchange=steps_per_exchange, warmup_per_replica=warmup_per_replica, n_epochs=n_epochs)

    # ensure plots directory exists
    import os
    os.makedirs('plots', exist_ok=True)

    plot_results(temps, chi_mc, chi_pt)

    # print short summary
    print('temps:', temps)
    print('chi_mc:', chi_mc)
    print('chi_pt:', chi_pt)
    print('swap_accept_rate_mean:', pt_out.get('swap_accept_rate_mean'))


if __name__ == '__main__':
    main()
