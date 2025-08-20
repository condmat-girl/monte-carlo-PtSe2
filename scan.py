# scan_irregular.py
import numpy as np
from lattice import Lattice
from monte_carlo import MonteCarlo

def run_one_method(method, temperatures, rows, cols, doping, kf, J0, warmup, steps):
    Mabs_mean, Mabs_err = [], []
    Chi_mean,  Chi_err  = [], []
    Tau_list            = []

    for T in temperatures:
        lat = Lattice(rows=rows, cols=cols, doping=doping, kf=kf, J0=J0)
        mc  = MonteCarlo(lat)
        mc.run_loop(warmup_steps=warmup, steps=steps, T=T, method=method)

        acc = mc.acc
        m   = np.asarray(acc.magnetization)       # raw magnetization series (±)
        m_abs = np.abs(m)

        # Use your Accumulator's ACF + tau
        acf = acc.autocorr_fft(m)                 # unbiased default in your impl
        tau = acc.tau_int_from_acf(acf)
        Tau_list.append(tau)

        # Means + correlated errors
        Mabs_mean.append(m_abs.mean())
        Mabs_err.append(acc.calculate_error(m_abs, tau))

        # Susceptibility from fluctuations of M (Z2-symmetric)
        chi_series = (lat.N / T) * (m - m.mean())**2
        Chi_mean.append(chi_series.mean())
        Chi_err.append(acc.calculate_error(chi_series, tau))

    return (np.array(Mabs_mean), np.array(Mabs_err),
            np.array(Chi_mean),  np.array(Chi_err),
            np.array(Tau_list))

if __name__ == "__main__":
    # ----- grids -----
    temperatures = np.linspace(1.0, 12.0, 31)
    rows, cols   = 12, 12

    doping_list  =  np.linspace(0.1, 0.5, 6) #[0.1,0.2, 0.3, 0.4]      
    kf_list      =  np.linspace(0.1, 0.5, 6) #[0.1,0.2, 0.3, 0.4]     
    J0           = -1.0                
    # sampling (Metropolis typically needs more)
    warmup_w, steps_w = 100,     1_000
    warmup_m, steps_m = 100_000, 1_000_000

    methods = ["wolff", "metropolis"]
    nD, nK, nT = len(doping_list), len(kf_list), len(temperatures)

    # allocate
    results = {m: {
        "Mabs_mean": np.zeros((nD, nK, nT)),
        "Mabs_err":  np.zeros((nD, nK, nT)),
        "Chi_mean":  np.zeros((nD, nK, nT)),
        "Chi_err":   np.zeros((nD, nK, nT)),
        "Tau":       np.zeros((nD, nK, nT)),
    } for m in methods}

    for di, dop in enumerate(doping_list):
        for ki, kf in enumerate(kf_list):
            print(f"\n=== doping={dop:.3f}, kf={kf:.3f} ===")

            Mw, Mw_err, Chiw, Chiw_err, Tauw = run_one_method(
                "wolff", temperatures, rows, cols, dop, kf, J0, warmup_w, steps_w
            )
            results["wolff"]["Mabs_mean"][di, ki] = Mw
            results["wolff"]["Mabs_err"][di,  ki] = Mw_err
            results["wolff"]["Chi_mean"][di,  ki] = Chiw
            results["wolff"]["Chi_err"][di,   ki] = Chiw_err
            results["wolff"]["Tau"][di,       ki] = Tauw

            Mm, Mm_err, Chim, Chim_err, Taum = run_one_method(
                "metropolis", temperatures, rows, cols, dop, kf, J0, warmup_m, steps_m
            )
            results["metropolis"]["Mabs_mean"][di, ki] = Mm
            results["metropolis"]["Mabs_err"][di,  ki] = Mm_err
            results["metropolis"]["Chi_mean"][di,  ki] = Chim
            results["metropolis"]["Chi_err"][di,   ki] = Chim_err
            results["metropolis"]["Tau"][di,       ki] = Taum

    # --- save NPZ ---
    np.savez(
        "scan_results.npz",
        temperatures=temperatures,
        rows=rows, cols=cols,
        doping_list=np.array(doping_list),
        kf_list=np.array(kf_list),
        J0=J0,
        wolff_Mabs_mean=results["wolff"]["Mabs_mean"],
        wolff_Mabs_err=results["wolff"]["Mabs_err"],
        wolff_Chi_mean=results["wolff"]["Chi_mean"],
        wolff_Chi_err=results["wolff"]["Chi_err"],
        wolff_Tau=results["wolff"]["Tau"],
        metro_Mabs_mean=results["metropolis"]["Mabs_mean"],
        metro_Mabs_err=results["metropolis"]["Mabs_err"],
        metro_Chi_mean=results["metropolis"]["Chi_mean"],
        metro_Chi_err=results["metropolis"]["Chi_err"],
        metro_Tau=results["metropolis"]["Tau"],
    )

    # --- save flat TXT tables (one per method) ---
    header = "doping\tkf\tT\tMabs\tMabs_err\tChi\tChi_err\tTau"
    for method in methods:
        R = results[method]
        rows_txt = []
        for di, dop in enumerate(doping_list):
            for ki, kf in enumerate(kf_list):
                for ti, T in enumerate(temperatures):
                    rows_txt.append([
                        dop, kf, T,
                        R["Mabs_mean"][di,ki,ti],
                        R["Mabs_err"][di,ki,ti],
                        R["Chi_mean"][di,ki,ti],
                        R["Chi_err"][di,ki,ti],
                        R["Tau"][di,ki,ti],
                    ])
        np.savetxt(f"scan_{method}.txt",
                   np.array(rows_txt), fmt="%.6g",
                   delimiter="\t", header=header)
    print("Saved: scan_results.npz, scan_wolff.txt, scan_metropolis.txt")

