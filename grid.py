import json, csv, os
import numpy as np
from scipy.interpolate import UnivariateSpline

from lattice import Lattice
from monte_carlo import MonteCarlo


def stderr(series, tau_int):
    x = np.asarray(series, float)
    n = x.size
    if n == 0:
        return float("nan")
    var = float(np.var(x, ddof=1)) if n > 1 else 0.0
    return float(np.sqrt(max(2.0 * tau_int, 1.0) * var / n))


def chi_peak(T, Chi):
    if len(T) < 3:
        i = int(np.argmax(Chi))
        return float(T[i]), float(Chi[i])
    s = UnivariateSpline(T, Chi, s=0, k=3)
    TT = np.linspace(float(np.min(T)), float(np.max(T)), 4001)
    YY = s(TT)
    i = int(np.argmax(YY))
    return float(TT[i]), float(YY[i])


def run_point(rows, cols, doping, kf, J0, T_grid, warmup, steps, method, outdir):
    lat = Lattice(rows, cols, doping, kf=kf, J0=J0)
    mc = MonteCarlo(lat, progress=False)

    meta = dict(
        rows=rows, cols=cols, N=len(lat.lattice_points),
        doping=doping, kF=kf, J0=J0, method=method,
        warmup=warmup, steps=steps, T_grid=list(map(float, T_grid))
    )
    entries = []

    for T in T_grid:
        info = mc.run_loop(warmup, steps, float(T), method=method)
        tauE, tauM = float(info["tau_E"]), float(info["tau_M"])

        E = mc.acc.energy
        M = mc.acc.magnetization
        MA = mc.acc.m_abs_array
        CH = mc.acc.susceptibility

        entry = dict(
            T=float(T),
            E=float(np.mean(E)) if len(E) else float("nan"),
            E_err=stderr(E, tauE),
            M=float(np.mean(M)) if len(M) else float("nan"),
            M_err=stderr(M, tauM),
            Mabs=float(np.mean(MA)) if len(MA) else float("nan"),
            Mabs_err=stderr(MA, tauM),
            Chi=float(np.mean(CH)) if len(CH) else float("nan"),
            Chi_err=stderr(CH, tauM),
            tau_E=tauE, tau_M=tauM
        )
        entries.append(entry)

    # locate χ peak
    T_arr = np.array([e["T"] for e in entries], float)
    Chi_arr = np.array([e["Chi"] for e in entries], float)
    Tpeak, Chimax = chi_peak(T_arr, Chi_arr)

    db = dict(meta=meta, entries=entries, peak=dict(T_peak=Tpeak, chi_max=Chimax))

    os.makedirs(outdir, exist_ok=True)
    tag_d = str(doping).replace(".", "p")
    tag_k = str(kf).replace(".", "p")
    fname = os.path.join(outdir, f"db_L{rows}_d{tag_d}_kf{tag_k}.json")
    with open(fname, "w") as f:
        json.dump(db, f, indent=2)
    return fname, (Tpeak, Chimax)


if __name__ == "__main__":
    # ----- fixed parameters -----
    rows, cols = 30, 30
    J0 = -1.0
    deltas = [0.05,0.075,0.1,0.15,0.20,0.25,0.30]
    kfs = [0.05,0.075,0.1,0.15,0.20,0.25,0.30]
    Tmin, Tmax, nT = 1.0, 65.0, 66
    warmup = 500
    steps = 2000
    method = "wolff"
    outdir = "db_wolff"

    # ----------------------------
    T_grid = np.linspace(Tmin, Tmax, nT)
    manifest_rows = []

    for d in deltas:
        for kf in kfs:
            fname, (Tp, Chimax) = run_point(
                rows=rows, cols=cols, doping=d, kf=kf, J0=J0,
                T_grid=T_grid, warmup=warmup, steps=steps,
                method=method, outdir=outdir
            )
            manifest_rows.append(dict(
                file=os.path.basename(fname), delta=d, kF=kf,
                T_peak=Tp, chi_max=Chimax
            ))
            print(f"Saved {fname} | T_peak={Tp:.3f}, chi_max={Chimax:.3f}")

    # summary CSV
    man_path = os.path.join(outdir, "manifest_peaks.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "delta", "kF", "T_peak", "chi_max"])
        w.writeheader(); w.writerows(manifest_rows)
    print(f"Manifest: {man_path}")
