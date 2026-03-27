import os, hashlib, json
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.interpolate import UnivariateSpline

from lattice import Lattice
from monte_carlo import MonteCarlo


class ObservablesGrid:

    def __init__(self,
                 deltas, kfs,
                 L=14, J0=-1.0,
                 T_grid=None,
                 warmup=1000, steps=5000,
                 method="wolff",
                 use_kac=False, kac_mode="abs",
                 outdir="plots",
                 cache_dir="cache",
                 cmap="cool"):
        
        self.deltas = list(deltas)
        self.kfs = list(kfs)
        self.L, self.J0 = int(L), float(J0)
        self.T_grid = np.linspace(2.0, 31.0, 30) if T_grid is None else np.asarray(T_grid, float)
        
        if isinstance(warmup, dict):
            self.warmup = warmup
        else:
            self.warmup = {"metropolis": int(warmup), "wolff": max(1, int(warmup) // 10)}

        if isinstance(steps, dict):
            self.steps = steps
        else:
            self.steps = {"metropolis": int(steps), "wolff": max(1, int(steps) // 10)}
            
        self.method = method
        self.use_kac, self.kac_mode = bool(use_kac), kac_mode

        self.outdir = outdir; os.makedirs(self.outdir, exist_ok=True)
        self.cache_dir = cache_dir; os.makedirs(self.cache_dir, exist_ok=True)

        self.results = {}     # (delta,kf) -> dict(T,E,M,Chi)
        self.peaks_map = {}   # (delta,kf) -> (T_peak, chi_max)

        self.cmap_name = cmap
        self.cmap = plt.get_cmap(cmap)

    def _set_cycle(self, ax, n):
        ax.set_prop_cycle(cycler(color=self.cmap(np.linspace(0, 1, max(1, n)))))

    @staticmethod
    def _as_list(values, name):
        if values is None:
            return None
        if np.isscalar(values):
            return [float(values)]
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError(f"{name} must not be empty")
        return list(arr)

    @staticmethod
    def _estimate_error(series, tau_int=np.nan):
        x = np.asarray(series, dtype=float)
        if x.size < 2:
            return np.nan
        var = float(np.var(x, ddof=1))
        if not np.isfinite(var):
            return np.nan
        tau = float(tau_int) if np.isfinite(tau_int) else 0.0
        # If tau_int is available, account for autocorrelation inflation.
        if tau > 0:
            return float(np.sqrt(2.0 * tau * var / x.size))
        return float(np.sqrt(var / x.size))

    # --- FIX: Error estimation for Susceptibility using Block Means ---
    @staticmethod
    def _estimate_chi_error(m_series, T, N_spins, blocks=10):
        """Estimates the error of susceptibility using the method of batch means."""
        if len(m_series) < blocks * 2:
            return np.nan
        
        # Split the time series into independent blocks
        m_splits = np.array_split(np.abs(m_series), blocks)
        chi_blocks = []
        
        for block in m_splits:
            m_sq_mean = np.mean(block**2)
            m_mean = np.mean(block)
            chi_val = (N_spins / T) * (m_sq_mean - m_mean**2)
            chi_blocks.append(chi_val)
            
        # The error is the standard error of the block estimates
        return float(np.std(chi_blocks, ddof=1) / np.sqrt(blocks))

    def _key(self, delta, kf):
        meta = dict(L=self.L, J0=self.J0, delta=float(delta), kf=float(kf),
                    warmup=self.warmup, steps=self.steps, method=self.method,
                    use_kac=self.use_kac, kac_mode=self.kac_mode,
                    T_grid=list(map(float, self.T_grid)))
        key_str = json.dumps(meta, sort_keys=True)
        h = hashlib.sha1(key_str.encode()).hexdigest()[:16]
        return f"L{self.L}_d{delta}_kf{kf}_{h}"

    def _cache_path(self, delta, kf):
        return os.path.join(self.cache_dir, self._key(delta, kf) + ".npz")

    def _save_cache(self, delta, kf, data):
        np.savez_compressed(self._cache_path(delta, kf), **data)

    def _load_cache(self, delta, kf):
        path = self._cache_path(delta, kf)
        if os.path.exists(path):
            z = np.load(path)
            return {k: z[k] for k in z.files}
        return None

    @staticmethod
    def _kac_normalize(lat, mode="abs"):
        J = lat.interaction_matrix
        if mode == "abs":
            S = np.mean(np.sum(np.abs(J), axis=1))
        elif mode == "rms":
            S = np.sqrt(np.mean(np.sum(J*J, axis=1)))
        else:
            raise ValueError("kac_mode must be 'abs' or 'rms'")
        if S != 0:
            lat.interaction_matrix = J / S

    def _run_vs_T(self, delta, kf, method=None):
        method_used = self.method if method is None else method
        lat = Lattice(self.L, self.L, delta, kf=kf, J0=self.J0)
        if self.use_kac:
            self._kac_normalize(lat, mode=self.kac_mode)
        mc = MonteCarlo(lat)

        E, Mabs, Chi = [], [], []
        E_err, M_err, Chi_err = [], [], []
        
        # Pull method-specific counts
        w = self.warmup.get(method_used, 1000)
        s = self.steps.get(method_used, 5000)
        N_spins = self.L * self.L

        for T in self.T_grid:
            mc.acc.energy.clear(); mc.acc.magnetization.clear(); mc.acc.susceptibility.clear()
            run_info = mc.run_loop(w, s, float(T), method=method_used)

            e_series = np.asarray(mc.acc.energy, dtype=float)
            m_series = np.asarray(mc.acc.magnetization, dtype=float)

            # Calculate Means
            E.append(np.mean(e_series))
            m_abs_mean = np.mean(np.abs(m_series))
            Mabs.append(m_abs_mean)

            # --- FIX: Calculate Chi directly from variance of m_series ---
            m_sq_mean = np.mean(m_series**2)
            chi_val = (N_spins / float(T)) * (m_sq_mean - m_abs_mean**2)
            Chi.append(chi_val)

            tau_e = run_info.get("tau_E", np.nan)
            tau_m = run_info.get("tau_M", np.nan)
            
            E_err.append(self._estimate_error(e_series, tau_e))
            M_err.append(self._estimate_error(np.abs(m_series), tau_m))
            Chi_err.append(self._estimate_chi_error(m_series, float(T), N_spins, blocks=10))

        data = dict(T=self.T_grid.copy(),
                    E=np.asarray(E, float),
                    M=np.asarray(Mabs, float),
                    Chi=np.asarray(Chi, float),
                    E_err=np.asarray(E_err, float),
                    M_err=np.asarray(M_err, float),
                    Chi_err=np.asarray(Chi_err, float))
        return data

    def ensure(self, delta, kf, force=False):
        key = (float(delta), float(kf))
        if (not force) and key in self.results:
            return self.results[key]

        if not force:
            cached = self._load_cache(*key)
            if cached is not None:
                self.results[key] = cached
                Tstar, chimax = self._chi_peak(cached["T"], cached["Chi"])
                self.peaks_map[key] = (Tstar, chimax)
                return cached

        data = self._run_vs_T(*key, method=self.method)
        self.results[key] = data
        self._save_cache(*key, data)
        Tstar, chimax = self._chi_peak(data["T"], data["Chi"])
        self.peaks_map[key] = (Tstar, chimax)

        i_max = int(np.argmax(data["Chi"]))
        if i_max == 0 or i_max == len(data["T"]) - 1:
            print(f"[warn] peak at boundary: δ={delta}, kF={kf}, T≈{data['T'][i_max]:.2f}. Refine T-grid.")
        return data

    @staticmethod
    def _chi_peak(T, chi):
        # --- FIX: s=None allows the spline to smooth rather than overfit ---
        s = UnivariateSpline(T, chi, s=None, k=3)
        TT = np.linspace(T.min(), T.max(), 4001)
        y = s(TT)
        i = int(np.argmax(y))
        return float(TT[i]), float(y[i])

    def run_grid(self, subset=None):
        pairs = subset or [(d, kf) for d in self.deltas for kf in self.kfs]
        for d, k in pairs:
            self.ensure(d, k)
        peaks = []
        for (d, k), (tstar, chimax) in self.peaks_map.items():
            peaks.append([d, k, tstar, chimax])
        if peaks:
            peaks = np.array(peaks, float)
            np.savetxt(os.path.join(self.outdir, "peaks_delta_kf.txt"),
                       peaks, header="delta  kF  T_peak  chi_max")
        return self.results

    def _filter_by_delta(self, delta):
        sub = {}
        for kf in self.kfs:
            key = (float(delta), float(kf))
            if key in self.results:
                sub[kf] = self.results[key]
        return sub

    def _filter_by_kf(self, kf):
        sub = {}
        for delta in self.deltas:
            key = (float(delta), float(kf))
            if key in self.results:
                sub[delta] = self.results[key]
        return sub

    def plot_by_kf(self, delta, fname=None):
        for kf in self.kfs:
            self.ensure(delta, kf)

        sub = self._filter_by_delta(delta)
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        n = len(sub)
        for ax in axs: self._set_cycle(ax, n)

        for kf, d in sorted(sub.items()):
            T = d["T"]
            axs[0].plot(T, d["E"],   'o-', ms=4, label=f'kF={kf:g}')
            axs[1].plot(T, d["M"],   'o-', ms=4, label=f'kF={kf:g}')
            axs[2].plot(T, d["Chi"], 'o-', ms=4, label=f'kF={kf:g}')
        axs[0].set_xlabel("T"); axs[0].set_ylabel("⟨E⟩")
        axs[1].set_xlabel("T"); axs[1].set_ylabel("⟨|M|⟩")
        axs[2].set_xlabel("T"); axs[2].set_ylabel("χ ")
        for ax in axs: ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        if fname is None: fname = f"by_kf_delta{str(delta).replace('.','p')}.png"
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180); plt.show()
        return path

    def plot_by_delta(self, kf, fname=None):
        for delta in self.deltas:
            self.ensure(delta, kf)

        sub = self._filter_by_kf(kf)
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        n = len(sub)
        for ax in axs: self._set_cycle(ax, n)

        for delta, d in sorted(sub.items()):
            T = d["T"]
            axs[0].plot(T, d["E"],   'o-', ms=4, label=f'δ={delta:g}')
            axs[1].plot(T, d["M"],   'o-', ms=4, label=f'δ={delta:g}')
            axs[2].plot(T, d["Chi"], 'o-', ms=4, label=f'δ={delta:g}')
        axs[0].set_xlabel("T"); axs[0].set_ylabel("⟨E⟩")
        axs[1].set_xlabel("T"); axs[1].set_ylabel("⟨|M|⟩")
        axs[2].set_xlabel("T"); axs[2].set_ylabel("χ")
        for ax in axs: ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        if fname is None: fname = f"by_delta_kf{str(kf).replace('.','p')}.png"
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180); plt.show()
        return path

    def run_susceptibility_sweep(self, deltas=None, kfs=None, methods=("metropolis", "wolff"), force=False):
        deltas_use = self._as_list(deltas, "deltas") if deltas is not None else list(map(float, self.deltas))
        kfs_use = self._as_list(kfs, "kfs") if kfs is not None else list(map(float, self.kfs))
        methods_use = [str(m) for m in methods]
        if len(methods_use) == 0:
            raise ValueError("methods must contain at least one method name")

        results = {}
        method_before = self.method
        try:
            for method in methods_use:
                self.method = method
                method_data = {}
                for delta in deltas_use:
                    for kf in kfs_use:
                        data = self.ensure(delta, kf, force=force)
                        method_data[(float(delta), float(kf))] = data
                results[method] = method_data
        finally:
            self.method = method_before
        return results

    def plot_susceptibility_compare(self, deltas=None, kfs=None, methods=("metropolis", "wolff"),
                                    by="kf", fixed_value=None, fname=None):
        by = str(by).lower()
        if by not in ("kf", "delta"):
            raise ValueError("by must be 'kf' or 'delta'")

        deltas_use = self._as_list(deltas, "deltas") if deltas is not None else list(map(float, self.deltas))
        kfs_use = self._as_list(kfs, "kfs") if kfs is not None else list(map(float, self.kfs))

        if by == "kf":
            if fixed_value is None:
                fixed_value = deltas_use[0]
            fixed_delta = float(fixed_value)
            self.run_susceptibility_sweep(deltas=[fixed_delta], kfs=kfs_use, methods=methods)
        else:
            if fixed_value is None:
                fixed_value = kfs_use[0]
            fixed_kf = float(fixed_value)
            self.run_susceptibility_sweep(deltas=deltas_use, kfs=[fixed_kf], methods=methods)

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        method_styles = {"metropolis": "-", "wolff": "--"}
        method_markers = {"metropolis": "o", "wolff": "s"}

        for method in methods:
            method = str(method)
            method_before = self.method
            self.method = method
            try:
                if by == "kf":
                    for kf in kfs_use:
                        d = self.ensure(fixed_delta, kf)
                        ax.errorbar(
                            d["T"], d["Chi"], yerr=d.get("Chi_err", None),
                            ls=method_styles.get(method, "-"),
                            marker=method_markers.get(method, "o"),
                            ms=3.5, capsize=2.5,
                            label=f"{method}, kF={kf:g}"
                        )
                else:
                    for delta in deltas_use:
                        d = self.ensure(delta, fixed_kf)
                        ax.errorbar(
                            d["T"], d["Chi"], yerr=d.get("Chi_err", None),
                            ls=method_styles.get(method, "-"),
                            marker=method_markers.get(method, "o"),
                            ms=3.5, capsize=2.5,
                            label=f"{method}, δ={delta:g}"
                        )
            finally:
                self.method = method_before

        ax.set_xlabel("T")
        ax.set_ylabel("χ")
        if by == "kf":
            ax.set_title(f"Susceptibility vs T at δ={fixed_delta:g}")
        else:
            ax.set_title(f"Susceptibility vs T at kF={fixed_kf:g}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        if fname is None:
            if by == "kf":
                fname = f"sus_compare_by_kf_delta{str(fixed_delta).replace('.','p')}.png"
            else:
                fname = f"sus_compare_by_delta_kf{str(fixed_kf).replace('.','p')}.png"
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180)
        plt.show()
        return path

    def plot_peak_map(self, what="T_peak", fname=None):
        self.run_grid() 
        Z = np.full((len(self.deltas), len(self.kfs)), np.nan)
        for i, d in enumerate(self.deltas):
            for j, k in enumerate(self.kfs):
                key = (float(d), float(k))
                if key not in self.peaks_map: continue
                tstar, chimax = self.peaks_map[key]
                Z[i, j] = tstar if what == "T_peak" else chimax

        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        im = ax.pcolormesh(self.kfs, self.deltas, Z, shading="nearest", cmap=self.cmap)
        cb = plt.colorbar(im); cb.set_label("T_peak" if what=="T_peak" else "χ_max")
        ax.set_xlabel("kF"); ax.set_ylabel("δ")
        plt.tight_layout()
        if fname is None:
            tag = "Tpeak" if what == "T_peak" else "Chimax"
            fname = f"map_{tag}.png"
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180); plt.show()
        return path