import os, hashlib, json
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.interpolate import UnivariateSpline

from lattice import Lattice
from monte_carlo import MonteCarlo


class ObservablesGrid:
    """Run and plot E, |M|, χ, and Binder cumulant over arbitrary combinations
    of (delta, kf, L, method).

    Parameters
    ----------
    deltas, kfs : array-like
        Default sweep values used when callers don't supply their own lists.
    Ls : int or list[int]
        Lattice sizes. Can be a single int (original behaviour) or a list,
        e.g. [14, 20, 28] for finite-size scaling.
    J0 : float
    T_grid : array-like or None
        Temperature points. Default: np.linspace(2, 31, 30).
    warmup, steps : int or dict
        Either a plain int applied to every method, or a per-method dict,
        e.g. {"metropolis": 2000, "wolff": 200}.
    method : str
        Default method when none is specified in ensure / plot calls.
    use_kac, kac_mode : Kac normalisation switches.
    outdir, cache_dir : output directories.
    cmap : matplotlib colormap name.
    chi_mode : {"step_average", "fluctuation"}
        How χ is computed from a production run:

        - ``step_average`` (default): mean of per-step values stored in
          ``MonteCarlo`` as ``N(1-M²)/T`` — matches the original pipeline and
          typical peak-shaped plots you had before.

        - ``fluctuation``: thermodynamic susceptibility
          ``(N/T)(⟨M²⟩ - ⟨M⟩²)`` from the full magnetization time series.
    """

    def __init__(self,
                 deltas, kfs,
                 Ls=14, J0=-1.0,
                 T_grid=None,
                 warmup=1000, steps=5000,
                 method="wolff",
                 chi_mode="step_average",
                 use_kac=False, kac_mode="abs",
                 outdir="plots",
                 cache_dir="cache",
                 cmap="cool"):

        self.deltas = list(map(float, deltas))
        self.kfs    = list(map(float, kfs))
        self.Ls     = [int(Ls)] if np.isscalar(Ls) else [int(x) for x in Ls]
        self.L      = self.Ls[0]          # convenience alias (single-L workflows)
        self.J0     = float(J0)

        self.T_grid = (np.linspace(2.0, 31.0, 30)
                       if T_grid is None else np.asarray(T_grid, float))

        self.warmup = warmup if isinstance(warmup, dict) else int(warmup)
        self.steps  = steps  if isinstance(steps,  dict) else int(steps)

        self.method = str(method)
        cm = str(chi_mode).lower()
        if cm not in ("step_average", "fluctuation"):
            raise ValueError("chi_mode must be 'step_average' or 'fluctuation'")
        self.chi_mode = cm
        self.use_kac  = bool(use_kac)
        self.kac_mode = kac_mode

        self.outdir    = outdir;    os.makedirs(self.outdir,    exist_ok=True)
        self.cache_dir = cache_dir; os.makedirs(self.cache_dir, exist_ok=True)

        # Keys: (delta, kf, L, method)
        self.results   = {}
        self.peaks_map = {}

        self.cmap_name = cmap
        self.cmap      = plt.get_cmap(cmap)

    # Internal helpers

    def _get_warmup(self, method):
        if isinstance(self.warmup, dict):
            return self.warmup.get(method, 1000)
        return int(self.warmup)

    def _get_steps(self, method):
        if isinstance(self.steps, dict):
            return self.steps.get(method, 5000)
        return int(self.steps)

    def _set_cycle(self, ax, n):
        ax.set_prop_cycle(cycler(color=self.cmap(np.linspace(0, 1, max(1, n)))))

    @staticmethod
    def _norm_methods(methods):
        """Accept None, a single string, or an iterable; return a list of strings.

        None  -> [self.method] handled by callers before calling this.
        """
        if isinstance(methods, str):
            return [methods]
        return [str(m) for m in methods]

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
    def _kac_normalize(lat, mode="abs"):
        J = lat.interaction_matrix
        if mode == "abs":
            S = np.mean(np.sum(np.abs(J), axis=1))
        elif mode == "rms":
            S = np.sqrt(np.mean(np.sum(J * J, axis=1)))
        else:
            raise ValueError("kac_mode must be 'abs' or 'rms'")
        if S != 0:
            lat.interaction_matrix = J / S

    # Error estimation

    @staticmethod
    def _block_error(series, n_blocks=20):
        """Standard error via block means (correlated-sample friendly).

        Uses as many blocks as the data length supports. If too few samples are
        available, returns np.nan (caller may decide how to display it).
        """
        x = np.asarray(series, dtype=float)
        if x.size < 4:
            return np.nan
        nb = int(min(n_blocks, x.size // 2))
        if nb < 2:
            return np.nan
        blocks = np.array_split(x, nb)
        means  = np.array([b.mean() for b in blocks])
        return float(np.std(means, ddof=1) / np.sqrt(nb))

    @staticmethod
    def _chi_block_error(m_series, T, N_spins, n_blocks=20):
        """Error of χ = (N/T)(⟨M²⟩ - ⟨M⟩²) via block means."""
        x = np.asarray(m_series, dtype=float)
        if x.size < 4:
            return np.nan
        nb = int(min(n_blocks, x.size // 2))
        if nb < 2:
            return np.nan
        blocks    = np.array_split(x, nb)
        chi_blocks = []
        for b in blocks:
            m2   = np.mean(b ** 2)
            m_sq = np.mean(b) ** 2
            chi_blocks.append((N_spins / T) * (m2 - m_sq))
        return float(np.std(chi_blocks, ddof=1) / np.sqrt(nb))

    @staticmethod
    def _binder_block_error(m_series, n_blocks=20):
        """Error of Binder cumulant U4 = 1 - ⟨M⁴⟩/(3⟨M²⟩²) via block means."""
        x = np.asarray(m_series, dtype=float)
        if x.size < 4:
            return np.nan
        nb = int(min(n_blocks, x.size // 2))
        if nb < 2:
            return np.nan
        blocks = np.array_split(x, nb)
        u4s = []
        for b in blocks:
            m2 = np.mean(b ** 2)
            m4 = np.mean(b ** 4)
            if m2 > 0:
                u4s.append(1.0 - m4 / (3.0 * m2 ** 2))
        if len(u4s) < 2:
            return np.nan
        return float(np.std(u4s, ddof=1) / np.sqrt(len(u4s)))

    # Cache

    def _key(self, delta, kf, L, method):
        meta = dict(
            L=int(L), J0=self.J0,
            delta=float(delta), kf=float(kf),
            warmup=self._get_warmup(method),
            steps=self._get_steps(method),
            method=method,
            chi_mode=self.chi_mode,
            use_kac=self.use_kac, kac_mode=self.kac_mode,
            T_grid=list(map(float, self.T_grid)),
        )
        h = hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]
        return f"L{L}_d{delta}_kf{kf}_{method[:4]}_{h}"

    def _cache_path(self, delta, kf, L, method):
        return os.path.join(self.cache_dir, self._key(delta, kf, L, method) + ".npz")

    def _save_cache(self, delta, kf, L, method, data):
        np.savez_compressed(self._cache_path(delta, kf, L, method), **data)

    def _load_cache(self, delta, kf, L, method):
        path = self._cache_path(delta, kf, L, method)
        if os.path.exists(path):
            z = np.load(path)
            return {k: z[k] for k in z.files}
        return None

    # Simulation

    def _run_vs_T(self, delta, kf, L, method):
        lat = Lattice(int(L), int(L), float(delta), kf=float(kf), J0=self.J0)
        if self.use_kac:
            self._kac_normalize(lat, mode=self.kac_mode)
        mc = MonteCarlo(lat)

        warmup_steps = self._get_warmup(method)
        prod_steps   = self._get_steps(method)
        N_spins      = int(L) * int(L)

        E_arr, M_arr, Chi_arr, Binder_arr = [], [], [], []
        E_err_arr, M_err_arr, Chi_err_arr, Binder_err_arr = [], [], [], []

        for T in self.T_grid:
            mc.run_loop(warmup_steps, prod_steps, float(T), method=method)

            m_series = np.asarray(mc.acc.magnetization, dtype=float)
            e_series = np.asarray(mc.acc.energy,        dtype=float)

            m2 = np.mean(m_series ** 2)
            m4 = np.mean(m_series ** 4)

            if self.chi_mode == "step_average":
                chi_series = np.asarray(mc.acc.susceptibility, dtype=float)
                chi_val = float(np.mean(chi_series))
                chi_err = self._block_error(chi_series)
            else:
                chi_val = (N_spins / float(T)) * (m2 - np.mean(m_series) ** 2)
                chi_err = self._chi_block_error(m_series, float(T), N_spins)

            binder_val = 1.0 - m4 / (3.0 * m2 ** 2) if m2 > 0 else np.nan

            E_arr.append(np.mean(e_series))
            M_arr.append(np.mean(np.abs(m_series)))
            Chi_arr.append(chi_val)
            Binder_arr.append(binder_val)

            E_err_arr.append(self._block_error(e_series))
            M_err_arr.append(self._block_error(np.abs(m_series)))
            Chi_err_arr.append(chi_err)
            Binder_err_arr.append(self._binder_block_error(m_series))

        return dict(
            T          = self.T_grid.copy(),
            E          = np.array(E_arr),
            M          = np.array(M_arr),
            Chi        = np.array(Chi_arr),
            Binder     = np.array(Binder_arr),
            E_err      = np.array(E_err_arr),
            M_err      = np.array(M_err_arr),
            Chi_err    = np.array(Chi_err_arr),
            Binder_err = np.array(Binder_err_arr),
        )

    # ensure – the central entry point for data

    def ensure(self, delta, kf, L=None, method=None, force=False):
        """Return data for (delta, kf, L, method), computing if necessary.

        L and method default to self.L and self.method when not supplied.
        """
        L_use      = int(L)      if L      is not None else self.L
        method_use = str(method) if method is not None else self.method
        key        = (float(delta), float(kf), L_use, method_use)

        if (not force) and key in self.results:
            return self.results[key]

        if not force:
            cached = self._load_cache(delta, kf, L_use, method_use)
            if cached is not None:
                self.results[key] = cached
                self._update_peak_entry(key, cached["T"], cached["Chi"])
                return cached

        data = self._run_vs_T(delta, kf, L_use, method_use)
        self.results[key] = data
        self._update_peak_entry(key, data["T"], data["Chi"])
        self._save_cache(delta, kf, L_use, method_use, data)

        i_max = int(np.argmax(data["Chi"]))
        nT = len(data["T"])
        if nT > 1 and (i_max == 0 or i_max == nT - 1):
            print(f"[warn] χ peak at boundary: δ={delta}, kF={kf}, "
                  f"L={L_use}, {method_use}, T≈{data['T'][i_max]:.2f}. Refine T_grid.")
        return data

    def _update_peak_entry(self, key, T_arr, chi_arr):
        """Store χ(T) summary: for a single T this is just (T, χ), not a spline peak."""
        T_arr = np.asarray(T_arr, dtype=float).ravel()
        chi_arr = np.asarray(chi_arr, dtype=float).ravel()
        if T_arr.size == 0:
            self.peaks_map[key] = (float("nan"), float("nan"))
        elif T_arr.size == 1:
            self.peaks_map[key] = (float(T_arr[0]), float(chi_arr[0]))
        else:
            self.peaks_map[key] = self._chi_peak(T_arr, chi_arr)

    @staticmethod
    def _chi_peak(T, chi):
        """Peak (T, χ) from a χ(T) curve. Cubic spline needs ≥4 points; fewer T use lower k."""
        T = np.asarray(T, dtype=float).ravel()
        chi = np.asarray(chi, dtype=float).ravel()
        n = T.size
        if n == 0:
            return float("nan"), float("nan")
        if n == 1:
            return float(T[0]), float(chi[0])
        k = min(3, n - 1)
        s = UnivariateSpline(T, chi, s=0, k=k)
        TT = np.linspace(T.min(), T.max(), max(4001, 50 * n))
        y = s(TT)
        i = int(np.argmax(y))
        return float(TT[i]), float(y[i])

    # ------------------------------------------------------------------
    # Bulk runners
    # ------------------------------------------------------------------

    def run_grid(self, subset=None, Ls=None, methods=None):
        """Run all (delta, kf, L, method) combinations and save a peak table."""
        Ls_use      = ([int(x) for x in Ls] if Ls is not None else self.Ls)
        methods_use = (self._norm_methods(methods) if methods is not None
                       else [self.method])
        pairs = (subset or
                 [(d, kf) for d in self.deltas for kf in self.kfs])
        for d, k in pairs:
            for L in Ls_use:
                for m in methods_use:
                    self.ensure(d, k, L=L, method=m)

        peaks = [[d, k, L, tstar, chimax]
                 for (d, k, L, meth), (tstar, chimax) in self.peaks_map.items()]
        if peaks:
            np.savetxt(
                os.path.join(self.outdir, "peaks_delta_kf.txt"),
                np.array(peaks, float),
                header="delta  kF  L  T_peak  chi_max",
            )
        return self.results

    def run_susceptibility_sweep(self, deltas=None, kfs=None,
                                 Ls=None, methods=None, force=False):
        """Run only the required (delta, kf, L, method) combinations.

        Each parameter can be a scalar or array-like; None falls back to
        the defaults stored in the instance.
        """
        deltas_use  = (self._as_list(deltas,  "deltas")  if deltas  is not None
                       else list(self.deltas))
        kfs_use     = (self._as_list(kfs,     "kfs")     if kfs     is not None
                       else list(self.kfs))
        Ls_use      = ([int(x) for x in (Ls if np.isscalar(Ls) else list(Ls))]
                       if Ls is not None else self.Ls)
        if np.isscalar(Ls_use):
            Ls_use = [int(Ls_use)]
        methods_use = (self._norm_methods(methods) if methods is not None
                       else [self.method])

        out = {}
        for method in methods_use:
            for L in Ls_use:
                for delta in deltas_use:
                    for kf in kfs_use:
                        data = self.ensure(delta, kf, L=L, method=method, force=force)
                        out[(float(delta), float(kf), L, method)] = data
        return out

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _color_cycle(self, ax, n):
        self._set_cycle(ax, n)

    def plot_by_kf(self, delta, L=None, method=None, fname=None):
        L_use = int(L) if L is not None else self.L
        m_use = str(method) if method is not None else self.method
        for kf in self.kfs:
            self.ensure(delta, kf, L=L_use, method=m_use)

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        n = len(self.kfs)
        for ax in axs:
            self._set_cycle(ax, n)

        for kf in self.kfs:
            d = self.results[(float(delta), float(kf), L_use, m_use)]
            T = d["T"]
            axs[0].plot(T, d["E"],   "o-", ms=4, label=f"kF={kf:g}")
            axs[1].plot(T, d["M"],   "o-", ms=4, label=f"kF={kf:g}")
            axs[2].plot(T, d["Chi"], "o-", ms=4, label=f"kF={kf:g}")

        axs[0].set(xlabel="T", ylabel="⟨E⟩")
        axs[1].set(xlabel="T", ylabel="⟨|M|⟩")
        axs[2].set(xlabel="T", ylabel="χ")
        for ax in axs:
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        if fname is None:
            fname = f"by_kf_delta{str(delta).replace('.','p')}_L{L_use}_{m_use}.png"
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180); plt.show()
        return path

    def plot_by_delta(self, kf, L=None, method=None, fname=None):
        L_use = int(L) if L is not None else self.L
        m_use = str(method) if method is not None else self.method
        for delta in self.deltas:
            self.ensure(delta, kf, L=L_use, method=m_use)

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        n = len(self.deltas)
        for ax in axs:
            self._set_cycle(ax, n)

        for delta in self.deltas:
            d = self.results[(float(delta), float(kf), L_use, m_use)]
            T = d["T"]
            axs[0].plot(T, d["E"],   "o-", ms=4, label=f"δ={delta:g}")
            axs[1].plot(T, d["M"],   "o-", ms=4, label=f"δ={delta:g}")
            axs[2].plot(T, d["Chi"], "o-", ms=4, label=f"δ={delta:g}")

        axs[0].set(xlabel="T", ylabel="⟨E⟩")
        axs[1].set(xlabel="T", ylabel="⟨|M|⟩")
        axs[2].set(xlabel="T", ylabel="χ")
        for ax in axs:
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        if fname is None:
            fname = f"by_delta_kf{str(kf).replace('.','p')}_L{L_use}_{m_use}.png"
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180); plt.show()
        return path

    def plot_susceptibility_compare(self, deltas=None, kfs=None,
                                    Ls=None, methods=None,
                                    by="kf", fixed_value=None, fname=None):
        """Plot χ(T) for Metropolis vs Wolff (or any subset) with error bars.

        Parameters
        ----------
        by : "kf"    – sweep over kfs at a fixed doping  (fixed_value = delta)
             "delta" – sweep over deltas at a fixed kf    (fixed_value = kf)
        methods : None (use self.method), a single string, or a list/tuple.
        """
        by = str(by).lower()
        if by not in ("kf", "delta"):
            raise ValueError("by must be 'kf' or 'delta'")

        deltas_use  = (self._as_list(deltas, "deltas") if deltas is not None
                       else list(self.deltas))
        kfs_use     = (self._as_list(kfs,    "kfs")    if kfs    is not None
                       else list(self.kfs))
        Ls_use      = ([int(x) for x in (Ls if not np.isscalar(Ls) else [Ls])]
                       if Ls is not None else [self.L])
        methods_use = (self._norm_methods(methods) if methods is not None
                       else [self.method])

        # Determine the fixed axis value
        if by == "kf":
            fixed_delta = float(fixed_value) if fixed_value is not None else deltas_use[0]
        else:
            fixed_kf = float(fixed_value) if fixed_value is not None else kfs_use[0]

        # Run only what is needed
        if by == "kf":
            self.run_susceptibility_sweep(
                deltas=[fixed_delta], kfs=kfs_use, Ls=Ls_use, methods=methods_use)
        else:
            self.run_susceptibility_sweep(
                deltas=deltas_use, kfs=[fixed_kf], Ls=Ls_use, methods=methods_use)

        method_ls  = {m: ls for m, ls in zip(["metropolis", "wolff"], ["-", "--"])}
        method_mk  = {m: mk for m, mk in zip(["metropolis", "wolff"], ["o", "s"])}
        n_curves   = (len(kfs_use) if by == "kf" else len(deltas_use)) * len(Ls_use)

        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        self._set_cycle(ax, n_curves * len(methods_use))

        for method in methods_use:
            ls = method_ls.get(method, "-")
            mk = method_mk.get(method, "o")
            for L in Ls_use:
                if by == "kf":
                    for kf in kfs_use:
                        d = self.results[(fixed_delta, float(kf), L, method)]
                        label = (f"{method}, kF={kf:g}" +
                                 (f", L={L}" if len(Ls_use) > 1 else ""))
                        ax.errorbar(d["T"], d["Chi"], yerr=d.get("Chi_err"),
                                    ls=ls, marker=mk, ms=3.5, capsize=2.5,
                                    label=label)
                else:
                    for delta in deltas_use:
                        d = self.results[(float(delta), fixed_kf, L, method)]
                        label = (f"{method}, δ={delta:g}" +
                                 (f", L={L}" if len(Ls_use) > 1 else ""))
                        ax.errorbar(d["T"], d["Chi"], yerr=d.get("Chi_err"),
                                    ls=ls, marker=mk, ms=3.5, capsize=2.5,
                                    label=label)

        ax.set_xlabel("T")
        ax.set_ylabel("χ")
        title = (f"Susceptibility vs T  |  δ={fixed_delta:g}"
                 if by == "kf"
                 else f"Susceptibility vs T  |  kF={fixed_kf:g}")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=max(1, len(methods_use)))
        plt.tight_layout()

        if fname is None:
            if by == "kf":
                fname = f"sus_compare_by_kf_delta{str(fixed_delta).replace('.','p')}.png"
            else:
                fname = f"sus_compare_by_delta_kf{str(fixed_kf).replace('.','p')}.png"
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180); plt.show()
        return path

    def plot_susceptibility_fss(self, delta, kf,
                                Ls=None, methods=None, fname=None):
        """χ(T) at fixed (δ, kF) for several L — complements Binder FSS plots."""
        Ls_use      = ([int(x) for x in Ls] if Ls is not None else self.Ls)
        methods_use = (self._norm_methods(methods) if methods is not None
                       else [self.method])

        self.run_susceptibility_sweep(
            deltas=[delta], kfs=[kf], Ls=Ls_use, methods=methods_use)

        method_ls = {m: ls for m, ls in zip(["metropolis", "wolff"], ["-", "--"])}
        method_mk = {m: mk for m, mk in zip(["metropolis", "wolff"], ["o", "s"])}

        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        n_total = len(Ls_use) * len(methods_use)
        self._set_cycle(ax, n_total)

        one_method = len(methods_use) == 1
        for method in methods_use:
            ls = method_ls.get(method, "-")
            mk = method_mk.get(method, "o")
            for L in Ls_use:
                d = self.results[(float(delta), float(kf), L, method)]
                lbl = f"L={L}" if one_method else f"{method}, L={L}"
                ax.errorbar(
                    d["T"], d["Chi"], yerr=d.get("Chi_err"),
                    ls=ls, marker=mk, ms=3.5, capsize=2.5,
                    label=lbl,
                )

        ax.set_xlabel("T")
        ax.set_ylabel("χ")
        ax.set_title(
            f"Susceptibility  |  δ={delta:g}, kF={kf:g}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        if fname is None:
            fname = (f"chi_fss_d{str(delta).replace('.','p')}"
                     f"_kf{str(kf).replace('.','p')}.png")
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180); plt.show()
        return path

    def plot_binder_crossing(self, delta, kf,
                             Ls=None, methods=None, fname=None):
        """Plot the Binder cumulant U₄(T) for multiple L and methods.

        The crossing of curves for different L gives an estimate of Tc.
        """
        Ls_use      = ([int(x) for x in Ls] if Ls is not None else self.Ls)
        methods_use = (self._norm_methods(methods) if methods is not None
                       else [self.method])

        self.run_susceptibility_sweep(
            deltas=[delta], kfs=[kf], Ls=Ls_use, methods=methods_use)

        method_ls = {m: ls for m, ls in zip(["metropolis", "wolff"], ["-", "--"])}
        method_mk = {m: mk for m, mk in zip(["metropolis", "wolff"], ["o", "s"])}

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        n_total = len(Ls_use) * len(methods_use)
        self._set_cycle(ax, n_total)

        one_method = len(methods_use) == 1
        for method in methods_use:
            ls = method_ls.get(method, "-")
            mk = method_mk.get(method, "o")
            for L in Ls_use:
                d = self.results[(float(delta), float(kf), L, method)]
                lbl = f"L={L}" if one_method else f"{method}, L={L}"
                ax.errorbar(d["T"], d["Binder"], yerr=d.get("Binder_err"),
                            ls=ls, marker=mk, ms=4, capsize=2.5,
                            label=lbl)

        ax.axhline(2 / 3, color="k", ls=":", lw=1, label="2/3 (Ising FM limit)")
        ax.set_xlabel("T")
        ax.set_ylabel("U₄  =  1 − ⟨M⁴⟩ / (3⟨M²⟩²)")
        ax.set_title(f"Binder cumulant  |  δ={delta:g}, kF={kf:g}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()

        if fname is None:
            fname = (f"binder_d{str(delta).replace('.','p')}"
                     f"_kf{str(kf).replace('.','p')}.png")
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180); plt.show()
        return path

    def plot_chi_phase_diagram_fixed_T(
        self,
        T_fixed,
        deltas=None,
        kfs=None,
        L=None,
        method=None,
        cmap="magma",
        fname=None,
        force=False,
    ):
        """Heatmap χ(δ, kF) at one temperature: one simulation per pixel, one χ value each.

        Sets ``T_grid`` internally to ``[T_fixed]`` (no T-sweep, no peak search).
        Returns ``(path, Z)`` where ``Z[i,j] = χ`` at ``(deltas[i], kfs[j])``.
        """
        T_fixed = float(T_fixed)
        L_use = int(L) if L is not None else self.L
        m_use = str(method) if method is not None else self.method

        deltas_use = (
            self._as_list(deltas, "deltas") if deltas is not None else list(self.deltas)
        )
        kfs_use = self._as_list(kfs, "kfs") if kfs is not None else list(self.kfs)

        old_T = self.T_grid
        self.T_grid = np.asarray([T_fixed], dtype=float)
        try:
            self.run_susceptibility_sweep(
                deltas=deltas_use,
                kfs=kfs_use,
                Ls=[L_use],
                methods=[m_use],
                force=force,
            )
            Z = np.full((len(deltas_use), len(kfs_use)), np.nan, dtype=float)
            for i, d in enumerate(deltas_use):
                for j, kf in enumerate(kfs_use):
                    data = self.results[(float(d), float(kf), L_use, m_use)]
                    Z[i, j] = float(data["Chi"][0])
        finally:
            self.T_grid = old_T

        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        im = ax.pcolormesh(
            np.asarray(kfs_use, float),
            np.asarray(deltas_use, float),
            Z,
            shading="nearest",
            cmap=cmap,
        )
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(f"χ ({m_use})")
        ax.set_xlabel("kF")
        ax.set_ylabel("δ")
        ax.set_title(f"χ(δ, kF) at T={T_fixed:g}, L={L_use}")
        plt.tight_layout()

        if fname is None:
            fname = (
                f"chi_map_T{str(T_fixed).replace('.', 'p')}_L{L_use}_{m_use}.png"
            )
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180)
        plt.show()
        return path, Z

    def plot_peak_map(self, what="T_peak", L=None, method=None, fname=None):
        L_use = int(L) if L is not None else self.L
        m_use = str(method) if method is not None else self.method
        self.run_grid(Ls=[L_use], methods=[m_use])

        Z = np.full((len(self.deltas), len(self.kfs)), np.nan)
        for i, d in enumerate(self.deltas):
            for j, k in enumerate(self.kfs):
                key = (float(d), float(k), L_use, m_use)
                if key not in self.peaks_map:
                    continue
                tstar, chimax = self.peaks_map[key]
                Z[i, j] = tstar if what == "T_peak" else chimax

        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        im = ax.pcolormesh(self.kfs, self.deltas, Z, shading="nearest", cmap=self.cmap)
        cb = plt.colorbar(im)
        cb.set_label("T_peak" if what == "T_peak" else "χ_max")
        ax.set_xlabel("kF"); ax.set_ylabel("δ")
        plt.tight_layout()

        if fname is None:
            tag = "Tpeak" if what == "T_peak" else "Chimax"
            fname = f"map_{tag}_L{L_use}_{m_use}.png"
        path = os.path.join(self.outdir, fname)
        plt.savefig(path, dpi=180); plt.show()
        return path
