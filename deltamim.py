"""Python translation of deltamim.m.

Estimate Borgonovo's delta moment-independent sensitivity measure from samples.

This is a NumPy/SciPy translation of Elmar Plischke's MATLAB `deltamim` code.
The main public function is `deltamim(x, y, opts=None, gfx=None)`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Callable, Iterable, Optional, Union, Any

import numpy as np

try:
    from scipy.optimize import brentq
    from scipy.special import erfinv, gamma
    from scipy.fftpack import dct, idct
except ImportError as exc:  # pragma: no cover
    raise ImportError("deltamim.py requires scipy in addition to numpy") from exc

ArrayLike = Union[np.ndarray, Iterable[float]]


def _default_partition_size(n: int) -> int:
    return min(int(np.ceil(n ** (2.0 / (7.0 + np.tanh((1500.0 - n) / 500.0))))), 48)


@dataclass
class DeltaMIMOptions:
    PartitionSize: Optional[int] = None
    QuadraturePoints: int = 110
    KSLevel: Union[float, Iterable[float]] = 0.95
    ZeroCrossing: str = "on"
    ParameterNames: Optional[list[str]] = None
    KDEstimator: str = "cheap"
    KDWidth: Union[str, float, np.ndarray, list[float]] = "auto"
    Complement: str = "off"
    SwitchXY: str = "off"  # retained for compatibility; unused in original code path
    OutputTrafo: Union[str, Callable[[np.ndarray], np.ndarray]] = "off"
    PlotCols: Optional[int] = None
    ShowOpts: str = "off"
    ShowSep: str = "on"
    KDShape: str = "epanechnikov"
    DDD: bool = False
    PowerLoss: float = 0.0
    PowerLossScale: float = 0.0
    PartitionSplit: Optional[ArrayLike] = None
    GfxTitle: str = ""

    @classmethod
    def with_defaults(cls, n: int, k: int) -> "DeltaMIMOptions":
        return cls(PartitionSize=_default_partition_size(n), PlotCols=int(np.ceil(np.sqrt(k))))


def deltamim(
    x: ArrayLike,
    y: ArrayLike,
    opts: Optional[Union[int, dict[str, Any], DeltaMIMOptions]] = None,
    gfx: Optional[str] = None,
):
    """Estimate Borgonovo's delta moment-independent measure.

    Parameters
    ----------
    x : array_like, shape (n_samples, n_inputs)
        Input samples. A one-dimensional input is accepted and reshaped to (n, 1).
    y : array_like, shape (n_samples,)
        Output samples.
    opts : int, dict, DeltaMIMOptions, optional
        If an int is supplied, it is used as the partition size. A dict or
        DeltaMIMOptions instance can override MATLAB-compatible option names.
    gfx : str, optional
        If provided, plot the unconditional and conditional density estimates.

    Returns
    -------
    delta : ndarray, shape (n_inputs,)
        Moment-independent delta measures.
    Si : ndarray, shape (n_inputs,)
        First-order variance-based effects computed as in the MATLAB code.
    acceptL : ndarray, shape (3, n_inputs)
        Acceptance levels for KS-style filtering. The first two rows correspond
        to the simple and full thresholds used by the original function; the
        third is the minimum full threshold.
    Seps : ndarray, shape (n_inputs, PartitionSize)
        Conditional separations per partition.

    Notes
    -----
    MATLAB accepts a variable number of outputs. Python returns all four values.
    The default KDE estimator is the translated "cheap" kernel estimator.
    "stats" is approximated by the same custom kernel estimator because MATLAB's
    `ksdensity` has no exact NumPy equivalent.
    "diffusion" uses the translated Botev diffusion KDE and requires SciPy DCT.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("x must be a 1D or 2D array")

    n, k = x.shape
    if y.size != n:
        raise ValueError("x and y must contain the same number of samples")

    o = DeltaMIMOptions.with_defaults(n, k)
    if opts is not None:
        if isinstance(opts, (int, np.integer)):
            o.PartitionSize = int(opts)
        elif isinstance(opts, DeltaMIMOptions):
            base = asdict(o)
            supplied = asdict(opts)
            for key, value in supplied.items():
                if value is not None:
                    base[key] = value
            o = DeltaMIMOptions(**base)
        elif isinstance(opts, dict):
            valid = {f.name for f in fields(DeltaMIMOptions)}
            for key, value in opts.items():
                if key in valid:
                    setattr(o, key, value)
                else:
                    raise ValueError(f"Unknown option: {key}")
        else:
            raise TypeError("opts must be None, int, dict, or DeltaMIMOptions")

    if gfx is None:
        gfx = o.GfxTitle or None

    if o.ParameterNames is None:
        o.ParameterNames = [f"x_{{{i + 1}}}" for i in range(k)] + ["y"]
    elif len(o.ParameterNames) == k:
        o.ParameterNames = list(o.ParameterNames) + ["y"]

    if o.PartitionSplit is not None:
        ps = np.asarray(o.PartitionSplit, dtype=float).reshape(-1)
        o.PartitionSize = ps.size + 1
    else:
        ps = None

    if o.KDEstimator.lower() == "diffusion":
        o.QuadraturePoints = int(2 ** np.ceil(np.log2(o.QuadraturePoints)))

    kernel = _make_kernel(o.KDShape)

    kscrit = _ks_critical_values(o.KSLevel, k, o.Complement)
    kstest_full = False
    kslevel_scalar = np.asarray(o.KSLevel).reshape(-1)[0] if np.asarray(o.KSLevel).size else 0
    if kslevel_scalar != 0 and kslevel_scalar < 0 and o.Complement.lower() != "on":
        kstest_full = True

    if o.ShowOpts.lower() != "off":
        print(o)

    if callable(o.OutputTrafo):
        y = np.asarray(o.OutputTrafo(y), dtype=float).reshape(-1)
        o.OutputTrafo = "off"

    indx = np.argsort(y, kind="mergesort")
    ys = y[indx].copy()
    yy, ysupp, ysupp2, y, ys = _transform_output(y, ys, indx, str(o.OutputTrafo).lower(), o.QuadraturePoints)
    yy_backup = yy.copy()

    xr = np.zeros((n, k), dtype=float)
    consticator = np.zeros(k, dtype=bool)
    for i in range(k):
        indxx = np.argsort(x[:, i], kind="mergesort")
        xx_sorted = x[indxx, i]
        consticator[i] = xx_sorted[-1] == xx_sorted[0]
        ranks_sorted = empcdf(xx_sorted)
        xx_ranks = np.empty(n, dtype=float)
        xx_ranks[indxx] = ranks_sorted
        xr[:, i] = xx_ranks[indx]

    alfa: Union[float, np.ndarray]
    if isinstance(o.KDWidth, str):
        alfa = 0.0
    else:
        alfa = np.asarray(o.KDWidth, dtype=float)
        if alfa.size == 1:
            alfa = float(alfa.reshape(-1)[0])

    f1, alfa = _density_estimate(ys, yy, alfa, o, kernel, ysupp, ysupp2, return_bandwidth=True)

    if o.PowerLoss != 0:
        gamma_power = o.PowerLoss - 1
        efygm1 = np.trapz(f1 ** (gamma_power + 1), yy)
        if o.PowerLossScale == 0:
            o.PowerLossScale = 2 / gamma_power / (gamma_power + 1)
    else:
        gamma_power = 0.0
        efygm1 = 0.0

    M = int(o.PartitionSize)
    segs = np.linspace(0, 1, M + 1) if ps is None else np.r_[0.0, ps, 1.0]

    do_plot = gfx is not None
    if do_plot:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("jet", M)
        layoutrows = int(np.ceil(k / o.PlotCols))
        if o.ShowSep.lower() == "on":
            layoutrows += 1
        if o.ShowSep.lower() == "only":
            layoutrows = 0
        plt.figure()
    else:
        plt = None
        cmap = None
        layoutrows = 0

    delta = np.zeros(k, dtype=float)
    Si = np.zeros(k, dtype=float)
    acceptL = np.zeros((3, k), dtype=float)
    Seps = np.zeros((k, M), dtype=float)
    ey = np.mean(y)
    vy = np.var(y, ddof=1)

    for i in range(k):
        if consticator[i]:
            continue

        if do_plot and layoutrows > 0 and o.ShowSep.lower() != "only":
            ax = plt.subplot(layoutrows, int(o.PlotCols), i + 1) if k > 1 else plt.subplot(layoutrows, 1, i + 1)
            ax.plot(yy_backup, f1, "k", linewidth=2)
            ax.set_ylabel("Density function")
            ax.set_xlabel(f"{o.ParameterNames[k]} given {o.ParameterNames[i]}")
            ax.set_title(gfx)

        Sr = np.zeros(M, dtype=float)
        nr = np.zeros(M, dtype=float)
        Vyc = np.zeros(M, dtype=float)
        Kr = np.zeros(M, dtype=float)

        for m in range(M):
            if o.Complement.lower() != "on":
                mask = (xr[:, i] >= segs[m]) & (xr[:, i] < segs[m + 1])
            else:
                mask = (xr[:, i] < segs[m]) | (xr[:, i] >= segs[m + 1])
            yx = ys[mask]
            nx = yx.size
            nr[m] = nx
            if nx == 0:
                continue

            if o.Complement.lower() != "on":
                Vyc[m] = nx * (np.mean(yx) - ey) ** 2
            else:
                Vyc[m] = nx ** 2 / (n - nx) * (np.mean(yx) - ey) ** 2 if n != nx else 0

            if isinstance(o.KDWidth, str) and o.KDWidth.lower() == "auto":
                alfa_use = 0.0
            else:
                alfa_use = alfa
            f2 = _density_estimate(yx, yy_backup, alfa_use, o, kernel, ysupp, ysupp2, return_bandwidth=False)

            if do_plot and layoutrows > 0 and o.ShowSep.lower() != "only":
                ax = plt.subplot(layoutrows, int(o.PlotCols), i + 1) if k > 1 else plt.subplot(layoutrows, 1, i + 1)
                if o.Complement.lower() != "on":
                    ax.plot(yy_backup, f2, color=cmap(m))
                else:
                    comp_f = np.maximum((n * f1 - nx * f2) / max(n - nx, 1), 0)
                    ax.plot(yy_backup, comp_f, color=cmap(m))

            if o.PowerLoss != 0:
                Sr[m] = o.PowerLossScale * (np.trapz(f2 ** (gamma_power + 1), yy_backup) - efygm1)
                continue

            fff = f1 - f2
            yy_cur = yy_backup.copy()
            deltaoffset = 0.0
            zc = o.ZeroCrossing.lower()

            if zc == "on":
                ff = np.abs(fff)
                deltafactor = 0.5
                crossings = np.flatnonzero(fff[:-1] * fff[1:] < 0)
                if crossings.size:
                    yz = yy_cur[crossings] + (yy_cur[crossings + 1] - yy_cur[crossings]) * fff[crossings] / (fff[crossings] - fff[crossings + 1])
                    for idx_insert, zval in sorted(zip(crossings, yz), reverse=True):
                        yy_cur = np.insert(yy_cur, idx_insert + 1, zval)
                        ff = np.insert(ff, idx_insert + 1, 0.0)
                        fff = np.insert(fff, idx_insert + 1, 0.0)
            elif zc == "positive":
                ff = np.maximum(fff, 0)
                deltafactor = 1.0
            elif zc == "negative":
                ff = -np.minimum(fff, 0)
                deltafactor = 1.0
            elif zc == "off":
                ff = np.abs(fff)
                deltafactor = 0.5
            elif zc == "test":
                ff = np.r_[np.maximum(fff, 0), -np.minimum(fff, 0)]
                yy_cur = np.r_[yy_cur, yy_cur - yy_cur[0] + yy_cur[-1]]
                deltafactor = 0.5
            elif zc == "min":
                ff = np.minimum(f1, f2)
                deltafactor = -1.0
                deltaoffset = 2.0
            else:
                ff = np.abs(fff)
                deltafactor = 0.5

            nx_for_test = nx
            if o.Complement.lower() == "on":
                ff = nx / (n - nx) * ff if n != nx else np.zeros_like(ff)
                nr[m] = n - nx
                nx_for_test = n - nx

            S = deltaoffset + 2 * deltafactor * np.trapz(ff, yy_cur)

            if kscrit[i] and nx_for_test > 0:
                stat = np.max(np.abs(_cumtrapz(fff, yy_cur))) if kstest_full else S * 0.5
                threshold = np.sqrt(1 / n + 1 / nx_for_test) * kscrit[i]
                if stat < threshold:
                    Sr[m] = 0.0
                    Vyc[m] = 0.0
                else:
                    Sr[m] = S
            else:
                Sr[m] = S

            Kr[m] = np.max(np.abs(_cumtrapz(fff, yy_cur)))

        valid = nr > 0
        if np.any(valid):
            denom = np.sqrt(1 / n + 1 / nr[valid])
            thres1 = np.max(0.5 * Sr[valid] / denom)
            thres2 = np.max(Kr[valid] / denom)
            thres3 = np.min(Kr[valid] / denom)
            if thres1 > 0:
                acceptL[0, i] = _kolmog(thres1, 0)
            if thres2 > 0:
                acceptL[1, i] = _kolmog(thres2, 0)
            if thres3 > 0:
                acceptL[2, i] = _kolmog(thres3, 0)

        delta[i] = 0.5 * np.dot(Sr, nr) / n
        Seps[i, :] = Sr
        Si[i] = np.sum(Vyc) / vy / (n - 1) if vy > 0 else np.nan

        if do_plot and o.ShowSep.lower() != "off":
            sep_ax = plt.subplot(layoutrows, 1, layoutrows) if layoutrows > 0 else plt.subplot(1, 1, 1)
            xvals = np.cumsum(nr) / n - 1 / (2 * M)
            sep_ax.plot(xvals, Sr, label=o.ParameterNames[i])

    if do_plot:
        if o.ShowSep.lower() != "off":
            sep_ax = plt.subplot(layoutrows, 1, layoutrows) if layoutrows > 0 else plt.subplot(1, 1, 1)
            sep_ax.set_ylabel("S_r")
            sep_ax.set_xlabel("Empirical cdf of inputs")
            sep_ax.set_title("Separation of Conditional Densities")
            if np.any(kscrit):
                q = M
                sep_ax.plot([0, 1], [kscrit[-1] * np.sqrt((q + 1) / n)] * 2, "k:", label="cut-off")
            sep_ax.legend()
        plt.tight_layout()

    return delta, Si, acceptL, Seps


def _make_kernel(shape: str) -> Callable[[np.ndarray], np.ndarray]:
    shape = shape.lower()
    if shape == "normal":
        return lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    if shape == "triangle":
        return lambda x: np.maximum(1 - np.abs(x / np.sqrt(6)), 0) / np.sqrt(6)
    if shape in {"epanechnikov", "parabolic"}:
        return lambda x: 3 / (4 * np.sqrt(5)) * np.maximum(1 - x**2 / 5, 0)
    if shape in {"box", "uniform"}:
        return lambda x: (np.abs(x / np.sqrt(3)) < 1).astype(float) / (2 * np.sqrt(3))
    if shape in {"biweight", "biquadratic"}:
        return lambda x: 15 / (16 * np.sqrt(7)) * np.maximum(1 - x**2 / 7, 0) ** 2
    raise ValueError(f"Unsupported kernel: {shape}")


def _kolmog(x: float, y: float) -> float:
    if x < 4:
        j = np.arange(1, 36, 2, dtype=float)
        return np.sqrt(2 * np.pi) / x * np.sum(np.exp(-(j**2) * np.pi**2 / (8 * x**2))) - y
    return 1.0 - y


def _ks_critical_values(ks_level, k: int, complement: str) -> np.ndarray:
    levels = np.asarray(ks_level, dtype=float).reshape(-1)
    if levels.size == 0 or np.all(levels == 0):
        return np.zeros(k)
    if complement.lower() == "on":
        levels = np.abs(levels)
    levels = np.abs(levels)
    vals = np.array([brentq(lambda z, lev=lev: _kolmog(z, lev), 0.001, 2.0) for lev in levels])
    if vals.size == 1:
        return np.full(k, vals.item())
    if vals.size != k:
        raise ValueError("KSLevel must be scalar or have one value per input column")
    return vals


def _transform_output(y, ys, indx, output_trafo, qpoints):
    if output_trafo in {"off", "none"}:
        ymaxmin = np.array([ys[0], ys[-1]])
        rangey = ymaxmin[1] - ymaxmin[0]
        ysupp = ymaxmin + 0.06 * rangey * np.array([-1, 1])
        yy = np.linspace(ymaxmin[0] - 0.04 * rangey, ymaxmin[1] + 0.04 * rangey, qpoints)
    elif output_trafo == "cdf":
        ys = empcdf(ys)
        y = y.copy()
        y[indx] = ys
        ysupp = np.array([-0.06, 1.06])
        yy = np.linspace(-0.04, 1.04, qpoints)
    elif output_trafo == "normal":
        ncdf = -np.sqrt(2) * erfinv(1 - 2 * empcdf(ys))
        y = y.copy()
        y[indx] = ncdf
        ys = ncdf
        ymaxmin = np.array([ys[0], ys[-1]])
        rangey = ymaxmin[1] - ymaxmin[0]
        ysupp = ymaxmin + 0.06 * rangey * np.array([-1, 1])
        yy = np.linspace(ymaxmin[0] - 0.04 * rangey, ymaxmin[1] + 0.04 * rangey, qpoints)
    elif output_trafo == "interpol":
        xp = (2 * np.arange(1, ys.size + 1) - 1) / (2 * ys.size)
        yy = np.interp(np.linspace(0, 1, qpoints), xp, ys)
        ysupp = np.array([yy[0], yy[-1]])
    elif output_trafo == "cdf-tight":
        ys = empcdf(ys)
        y = y.copy()
        y[indx] = ys
        ysupp = np.array([-0.02, 1.02])
        yy = np.linspace(-0.01, 1.01, qpoints)
    elif output_trafo == "cdf-loose":
        ys = empcdf(ys)
        y = y.copy()
        y[indx] = ys
        ysupp = np.array([-0.1, 1.1])
        yy = np.linspace(-0.08, 1.08, qpoints)
    else:
        raise ValueError(f"Unsupported output transformation: {output_trafo}")
    return yy, ysupp, yy[[0, -1]], y, ys


def _density_estimate(y, yy, alfa, opts, kernel, ysupp, ysupp2, return_bandwidth=False):
    estimator = opts.KDEstimator.lower()
    if estimator in {"cheap", "stats"}:
        est, h = kdest(y, yy, alfa, kernel)
    elif estimator == "pilot":
        est, h = kdepilot(y, yy, alfa, kernel)
    elif estimator == "diffusion":
        if _is_zero_bandwidth(alfa):
            h, est, _ = kde_diffusion(y, opts.QuadraturePoints, ysupp2[0], ysupp2[1])
        else:
            bw = float(np.asarray(alfa).reshape(-1)[0])
            h, est, _ = kde_diffusion(y, opts.QuadraturePoints, ysupp2[0], ysupp2[1], bw)
        est = np.maximum(est, 0)
    elif estimator == "hist":
        counts, _ = np.histogram(y, bins=np.r_[yy, yy[-1] + (yy[-1] - yy[-2])])
        est = counts.astype(float)
        area = np.trapz(est, yy)
        est = est / area if area else est
        h = alfa
    elif estimator == "nearestneighbor":
        kval = None if _is_zero_bandwidth(alfa) else int(np.asarray(alfa).reshape(-1)[-1])
        est = densNN(np.asarray(y).reshape(-1, 1), np.asarray(yy).reshape(-1, 1), kval)
        area = np.trapz(est, yy)
        est = est / area if area else est
        h = alfa
    else:
        raise ValueError(f"Unknown kernel density estimator: {opts.KDEstimator}")
    if return_bandwidth:
        return est, h
    return est


def _is_zero_bandwidth(h) -> bool:
    arr = np.asarray(h)
    return arr.size == 0 or np.all(arr == 0)


def kdest(y, z, h=0.0, kernel: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """Bowman/Azzalini-style one-dimensional kernel density estimator."""
    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    if kernel is None:
        kernel = _make_kernel("epanechnikov")
    if _is_zero_bandwidth(h):
        med = np.median(y)
        s = min(np.std(y, ddof=1), np.median(np.abs(med - y)) / 0.675)
        h = s / (((3 * y.size) / 4) ** (1 / 5)) if s > 0 else np.finfo(float).eps
    else:
        h = float(np.asarray(h).reshape(-1)[0])
    est = np.mean(kernel((z[:, None] - y[None, :]) / h), axis=1) / h
    return est, h


def kdepilot(y, z, h=0.0, kernel: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """Two-step pilot kernel density estimator."""
    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    if kernel is None:
        kernel = _make_kernel("epanechnikov")
    if _is_zero_bandwidth(h):
        med = np.median(y)
        s = min(np.std(y, ddof=1), np.median(np.abs(med - y)) / 0.675)
        h = s / (((3 * y.size) / 4) ** (1 / 5)) if s > 0 else np.finfo(float).eps
    else:
        h = float(np.asarray(h).reshape(-1)[0])
    h0 = 1.5 * h
    alfa = np.mean(kernel((y[:, None] - y[None, :]) / h0), axis=1) / h0
    est0 = np.mean(kernel((z[:, None] - y[None, :]) / h0), axis=1) / h0
    weights = np.divide(1.0, alfa, out=np.zeros_like(alfa), where=alfa != 0)
    est = (kernel((z[:, None] - y[None, :]) / h) @ weights) / h / y.size * est0
    return est, h


def empcdf(xs):
    """Empirical CDF ranks for sorted data, using mid-ranks for ties."""
    xs = np.asarray(xs, dtype=float).reshape(-1)
    n = xs.size
    if n == 0:
        return np.array([])
    ranks = np.arange(1, n + 1, dtype=float)
    start = 0
    while start < n:
        end = start + 1
        while end < n and xs[end] == xs[start]:
            end += 1
        if end - start > 1:
            # MATLAB code assigns run + len/2 using 1-based positions.
            ranks[start:end] = (start + 1) + (end - start) / 2
        start = end
    return (ranks - 0.5) / n


def kde_diffusion(data, n=2**14, MIN=None, MAX=None, bandwidth_in=None):
    """Botev diffusion KDE translation used by the MATLAB code."""
    data = np.asarray(data, dtype=float).reshape(-1)
    n = int(2 ** np.ceil(np.log2(n)))
    if MIN is None or MAX is None:
        minimum = np.min(data)
        maximum = np.max(data)
        data_range = maximum - minimum
        MIN = minimum - data_range / 10
        MAX = maximum + data_range / 10
    R = MAX - MIN
    dx = R / (n - 1)
    xmesh = MIN + np.arange(n) * dx
    N = data.size
    counts, _ = np.histogram(data, bins=np.r_[xmesh, xmesh[-1] + dx])
    initial_data = counts / N

    a = dct(initial_data, type=2, norm=None)
    a = a * np.sqrt(2 * n)
    a[0] = a[0] / np.sqrt(2)

    if bandwidth_in is None:
        I = np.arange(1, n, dtype=float) ** 2
        a2 = (a[1:] / 2) ** 2
        try:
            t_star = brentq(lambda t: _fixed_point(t, N, I, a2), 0.0, 0.1)
        except ValueError:
            t_star = 0.01
    else:
        t_star = (bandwidth_in / R) ** 2

    a_t = a * np.exp(-(np.arange(n, dtype=float) ** 2) * np.pi**2 * t_star / 2)
    a_t = a_t / np.sqrt(2 * n)
    a_t[0] = a_t[0] * np.sqrt(2)
    density = idct(a_t, type=2, norm=None) / n / R
    bandwidth = np.sqrt(t_star) * R
    return bandwidth, density, xmesh


def _fixed_point(t, N, I, a2):
    ell = 7
    f = 2 * np.pi ** (2 * ell) * np.sum(I**ell * a2 * np.exp(-I * np.pi**2 * t))
    for s in range(ell - 1, 1, -1):
        K0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 0.5)) / 3
        time = (2 * const * K0 / N / f) ** (2 / (3 + 2 * s))
        f = 2 * np.pi ** (2 * s) * np.sum(I**s * a2 * np.exp(-I * np.pi**2 * time))
    return t - (2 * N * np.sqrt(np.pi) * f) ** (-2 / 5)


def densNN(xx, xs, k=None):
    """k-th nearest-neighbor density estimator."""
    xx = np.asarray(xx, dtype=float)
    xs = np.asarray(xs, dtype=float)
    if xx.ndim == 1:
        xx = xx.reshape(-1, 1)
    if xs.ndim == 1:
        xs = xs.reshape(-1, 1)
    n, d = xx.shape
    if xs.shape[1] != d:
        raise ValueError("densNN: dimension mismatch")
    if k is None:
        k = int(np.ceil(np.sqrt(n)))
    if k > n:
        raise ValueError("densNN: out of samples")
    dist2 = np.sum(xx * xx, axis=1)[:, None] + np.sum(xs * xs, axis=1)[None, :] - 2 * xx @ xs.T
    dist2 = np.maximum(dist2, 0)
    sorted_dist2 = np.sort(dist2, axis=0)
    Vd = np.pi ** (d / 2) / gamma(d / 2 + 1)
    return k / (n * Vd * sorted_dist2[k - 1, :] ** (d / 2))


def _cumtrapz(f, x):
    f = np.asarray(f, dtype=float)
    x = np.asarray(x, dtype=float)
    if f.size < 2:
        return np.zeros_like(f)
    increments = 0.5 * (f[:-1] + f[1:]) * np.diff(x)
    return np.r_[0.0, np.cumsum(increments)]


__all__ = [
    "DeltaMIMOptions",
    "deltamim",
    "empcdf",
    "kdest",
    "kdepilot",
    "kde_diffusion",
    "densNN",
]
