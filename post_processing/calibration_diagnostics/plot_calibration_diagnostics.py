#!/usr/bin/env python3
"""
Pr_t Calibration Diagnostics (Mach 14 flat plate, cold wall).

Purpose
-------
Three independent checks that test whether the calibrated turbulent Prandtl
number (Pr_t = 0.566) is a *physical* correction or merely a fudge factor that
hides a momentum-model error. Motivated by the observation that the RANS and DNS
velocity profiles do not overlap in the `u/U_inf` vs `y/delta` "Do No Harm" plot.

  Check 1 - Van Driest u_vd+ vs y+  (momentum field)
      If the RANS velocity collapses onto the DNS / log-law in wall units, the
      momentum field is sound and the y/delta gap is a normalization artifact
      (different station, pressure, Re and delta definition), not a model defect.

  Check 2 - T(u) shape vs level     (calibration space)
      Temperature is a self-similar function of velocity (Crocco-Busemann /
      strong Reynolds analogy), so T(u) is station-invariant once normalized.
      If RANS reproduces the DNS T(u) *shape* and Pr_t only shifts the *level*,
      then Pr_t is the correct single knob.

  Check 3 - Pr_t(RANS) vs Pr_t(DNS) (physical recovery)
      The DNS file carries a measured Pr_t profile (column 27). If the
      RANS-optimal Pr_t lands near the DNS boundary-layer average, the
      calibration recovered a physical quantity rather than an arbitrary fit.

This script is fully self-contained: it does not import or modify any existing
pipeline code. All inputs are read from data already on disk (no CFD reruns).

Usage
-----
  cd post_processing/calibration_diagnostics
  python plot_calibration_diagnostics.py
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Force non-interactive backend before importing pyplot (WSL convention).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================================
#                              CONFIGURATION
# ==========================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"

LOG_FILE = RESULTS_ROOT / "optimization_log.csv"
DNS_STAT_FILE = DATA_DIR / "dns_database" / "M14Tw018_Stat.dat"

# Mach 14 freestream (unified DNS-consistent M14Tw018: Mach 13.68, derived Re).
U_INF = 1882.1   # [m/s]
T_INF = 47.1     # [K]

# Profile extraction location (matches su2_interface.SU2Interface.X_STATION).
X_STATION = 1.5
X_TOLERANCE = 0.005

# Default Pr_t (Spalart-Allmaras hard-coded value).
PR_T_DEFAULT = 0.9

# Boundary-layer mask for self-similar / DNS-Pr_t comparisons.
U_MASK_LO, U_MASK_HI = 0.1, 0.9

# Figure color convention (repo standard).
C_DNS = "k"          # black hollow circles
C_STANDARD = "r"     # red dashed  -> Pr_t = 0.9
C_CALIBRATED = "b"   # blue solid  -> Pr_t = 0.566

# DNS column indices (1-indexed in the file header; converted to 0-indexed).
# variables = z, zod, zplus, zstar, u_uinf, p_pinf, t_tinf, ..., Uvd(25), ..., Prt(27)
DNS_COL = {"zod": 1, "zplus": 2, "u_uinf": 4, "t_tinf": 6, "Uvd": 24, "Prt": 26}

OUTPUT_DIR = SCRIPT_DIR


# ==========================================================================
#                              DATA LOADING
# ==========================================================================
def detect_optimal_pr() -> tuple[float, float]:
    """Return (optimal Pr_t, its RMSE) from the optimization log (min RMSE)."""
    log = pd.read_csv(LOG_FILE)
    best = log.loc[log["RMSE"].idxmin()]
    return float(best["Pr_t"]), float(best["RMSE"])


def baseline_rmse() -> float:
    """Return the Pr_t = 0.9 baseline RMSE (iteration 0 of the log)."""
    log = pd.read_csv(LOG_FILE)
    row = log.loc[(log["Pr_t"] - PR_T_DEFAULT).abs().idxmin()]
    return float(row["RMSE"])


def load_su2_profile(dat_file: Path) -> pd.DataFrame:
    """
    Load an SU2 Tecplot `flow.dat`, slice at X_STATION, and return a clean,
    wall-to-edge-sorted profile with standardized column names.

    Columns returned: y, u, T, rho, mu  (plus normalized u_norm, t_norm).
    """
    if not dat_file.exists():
        raise FileNotFoundError(f"SU2 result not found: {dat_file}")

    with open(dat_file, "r") as fh:
        lines = fh.readlines()

    header_rows, col_names = 0, []
    for i, line in enumerate(lines):
        if "VARIABLES" in line:
            col_names = re.findall(r'"(.*?)"', line)
        if "ZONE" in line:
            header_rows = i + 1
            break
    if not col_names:
        raise ValueError(f"Could not parse VARIABLES header in {dat_file}")

    df = pd.read_csv(
        dat_file, skiprows=header_rows, sep=r"\s+",
        names=col_names, on_bad_lines="skip", low_memory=False,
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(how="all", axis=1, inplace=True)

    rename = {}
    for col in df.columns:
        c = col.lower()
        if c == "x" or "coordinatex" in c:
            rename[col] = "x"
        elif c == "y" or "coordinatey" in c:
            rename[col] = "y"
        elif "temperature" in c:
            rename[col] = "T"
        elif "density" in c:
            rename[col] = "rho"
        elif "laminar_viscosity" in c:
            rename[col] = "mu"
        elif "velocity" in c and "x" in c and "y" not in c:
            rename[col] = "u"
        elif "momentum" in c and "x" in c and "y" not in c:
            rename[col] = "mom_x"
    df.rename(columns=rename, inplace=True)

    if "u" not in df.columns and {"mom_x", "rho"} <= set(df.columns):
        df["u"] = df["mom_x"] / df["rho"]

    required = {"x", "y", "u", "T", "rho", "mu"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{dat_file} missing columns: {missing}")

    sl = df[(df["x"] > X_STATION - X_TOLERANCE) & (df["x"] < X_STATION + X_TOLERANCE)].copy()
    if sl.empty:
        raise ValueError(f"No nodes found at x = {X_STATION} m in {dat_file}")

    sl = sl.sort_values(by="y").drop_duplicates(subset="y").reset_index(drop=True)
    sl["u_norm"] = sl["u"] / U_INF
    sl["t_norm"] = sl["T"] / T_INF
    return sl[["y", "u", "T", "rho", "mu", "u_norm", "t_norm"]]


def load_dns() -> pd.DataFrame:
    """
    Load the NASA TMR / Duan et al. DNS statistics (M14Tw018).

    Returns columns: y_delta, y_plus, u_norm, t_norm, u_vd_plus, prt.
    Pr_t beyond the BL edge is unreliable (dT/dy -> 0); callers should mask.
    """
    rows: list[list[float]] = []
    with open(DNS_STAT_FILE, "r") as fh:
        for line in fh:
            s = line.strip()
            if not s or not (s[0].isdigit() or s[0] in "+-."):
                continue
            try:
                vals = [float(v) for v in s.split()]
            except ValueError:
                continue
            if len(vals) >= 28:
                rows.append(vals)
    if not rows:
        raise ValueError(f"No data rows parsed from {DNS_STAT_FILE}")

    arr = np.array(rows)
    return pd.DataFrame({
        "y_delta":   arr[:, DNS_COL["zod"]],
        "y_plus":    arr[:, DNS_COL["zplus"]],
        "u_norm":    arr[:, DNS_COL["u_uinf"]],
        "t_norm":    arr[:, DNS_COL["t_tinf"]],
        "u_vd_plus": arr[:, DNS_COL["Uvd"]],
        "prt":       arr[:, DNS_COL["Prt"]],
    })


# ==========================================================================
#                              PHYSICS HELPERS
# ==========================================================================
def van_driest(profile: pd.DataFrame, n_wall_fit: int = 4) -> dict:
    """
    Compute the van Driest-transformed velocity in wall units for a RANS slice.

    u_vd+ = (1/u_tau) * integral_0^u sqrt(rho/rho_w) du'
    y+    = y * u_tau / nu_w
    u_tau = sqrt(tau_w / rho_w),  tau_w = mu_w * (du/dy)|_wall

    Wall properties are taken at the first (smallest-y) node; the near-wall
    velocity gradient is estimated by a linear fit through the first few nodes.
    """
    y = profile["y"].to_numpy()
    u = profile["u"].to_numpy()
    rho = profile["rho"].to_numpy()
    mu = profile["mu"].to_numpy()

    rho_w, mu_w = rho[0], mu[0]
    nu_w = mu_w / rho_w

    k = min(n_wall_fit, len(y))
    dudy_wall = np.polyfit(y[:k], u[:k], 1)[0]
    tau_w = mu_w * dudy_wall
    u_tau = float(np.sqrt(max(tau_w, 0.0) / rho_w))

    integrand = np.sqrt(rho / rho_w)
    u_vd = np.concatenate(([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(u))))

    y_plus = y * u_tau / nu_w
    u_vd_plus = u_vd / u_tau
    return {"y_plus": y_plus, "u_vd_plus": u_vd_plus, "u_tau": u_tau, "cf": tau_w / (0.5 * rho[-1] * U_INF**2)}


def shape_vs_level(u_rans: np.ndarray, t_rans: np.ndarray,
                   u_dns: np.ndarray, t_dns: np.ndarray) -> dict:
    """
    Decompose the RANS-vs-DNS T(u) mismatch into a uniform level offset and a
    residual shape error. Comparison is done on a common velocity grid inside
    the boundary layer (U_MASK_LO < u/U_inf < U_MASK_HI).
    """
    u_grid = np.linspace(U_MASK_LO, U_MASK_HI, 100)
    order_r = np.argsort(u_rans)
    order_d = np.argsort(u_dns)
    t_r = np.interp(u_grid, u_rans[order_r], t_rans[order_r])
    t_d = np.interp(u_grid, u_dns[order_d], t_dns[order_d])

    diff = t_r - t_d
    raw_rmse = float(np.sqrt(np.mean(diff**2)))
    offset = float(np.mean(diff))
    shape_rmse = float(np.sqrt(np.mean((diff - offset)**2)))
    return {"raw_rmse": raw_rmse, "level_offset": offset, "shape_rmse": shape_rmse}


# ==========================================================================
#                              AIAA STYLE
# ==========================================================================
def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "lines.linewidth": 1.5,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.figsize": (3.5, 2.5),
        "figure.dpi": 600,
        "mathtext.fontset": "stix",
        "savefig.bbox": "tight",
    })


def _save(fig, stem: str) -> None:
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Saved] {png.name}, {pdf.name}")


# ==========================================================================
#                              CHECK 1
# ==========================================================================
def check1_van_driest(base: pd.DataFrame, calib: Optional[pd.DataFrame],
                      dns: pd.DataFrame, pr_opt: float) -> dict:
    print("\n[Check 1] Van Driest u_vd+ vs y+ (momentum field)")
    vd_b = van_driest(base)
    vd_c = van_driest(calib) if calib is not None else None
    if vd_c is not None:
        print(f"   u_tau: baseline={vd_b['u_tau']:.2f} m/s, calibrated={vd_c['u_tau']:.2f} m/s "
              f"(DNS header = 67.6 m/s)")
    else:
        print(f"   u_tau: baseline={vd_b['u_tau']:.2f} m/s (DNS header = 67.6 m/s) "
              f"[baseline-only; calibrated curve deferred to 1B]")

    dns_bl = dns[(dns["y_plus"] > 0.5) & (dns["u_norm"] < 0.999)]

    apply_style()
    fig, ax = plt.subplots()

    yp = np.logspace(0, np.log10(max(vd_b["y_plus"].max(), 1e3)), 200)
    ax.plot(yp, yp, ":", color="0.6", lw=1.0, label=r"$u^+ = y^+$")
    log_law = (1.0 / 0.41) * np.log(yp) + 5.2
    ax.plot(yp, log_law, "-.", color="0.4", lw=1.0,
            label=r"$\frac{1}{0.41}\ln y^+ + 5.2$")

    ax.plot(dns_bl["y_plus"], dns_bl["u_vd_plus"], "o" + C_DNS, markersize=3,
            markerfacecolor="white", markeredgewidth=0.8, label="DNS", zorder=5)
    ax.plot(vd_b["y_plus"], vd_b["u_vd_plus"], "--" + C_STANDARD, lw=1.5,
            label=r"Standard RANS ($Pr_t=0.9$)", zorder=4)
    if vd_c is not None:
        ax.plot(vd_c["y_plus"], vd_c["u_vd_plus"], "-" + C_CALIBRATED, lw=1.5,
                label=rf"Calibrated RANS ($Pr_t={pr_opt:.3f}$)", zorder=10)

    ax.set_xscale("log")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$u_{vd}^+$")
    ax.set_xlim(1, max(vd_b["y_plus"].max(), 1e3))
    ax.set_ylim(0, 30)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    _save(fig, "check1_van_driest_uplus")

    return {"u_tau_base": vd_b["u_tau"],
            "u_tau_calib": (vd_c["u_tau"] if vd_c is not None else None)}


# ==========================================================================
#                              CHECK 2
# ==========================================================================
def check2_tu_shape(base: pd.DataFrame, calib: pd.DataFrame, dns: pd.DataFrame,
                    pr_opt: float) -> dict:
    print("\n[Check 2] T(u) shape vs level (calibration space)")
    m_base = shape_vs_level(base["u_norm"].to_numpy(), base["t_norm"].to_numpy(),
                            dns["u_norm"].to_numpy(), dns["t_norm"].to_numpy())
    m_calib = shape_vs_level(calib["u_norm"].to_numpy(), calib["t_norm"].to_numpy(),
                             dns["u_norm"].to_numpy(), dns["t_norm"].to_numpy())
    print(f"   Standard  (0.9):   raw RMSE={m_base['raw_rmse']:.3f}, "
          f"level offset={m_base['level_offset']:+.3f}, shape RMSE={m_base['shape_rmse']:.3f}")
    print(f"   Calibrated({pr_opt:.3f}): raw RMSE={m_calib['raw_rmse']:.3f}, "
          f"level offset={m_calib['level_offset']:+.3f}, shape RMSE={m_calib['shape_rmse']:.3f}")

    dns_bl = dns[(dns["u_norm"] >= 0.0) & (dns["u_norm"] <= 1.0)]

    apply_style()
    fig, ax = plt.subplots()
    ax.plot(dns_bl["u_norm"], dns_bl["t_norm"], "o" + C_DNS, markersize=3,
            markerfacecolor="white", markeredgewidth=0.8, label="DNS", zorder=5)
    ax.plot(base["u_norm"], base["t_norm"], "--" + C_STANDARD, lw=1.5,
            label=r"Standard RANS ($Pr_t=0.9$)", zorder=4)
    ax.plot(calib["u_norm"], calib["t_norm"], "-" + C_CALIBRATED, lw=1.5,
            label=rf"Calibrated RANS ($Pr_t={pr_opt:.3f}$)", zorder=10)

    ax.set_xlabel(r"$u / U_\infty$")
    ax.set_ylabel(r"$T / T_\infty$")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    _save(fig, "check2_tu_shape")

    return {"base": m_base, "calib": m_calib}


# ==========================================================================
#                              CHECK 3
# ==========================================================================
def check3_prt_recovery(dns: pd.DataFrame, pr_opt: float, pr_rmse: float,
                        base_rmse: float) -> dict:
    print("\n[Check 3] Pr_t(RANS) vs Pr_t(DNS) (physical recovery)")
    mask = (dns["u_norm"] >= U_MASK_LO) & (dns["u_norm"] <= U_MASK_HI)
    dns_bl = dns[mask]
    prt_bl_avg = float(dns_bl["prt"].mean())
    print(f"   DNS Pr_t (BL-avg, {U_MASK_LO}<u/U<{U_MASK_HI}) = {prt_bl_avg:.3f}")
    print(f"   RANS-optimal Pr_t                          = {pr_opt:.3f}")
    print(f"   |difference|                               = {abs(prt_bl_avg - pr_opt):.3f} "
          f"({100*abs(prt_bl_avg - pr_opt)/prt_bl_avg:.1f}% of DNS)")

    apply_style()
    fig, ax = plt.subplots()
    ax.plot(dns_bl["prt"], dns_bl["y_delta"], "o" + C_DNS, markersize=3,
            markerfacecolor="white", markeredgewidth=0.8,
            label="DNS $Pr_t$ profile", zorder=5)
    ax.axvline(PR_T_DEFAULT, color=C_STANDARD, ls="--", lw=1.5,
               label=r"Standard ($Pr_t=0.9$)", zorder=4)
    ax.axvline(pr_opt, color=C_CALIBRATED, ls="-", lw=1.5,
               label=rf"Calibrated ($Pr_t={pr_opt:.3f}$)", zorder=10)
    ax.axvline(prt_bl_avg, color="0.4", ls=":", lw=1.2,
               label=rf"DNS BL-avg ({prt_bl_avg:.3f})", zorder=3)

    ax.set_xlabel(r"$Pr_t$")
    ax.set_ylabel(r"$y / \delta$")
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    _save(fig, "check3_prt_recovery")

    return {"dns_prt_bl_avg": prt_bl_avg, "rans_opt_prt": pr_opt}


# ==========================================================================
#                              MAIN
# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Pr_t calibration diagnostics.")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="Path to the Pr_t=0.9 baseline flow.dat "
                         "(default: results/Pr_0.9000/flow.dat).")
    ap.add_argument("--check1-only", action="store_true",
                    help="Regenerate only Check 1 (van Driest momentum check) "
                         "for the new baseline; defer Check 2/3 to 1B.")
    args = ap.parse_args()

    print("=" * 66)
    print(" Pr_t Calibration Diagnostics  (Mach 14 flat plate, cold wall)")
    print("=" * 66)

    base_path = args.baseline or (RESULTS_ROOT / "Pr_0.9000" / "flow.dat")
    base = load_su2_profile(base_path)
    dns = load_dns()
    print(f"Baseline: {base_path} ({len(base)} nodes), DNS ({len(dns)} points)")

    if args.check1_only:
        # Momentum check is ~Pr_t-independent; baseline-only is sufficient.
        check1_van_driest(base, None, dns, pr_opt=float("nan"))
        print("\n" + "=" * 66)
        print(" Done (Check 1 only). Figures written to:", OUTPUT_DIR)
        print("=" * 66)
        return

    pr_opt, pr_rmse = detect_optimal_pr()
    base_rmse = baseline_rmse()
    pr_opt_str = f"{pr_opt:.4f}"
    print(f"Optimal Pr_t = {pr_opt:.3f} (RMSE {pr_rmse:.3f})  |  "
          f"baseline Pr_t=0.9 RMSE {base_rmse:.3f}")

    calib = load_su2_profile(RESULTS_ROOT / f"Pr_{pr_opt_str}" / "flow.dat")
    print(f"Loaded calibrated ({len(calib)} nodes)")

    check1_van_driest(base, calib, dns, pr_opt)
    check2_tu_shape(base, calib, dns, pr_opt)
    check3_prt_recovery(dns, pr_opt, pr_rmse, base_rmse)

    print("\n" + "=" * 66)
    print(" Done. Figures written to:", OUTPUT_DIR)
    print("=" * 66)


if __name__ == "__main__":
    main()
