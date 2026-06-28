#!/usr/bin/env python3
"""
Wall-shear / boundary-layer forensics for the Mach 14 baseline RANS run.

Why this exists
---------------
The van Driest check (Check 1 in plot_calibration_diagnostics.py) showed the RANS
near-wall velocity does NOT collapse onto the DNS / log-law: RANS u_tau ~ 15 m/s
vs DNS 67.6 m/s, and SU2's own Cf ~ 4e-5 is below even laminar Blasius. This
script localizes the cause by mapping wall and boundary-layer quantities along
the plate, rather than at a single station:

  1. Cf(x)        - SU2 column + recomputed from the near-wall velocity gradient,
                    against laminar Blasius and turbulent (incompressible +
                    van Driest II compressible) references.
  2. delta(x)     - 99% boundary-layer thickness growth vs theory.
  3. Re_x, Re_theta, u_tau, Re_tau at the analysis station.
  4. Velocity profiles at several x to see whether x=1.5 m is anomalous or the
     whole solution carries low wall shear (transition / under-development).

Read-only: loads results/Pr_0.9000/flow.dat once. Writes figures into this folder.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Unified DNS-consistent Mach 14 baseline (Pr_t=0.9, derived Re=1.07e7).
# Override the flow.dat path on the command line: `python investigate_wall_shear.py <flow.dat>`.
DEFAULT_FLOW_DAT = REPO_ROOT / "results" / "baseline_M14Tw018" / "Pr_0.9000" / "flow.dat"
FLOW_DAT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FLOW_DAT

# Freestream constants for the unified M14Tw018 case (case_config.dns_M14Tw018).
U_INF = 1882.1
T_INF = 47.1
GAMMA = 1.4
R_GAS = 287.0
MACH = 13.68

# Sutherland (air) for laminar viscosity references.
MU_REF, T_REF, S_SUTH = 1.716e-5, 273.15, 110.4


def sutherland(T: np.ndarray | float):
    T = np.asarray(T, dtype=float)
    return MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)


def load_full_field(dat_file: Path) -> pd.DataFrame:
    with open(dat_file, "r") as fh:
        head = [next(fh) for _ in range(5)]
    col_names, header_rows = [], 0
    with open(dat_file, "r") as fh:
        for i, line in enumerate(fh):
            if "VARIABLES" in line:
                col_names = re.findall(r'"(.*?)"', line)
            if "ZONE" in line:
                header_rows = i + 1
                break
    df = pd.read_csv(dat_file, skiprows=header_rows, sep=r"\s+",
                     names=col_names, on_bad_lines="skip", low_memory=False)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(how="all", axis=1, inplace=True)
    rename = {}
    for col in df.columns:
        c = col.lower()
        if c == "x" or "coordinatex" in c: rename[col] = "x"
        elif c == "y" or "coordinatey" in c: rename[col] = "y"
        elif "temperature" in c: rename[col] = "T"
        elif "density" in c: rename[col] = "rho"
        elif "laminar_viscosity" in c: rename[col] = "mu"
        elif "pressure" == c or (c == "pressure"): rename[col] = "p"
        elif "skin_friction_coefficient_x" in c: rename[col] = "cf_su2"
        elif "velocity" in c and "x" in c and "y" not in c: rename[col] = "u"
        elif "momentum" in c and "x" in c and "y" not in c: rename[col] = "mom_x"
    # pressure column name is exactly "Pressure"
    for col in df.columns:
        if col.lower() == "pressure":
            rename[col] = "p"
    df.rename(columns=rename, inplace=True)
    if "u" not in df.columns and {"mom_x", "rho"} <= set(df.columns):
        df["u"] = df["mom_x"] / df["rho"]
    return df


def wall_line(df: pd.DataFrame) -> pd.DataFrame:
    """Return wall-boundary nodes (y ~ 0) sorted by x."""
    w = df[df["y"] < 1e-9].copy()
    w = w.sort_values("x").drop_duplicates(subset="x")
    return w


def profile_at(df: pd.DataFrame, x0: float, tol: float = 0.005) -> pd.DataFrame:
    sl = df[(df["x"] > x0 - tol) & (df["x"] < x0 + tol)].copy()
    return sl.sort_values("y").drop_duplicates(subset="y").reset_index(drop=True)


def delta99(prof: pd.DataFrame) -> float:
    y, u = prof["y"].to_numpy(), prof["u"].to_numpy()
    target = 0.99 * U_INF
    idx = np.argmax(u >= target)
    if idx == 0:
        return float(y[-1])
    return float(np.interp(target, [u[idx - 1], u[idx]], [y[idx - 1], y[idx]]))


def u_tau_from(prof: pd.DataFrame, k: int = 4) -> float:
    y, u = prof["y"].to_numpy(), prof["u"].to_numpy()
    rho_w, mu_w = prof["rho"].iloc[0], prof["mu"].iloc[0]
    dudy = np.polyfit(y[:k], u[:k], 1)[0]
    return float(np.sqrt(max(mu_w * dudy, 0.0) / rho_w))


def main() -> None:
    print("=" * 66)
    print(" Wall-shear forensics : Mach 14 baseline (Pr_t = 0.9)")
    print("=" * 66)
    df = load_full_field(FLOW_DAT)
    print(f"Loaded field: {len(df)} nodes, columns = {list(df.columns)}")

    # Freestream reference: median over pure-freestream nodes (Mach near M_inf,
    # outside the boundary layer). Using max-y is unsafe (corner/far-field nodes
    # can carry non-physical imposed values).
    fs_mask = (df["Mach"] > MACH - 0.5) & (df["Mach"] < MACH + 0.5) & (df["y"] > 0.1)
    fs = df[fs_mask]
    rho_inf = float(fs["rho"].median())
    mu_inf = float(sutherland(T_INF))
    p_inf = float(fs["p"].median()) if "p" in fs else float("nan")
    print(f"Freestream: rho_inf={rho_inf:.4e} kg/m3, mu_inf={mu_inf:.4e}, "
          f"p_inf={p_inf:.2f} Pa, unit-Re={rho_inf*U_INF/mu_inf:.3e} /m")

    # --- Wall line ---
    w = wall_line(df)
    w = w[w["x"] > 0.0]  # on the plate
    print(f"Wall nodes on plate: {len(w)}  (x in [{w['x'].min():.3f}, {w['x'].max():.3f}])")

    x = w["x"].to_numpy()
    cf_su2 = w["cf_su2"].to_numpy() if "cf_su2" in w else np.full_like(x, np.nan)
    p_w = w["p"].to_numpy() if "p" in w else np.full_like(x, np.nan)
    T_w = w["T"].to_numpy()

    Re_x = rho_inf * U_INF * x / mu_inf
    cf_lam = 0.664 / np.sqrt(np.clip(Re_x, 1e-9, None))
    cf_turb_inc = 0.0592 / np.clip(Re_x, 1e-9, None) ** 0.2
    # van Driest II compressible factor (cold wall, adiabatic-ref simplified):
    # use a representative Fc from temperature ratio Tw/Tinf.
    Fc = (T_w / T_INF) ** -0.5  # crude compressibility scaling for plotting only
    cf_turb_comp = cf_turb_inc * np.clip(Fc, 0.05, 1.0)

    print(f"\nAt x=1.5 m:")
    i15 = int(np.argmin(np.abs(x - 1.5)))
    print(f"  Cf(SU2)={cf_su2[i15]:.3e}  Cf_lam={cf_lam[i15]:.3e}  "
          f"Cf_turb_inc={cf_turb_inc[i15]:.3e}  T_w={T_w[i15]:.1f}K  p_w={p_w[i15]:.1f}Pa")

    # --- BL growth + station metrics ---
    x_stations = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    deltas, profiles = [], {}
    print("\n  x[m]   delta99[mm]   u_tau[m/s]   Re_x        Re_tau")
    for xs in x_stations:
        pr = profile_at(df, xs)
        if len(pr) < 5:
            deltas.append(np.nan); continue
        d99 = delta99(pr)
        ut = u_tau_from(pr)
        nu_w = pr["mu"].iloc[0] / pr["rho"].iloc[0]
        re_tau = d99 * ut / nu_w
        re_x = rho_inf * U_INF * xs / mu_inf
        deltas.append(d99)
        profiles[xs] = pr
        print(f"  {xs:4.2f}   {d99*1e3:9.2f}   {ut:9.2f}   {re_x:.3e}   {re_tau:8.1f}")

    # ---------------- Figure 1: Cf(x) ----------------
    _style()
    fig, ax = plt.subplots()
    ax.plot(x, cf_su2, "-b", lw=1.5, label=r"RANS $C_f$ (SU2)")
    ax.plot(x, cf_lam, "--", color="0.4", lw=1.2, label=r"Laminar Blasius")
    ax.plot(x, cf_turb_inc, "-.", color="0.6", lw=1.2, label=r"Turbulent (incomp.)")
    ax.plot(x, cf_turb_comp, ":", color="0.3", lw=1.2, label=r"Turbulent (van Driest II)")
    ax.axvline(1.5, color="r", ls=":", lw=1.0, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$C_f$")
    ax.set_ylim(1e-5, 1e-2)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    _save(fig, "diag_cf_vs_x")

    # ---------------- Figure 2: delta(x) ----------------
    fig, ax = plt.subplots()
    xx = np.array(x_stations)
    ax.plot(xx, np.array(deltas) * 1e3, "-ob", ms=3, lw=1.5, label=r"RANS $\delta_{99}$")
    re_xx = rho_inf * U_INF * xx / mu_inf
    d_turb = 0.37 * xx / re_xx ** 0.2
    d_lam = 5.0 * xx / np.sqrt(re_xx)
    ax.plot(xx, d_turb * 1e3, "-.", color="0.6", lw=1.2, label=r"Turbulent $0.37x/Re_x^{0.2}$")
    ax.plot(xx, d_lam * 1e3, "--", color="0.4", lw=1.2, label=r"Laminar $5x/\sqrt{Re_x}$")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$\delta_{99}$ [mm]")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black")
    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    _save(fig, "diag_delta_vs_x")

    # ---------------- Figure 3: velocity profiles at several x ----------------
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(profiles)))
    for (xs, pr), col in zip(profiles.items(), cmap):
        ax.plot(pr["u"] / U_INF, pr["y"] * 1e3, "-", color=col, lw=1.3, label=f"x={xs:.2f} m")
    ax.set_xlabel(r"$u / U_\infty$")
    ax.set_ylabel(r"$y$ [mm]")
    ax.set_ylim(0, 90)
    ax.set_xlim(0, 1.02)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black", ncol=2)
    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    _save(fig, "diag_profiles_vs_x")

    print("\nVerdict hints:")
    print("  - If Cf(x) sits far below the laminar line everywhere -> not a")
    print("    transition artifact; either solver convergence or a setup issue.")
    print("  - If delta(x) >> turbulent theory -> BL is over-thick (gentle")
    print("    gradients -> low wall shear), consistent with the van Driest gap.")
    print("=" * 66)


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10, "lines.linewidth": 1.5, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "in", "ytick.direction": "in",
        "figure.figsize": (3.5, 2.5), "figure.dpi": 600,
        "mathtext.fontset": "stix", "savefig.bbox": "tight",
    })


def _save(fig, stem: str):
    png = SCRIPT_DIR / f"{stem}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(SCRIPT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"   [Saved] {png.name}")


if __name__ == "__main__":
    main()
