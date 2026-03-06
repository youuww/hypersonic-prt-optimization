#!/usr/bin/env python3
"""
Publication-quality convergence plots for Pr_t calibration campaigns.

Reusable class designed to handle multiple geometries (flat plate, ramp, etc.)
and produce AIAA-format figures with consistent styling across the paper.

Color convention (consistent across all paper figures):
    - Blue solid   = Calibrated / optimized (the "fix")
    - Red dashed   = Standard / default   (the "error")
    - Black circles = DNS benchmark data

Usage (standalone):
    python plot_convergence.py                              # auto-detect from results/
    python plot_convergence.py path/to/optimization_log.csv # explicit CSV path

Usage (as module):
    from plot_convergence import ConvergencePlotter
    plotter = ConvergencePlotter(log_csv="results/optimization_log.csv")
    plotter.plot_aiaa()
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ==========================================
#              CONFIGURATION
# ==========================================
SCRIPT_DIR: Path = Path(__file__).resolve().parent
RESULTS_ROOT: Path = SCRIPT_DIR.parent / "results"
DATA_DIR: Path = SCRIPT_DIR.parent / "data"
DNS_FILE: Path = DATA_DIR / "DNS Dataset.csv"     # T/U coupling (digitized from Murphy-Agarwal Fig. 5b)

# Physics constants — Mach 14, cold-wall flat plate (Zhang, Duan & Choudhari 2018)
U_INF: float = 1882.0    # [m/s]  freestream velocity
T_INF: float = 47.4      # [K]    freestream temperature
X_STATION: float = 1.5   # [m]    profile extraction location (downstream of transition)
X_TOLERANCE: float = 0.005  # [m] half-width of the extraction band

# Threshold for detecting crashed / diverged SU2 runs (penalty value from optimizer)
CRASH_RMSE_THRESHOLD: float = 20.0

# AIAA single-column style (shared across all publication plots)
# Ref: AIAA Journal formatting guidelines — 3.5 in column, serif fonts, inward ticks
AIAA_STYLE: dict[str, object] = {
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
    "figure.figsize": (3.5, 2.5),   # Single-column width
    "figure.dpi": 600,
    "mathtext.fontset": "stix",      # LaTeX-like math rendering
    "savefig.bbox": "tight",         # Trim white margins
}


# ==========================================
#        CONVERGENCE PLOTTER CLASS
# ==========================================
class ConvergencePlotter:
    """
    Generates publication-quality convergence plots from optimization logs.

    Designed for reuse across multiple calibration campaigns:
    - Flat plate (current)
    - Compression ramp (future)
    - Cylinder-flare (future)

    Parameters
    ----------
    log_csv : Path
        Path to optimization_log.csv (columns: Iteration, Pr_t, RMSE, Time_Sec).
    results_dir : Path, optional
        Parent directory containing Pr_*/flow.dat folders. Defaults to log_csv's parent.
    dns_csv : Path, optional
        Path to DNS benchmark CSV (u/U_inf, T/T_inf). Defaults to data/DNS Dataset.csv.
    """

    def __init__(
        self,
        log_csv: Path | str,
        results_dir: Optional[Path | str] = None,
        dns_csv: Path | str = DNS_FILE,
    ) -> None:
        self.log_csv: Path = Path(log_csv)
        self.results_dir: Path = Path(results_dir) if results_dir else self.log_csv.parent
        self.dns_csv: Path = Path(dns_csv)

        # --- Load optimization history (columns: Iteration, Pr_t, RMSE, Time_Sec) ---
        self.log: pd.DataFrame = pd.read_csv(self.log_csv)

        # Split into converged vs crashed runs (optimizer assigns penalty RMSE ≥ 20)
        self.valid: pd.DataFrame = self.log[self.log["RMSE"] < CRASH_RMSE_THRESHOLD].copy()
        self.crashed: pd.DataFrame = self.log[self.log["RMSE"] >= CRASH_RMSE_THRESHOLD].copy()

        # --- Load DNS benchmark (2-column CSV: u/U∞, T/T∞) ---
        dns: pd.DataFrame = pd.read_csv(self.dns_csv, header=None)
        self.dns_u: np.ndarray = dns.iloc[:, 0].values  # u / U_∞
        self.dns_t: np.ndarray = dns.iloc[:, 1].values  # T / T_∞

        # --- Identify best (minimum RMSE) iteration ---
        best_idx: int = int(self.valid["RMSE"].idxmin())
        self.best_iter: int = int(self.valid.loc[best_idx, "Iteration"])
        self.best_rmse: float = float(self.valid.loc[best_idx, "RMSE"])
        self.best_prt: float = float(self.valid.loc[best_idx, "Pr_t"])

        print(
            f"[ConvergencePlotter] Loaded {len(self.log)} iterations "
            f"from {self.log_csv.name}\n"
            f"   Best: Iter {self.best_iter}, Pr_t = {self.best_prt:.4f}, "
            f"RMSE = {self.best_rmse:.4f}"
        )

    # --------------------------------------------------
    #  Baseline RMSE computation (Pr_t = 0.9 default)
    # --------------------------------------------------
    def compute_baseline_rmse(self, pr_t: float = 0.9) -> Optional[float]:
        """
        Compute RMSE for a given Pr_t by loading its flow.dat and comparing T/U vs DNS.

        Uses the **same metric** as the optimizer (``su2_interface.calculate_loss``):
            RMSE = sqrt( mean( (T_rans - T_dns)^2 ) )
        evaluated on the T(u) profile at x = 1.5 m.

        This enables drawing a horizontal reference line on the convergence plot
        showing where the default (uncalibrated) model sits.

        Parameters
        ----------
        pr_t : float
            The Prandtl number whose results folder (``Pr_{pr_t:.4f}/flow.dat``)
            will be loaded and evaluated.

        Returns
        -------
        float or None
            RMSE value, or None if the flow.dat is missing / unparseable.
        """
        flow_dat: Path = self.results_dir / f"Pr_{pr_t:.4f}" / "flow.dat"
        if not flow_dat.exists():
            print(f"[Baseline] flow.dat not found: {flow_dat}")
            return None

        # --- Parse SU2 Tecplot file ---
        df: Optional[pd.DataFrame] = self._load_su2_flow(flow_dat)
        if df is None:
            return None

        # --- Extract wall-normal slice at X_STATION ± tolerance ---
        sl: pd.DataFrame = df[
            (df["x"] > X_STATION - X_TOLERANCE) & (df["x"] < X_STATION + X_TOLERANCE)
        ].copy()
        if sl.empty:
            return None

        # --- Normalize by freestream values ---
        sl["u_norm"] = sl["u"] / U_INF
        sl["t_norm"] = sl["T"] / T_INF
        sl = sl.sort_values("u_norm").drop_duplicates(subset="u_norm")

        # --- Interpolate DNS onto RANS u-grid and compute RMSE ---
        t_dns_interp: np.ndarray = np.interp(
            sl["u_norm"].values, self.dns_u, self.dns_t
        )
        rmse: float = float(np.sqrt(np.mean((sl["t_norm"].values - t_dns_interp) ** 2)))
        print(f"[Baseline] Pr_t = {pr_t} → RMSE = {rmse:.4f}")
        return rmse

    # --------------------------------------------------
    #  AIAA Publication Plot
    # --------------------------------------------------
    def plot_aiaa(
        self,
        baseline_pr: float = 0.9,
        output_prefix: str = "convergence_aiaa",
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Generate AIAA-format convergence plot (Fig. 3 in the paper).

        Features:
        - Optimization path (blue markers + line)
        - Best iteration highlighted (green star)
        - Horizontal dashed red line for default Pr_t = 0.9 RMSE
        - No title (AIAA convention — caption goes in the paper body)
        - Crashed iterations marked with red ×

        Parameters
        ----------
        baseline_pr : float
            Default Pr_t whose RMSE is drawn as a horizontal reference line.
        output_prefix : str
            Filename stem for PNG/PDF output.
        output_dir : Path, optional
            Where to save. Defaults to the post_processing/ folder.
        """
        out: Path = Path(output_dir) if output_dir else SCRIPT_DIR
        png_path: Path = out / f"{output_prefix}.png"
        pdf_path: Path = out / f"{output_prefix}.pdf"

        # --- Compute baseline RMSE for horizontal reference line ---
        baseline_rmse: Optional[float] = self.compute_baseline_rmse(baseline_pr)

        # --- Apply AIAA style globally ---
        plt.rcParams.update(AIAA_STYLE)
        fig, ax = plt.subplots()

        # ---- Optimization path: blue circles + connecting line ----
        # Blue = "calibrated / optimized" (consistent with T/U and velocity plots)
        if not self.valid.empty:
            ax.plot(
                self.valid["Iteration"], self.valid["RMSE"],
                "-o", color="b", markersize=4, markerfacecolor="b",
                markeredgecolor="b", lw=1.2,
                label="Brent's method", zorder=5,
            )

        # ---- Best iteration: green star with black edge ----
        ax.plot(
            self.best_iter, self.best_rmse,
            "*", color="green", markersize=10,
            markeredgecolor="k", markeredgewidth=0.5,
            label=rf"Optimal ($Pr_t = {self.best_prt:.3f}$)",
            zorder=10,
        )

        # ---- Baseline horizontal line: red dashed = "the default error" ----
        # Red = "standard / uncalibrated" (consistent color convention)
        if baseline_rmse is not None:
            ax.axhline(
                baseline_rmse, color="r", ls="--", lw=1.2,
                label=rf"Standard ($Pr_t = {baseline_pr}$)",
                zorder=2,
            )

        # ---- Crashed / diverged runs: red × at plot ceiling ----
        if not self.crashed.empty:
            y_top: float = ax.get_ylim()[1]
            ax.scatter(
                self.crashed["Iteration"],
                [y_top * 0.95] * len(self.crashed),
                c="red", marker="x", s=30, label="Diverged", zorder=6,
            )

        # ---- Axis labels (LaTeX math, no title — AIAA convention) ----
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"RMSE [$T/T_\infty$]")

        # Ensure y-axis starts at 0 with headroom above the baseline
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

        # Force integer ticks on x-axis (iterations are discrete)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        # ---- Legend & grid ----
        ax.legend(
            loc="center right", bbox_to_anchor=(1.0, 0.60),
            frameon=True, fancybox=False, edgecolor="black",
        )
        ax.grid(True, which="major", ls="--", alpha=0.4)

        # ---- Save high-resolution output (PNG for review, PDF for LaTeX) ----
        plt.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved: {png_path.name}, {pdf_path.name}")

        # ---- Print improvement summary to console ----
        if baseline_rmse is not None:
            reduction: float = (baseline_rmse - self.best_rmse) / baseline_rmse * 100
            print(
                f"[Summary] Default RMSE = {baseline_rmse:.4f}, "
                f"Calibrated RMSE = {self.best_rmse:.4f} "
                f"({reduction:+.1f}% reduction)"
            )

    # --------------------------------------------------
    #  Internal: SU2 Tecplot parser (same as su2_interface)
    # --------------------------------------------------
    @staticmethod
    def _load_su2_flow(dat_file: Path) -> Optional[pd.DataFrame]:
        """
        Parse SU2 Tecplot-format ``flow.dat`` into a pandas DataFrame.

        Handles the two-line header (VARIABLES + ZONE) and standardizes
        column names to: ``x``, ``T``, ``u`` (computed from momentum/density
        if the solver wrote conservative variables).

        Parameters
        ----------
        dat_file : Path
            Absolute path to the flow.dat file.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with at least columns ``x``, ``T``, ``u``.
            Returns None if parsing fails.
        """
        with open(dat_file, "r") as f:
            lines: list[str] = f.readlines()

        # --- Detect header: find VARIABLES line for column names, ZONE for data start ---
        header_rows: int = 0
        col_names: list[str] = []
        for i, line in enumerate(lines):
            if "VARIABLES" in line:
                col_names = re.findall(r'"(.*?)"', line)
            if "ZONE" in line:
                header_rows = i + 1
                break

        if not col_names:
            return None

        # --- Read numeric data ---
        df: pd.DataFrame = pd.read_csv(
            dat_file,
            skiprows=header_rows,
            sep=r"\s+",
            names=col_names,
            on_bad_lines="skip",
            low_memory=False,
        )
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)

        # --- Standardize column names (SU2 names vary by config) ---
        rename_map: dict[str, str] = {}
        for col in df.columns:
            c: str = col.lower()
            if c == "x" or "coordinatex" in c:
                rename_map[col] = "x"
            if "temperature" in c:
                rename_map[col] = "T"
            if "momentum" in c and "x" in c:
                rename_map[col] = "mom_x"
            if "density" in c:
                rename_map[col] = "rho"
        df.rename(columns=rename_map, inplace=True)

        # --- Derive velocity from conservative variables if needed ---
        if "u" not in df.columns and "mom_x" in df.columns:
            df["u"] = df["mom_x"] / df["rho"]

        return df


# ==========================================
#              STANDALONE ENTRY
# ==========================================
def _find_log_csv(root: Path) -> Path:
    """
    Auto-detect ``optimization_log.csv`` location.

    Search order:
      1. Latest ``*iter*`` run folder (e.g. ``turb_SA_flatplate_M14_11iter_260208/``)
      2. Flat ``results/`` directory (legacy layout)

    Raises FileNotFoundError if no log is found.
    """
    # Check for timestamped run folders first
    run_dirs: list[Path] = [d for d in root.iterdir() if d.is_dir() and "iter" in d.name]
    if run_dirs:
        latest: Path = max(run_dirs, key=lambda d: d.stat().st_mtime)
        log: Path = latest / "optimization_log.csv"
        if log.exists():
            return log

    # Fall back to flat results/ layout
    log = root / "optimization_log.csv"
    if log.exists():
        return log

    raise FileNotFoundError(f"No optimization_log.csv found under {root}")


# ==========================================
#              STANDALONE ENTRY
# ==========================================
if __name__ == "__main__":
    import sys

    # Accept explicit CSV path, otherwise auto-detect from results/
    csv_path: Path = Path(sys.argv[1]) if len(sys.argv) > 1 else _find_log_csv(RESULTS_ROOT)

    plotter = ConvergencePlotter(log_csv=csv_path)
    plotter.plot_aiaa()
