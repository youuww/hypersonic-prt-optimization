"""CFL / convergence probe for the unified Mach 14 baseline (Scope 1A, Phase 3).

Runs short SU2 runs (Pr_t = 0.9, derived Re) at several adaptive-CFL ceilings
and reports, per setting:

  * stability      -- no NaN/Inf in residuals; density residual still descending,
  * energy descent -- slope of rms[RhoE] over the final window [decades / 1000 it],
  * QoI plateau    -- relative drift of the wall heat flux (HF) over the final window.

The goal is to pick the highest CFL ceiling that stays stable at Mach 13.68 and
gives the fastest, monotone energy-residual descent with a stationary wall heat
flux. If none is clearly stable, we fall back to CFL=50 and define convergence
by QoI stationarity rather than a residual floor.

Usage (from anywhere):
    python run_cfl_probe.py                 # CFL in {50,150,250}, 3000 iter, 6 cores
    python run_cfl_probe.py --iter 2000 --cores 6 --cfl 50 150 250

Read/exec model: SU2 is executed in src/ (mesh lives there); per-CFL history CSVs
and the summary live in this folder under probe_results/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
PROBE_DIR = SCRIPT_DIR / "probe_results"

sys.path.insert(0, str(SRC_DIR))
from case_config import FlowCondition  # noqa: E402
from su2_interface import SU2Interface  # noqa: E402


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().strip('"').strip() for c in df.columns]
    return df


def patch_cfl(cfg_path: Path, cfl_max: float) -> None:
    """Rewrite the CFL_ADAPT_PARAM ceiling in an already-generated SU2 config."""
    lines = cfg_path.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip().startswith("CFL_ADAPT_PARAM="):
            lines[i] = f"CFL_ADAPT_PARAM= ( 0.1, 1.1, 1.0, {cfl_max:.1f} )\n"
    cfg_path.write_text("".join(lines))


def run_one(cfl_max: float, n_iter: int, cores: int) -> Path:
    """Run a single probe at the given CFL ceiling; return the saved history CSV."""
    runner = SU2Interface(flow=FlowCondition.dns_M14Tw018(), num_cores=cores)
    runner.ITERATIONS = n_iter
    runner.SAVE_FREQ = n_iter  # only the final volume file

    run_id = f"probe_cfl{int(cfl_max)}"
    cfg = runner.generate_config(0.9, run_id)
    patch_cfl(cfg, cfl_max)

    # SU2 reads the mesh and writes outputs in the cwd; run in src/.
    for stale in ("flow.dat", "history.csv", "flow.vtu", "restart_flow.dat"):
        (SRC_DIR / stale).unlink(missing_ok=True)

    cmd = ["mpirun", "-n", str(cores), "SU2_CFD", str(cfg)]
    if cores <= 1:
        cmd = ["SU2_CFD", str(cfg)]

    print(f"\n=== CFL ceiling {cfl_max:g} | {n_iter} iter | {cores} cores ===")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=SRC_DIR, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  !!! SU2 exited {proc.returncode} (likely unstable at this CFL).")
        print("  stderr tail:", proc.stderr.decode(errors="ignore")[-300:])

    out = PROBE_DIR / f"history_cfl{int(cfl_max)}.csv"
    hist = SRC_DIR / "history.csv"
    if hist.exists():
        shutil.copy(hist, out)
    print(f"  done in {dt/60:.1f} min -> {out.name}")
    cfg.unlink(missing_ok=True)
    return out


def analyse(csv: Path, cfl_max: float, window: int) -> dict:
    """Extract stability + energy-descent + QoI-stationarity metrics."""
    if not csv.exists():
        return {"cfl_max": cfl_max, "stable": False, "reason": "no history (crashed)"}

    df = _clean_cols(pd.read_csv(csv))
    it = df["Inner_Iter"].to_numpy(dtype=float)
    rho = df["rms[Rho]"].to_numpy(dtype=float)
    rhoE = df["rms[RhoE]"].to_numpy(dtype=float)
    hf = df["HF"].to_numpy(dtype=float) if "HF" in df.columns else np.full_like(it, np.nan)

    finite = np.all(np.isfinite(rho)) and np.all(np.isfinite(rhoE))
    n = len(it)
    w = min(window, max(2, n // 2))

    # Energy descent slope over the final window: decades per 1000 iterations.
    sl_E = np.polyfit(it[-w:], rhoE[-w:], 1)[0] * 1000.0
    sl_rho = np.polyfit(it[-w:], rho[-w:], 1)[0] * 1000.0

    # Wall-heat-flux relative drift over the final window.
    hf_drift = (abs(hf[-1] - hf[-w]) / abs(hf[-1])) if np.isfinite(hf[-1]) and hf[-1] != 0 else np.nan

    # Stable = finite residuals AND density not growing in the final window.
    stable = bool(finite and sl_rho <= 0.05)

    return {
        "cfl_max": cfl_max,
        "n_iter": int(it[-1]),
        "stable": stable,
        "rms_rho_final": float(rho[-1]),
        "rms_rhoE_final": float(rhoE[-1]),
        "energy_slope_dec_per_1k": float(sl_E),
        "density_slope_dec_per_1k": float(sl_rho),
        "hf_final": float(hf[-1]) if np.isfinite(hf[-1]) else None,
        "hf_drift_final_window": float(hf_drift) if np.isfinite(hf_drift) else None,
    }


def pick(results: list[dict]) -> dict | None:
    """Highest stable CFL with genuine (negative) energy descent."""
    ok = [r for r in results if r.get("stable") and r.get("energy_slope_dec_per_1k", 0) < 0]
    if not ok:
        return None
    return sorted(ok, key=lambda r: r["cfl_max"])[-1]


def plot(results: list[dict], window: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for r in results:
        cfl = r["cfl_max"]
        csv = PROBE_DIR / f"history_cfl{int(cfl)}.csv"
        if not csv.exists():
            continue
        df = _clean_cols(pd.read_csv(csv))
        it = df["Inner_Iter"]
        ax1.plot(it, df["rms[RhoE]"], lw=2, label=f"CFL max {cfl:g}")
        if "HF" in df.columns:
            ax2.plot(it, df["HF"], lw=2, label=f"CFL max {cfl:g}")
    ax1.set_xlabel("Inner iteration")
    ax1.set_ylabel(r"$\mathrm{rms}[\rho E]$  (energy residual)")
    ax1.set_title("Energy-residual descent vs CFL ceiling")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.set_xlabel("Inner iteration")
    ax2.set_ylabel("Wall heat flux  HF")
    ax2.set_title("QoI (wall heat flux) approach to stationarity")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PROBE_DIR / f"cfl_probe.{ext}", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfl", type=float, nargs="+", default=[50, 150, 250])
    ap.add_argument("--iter", type=int, default=3000)
    ap.add_argument("--cores", type=int, default=6)
    ap.add_argument("--window", type=int, default=1000,
                    help="final-iteration window for slope/stationarity metrics")
    args = ap.parse_args()

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for cfl in args.cfl:
        csv = run_one(cfl, args.iter, args.cores)
        results.append(analyse(csv, cfl, args.window))

    chosen = pick(results)
    summary = {
        "case": "M14Tw018 (Pr_t=0.9, derived Re=1.07e7)",
        "n_iter": args.iter,
        "cores": args.cores,
        "window": args.window,
        "results": results,
        "chosen_cfl_max": chosen["cfl_max"] if chosen else None,
        "note": ("Highest stable CFL with negative energy slope."
                 if chosen else
                 "No stable accelerated setting; fall back to CFL=50 + QoI stationarity."),
    }
    (PROBE_DIR / "cfl_probe_summary.json").write_text(json.dumps(summary, indent=2))
    plot(results, args.window)

    print("\n================ CFL PROBE SUMMARY ================")
    for r in results:
        print(json.dumps(r, indent=2))
    print(f"\nChosen CFL ceiling: {summary['chosen_cfl_max']}")
    print(summary["note"])
    print("==================================================")


if __name__ == "__main__":
    main()
