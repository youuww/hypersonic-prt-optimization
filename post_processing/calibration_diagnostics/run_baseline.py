"""Full Mach 14 baseline run (Scope 1A, Phase 3).

Runs ONE SU2 simulation of the unified DNS-consistent M14Tw018 case at the
standard Pr_t = 0.9 and the derived Reynolds number (Re = 1.07e7), using the
CFL ceiling selected by run_cfl_probe.py, to a converged / stationary target.

Outputs land in  results/baseline_M14Tw018/Pr_0.9000/  (flow.dat, flow.vtu,
history.csv, restart_flow.dat), which the diagnostics scripts read via:

    python plot_calibration_diagnostics.py --check1-only \
        --baseline ../../results/baseline_M14Tw018/Pr_0.9000/flow.dat
    python investigate_wall_shear.py ../../results/baseline_M14Tw018/Pr_0.9000/flow.dat

Usage:
    python run_baseline.py --iter 30000 --cores 4 --cfl 150
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
from case_config import FlowCondition  # noqa: E402
from su2_interface import SU2Interface  # noqa: E402

RUN_NAME = "baseline_M14Tw018"


def patch_cfl(cfg_path: Path, cfl_max: float) -> None:
    lines = cfg_path.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip().startswith("CFL_ADAPT_PARAM="):
            lines[i] = f"CFL_ADAPT_PARAM= ( 0.1, 1.1, 1.0, {cfl_max:.1f} )\n"
    cfg_path.write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--cfl", type=float, default=150.0,
                    help="Adaptive-CFL ceiling chosen by run_cfl_probe.py.")
    ap.add_argument("--save-freq", type=int, default=2000)
    args = ap.parse_args()

    # SU2Interface.run_su2 launches SU2 in the current working directory and the
    # mesh lives in src/, so execute from there (configs/outputs also land in src/).
    os.chdir(SRC_DIR)

    flow = FlowCondition.dns_M14Tw018()
    runner = SU2Interface(flow=flow, num_cores=args.cores)
    runner.ITERATIONS = args.iter
    runner.SAVE_FREQ = args.save_freq

    out_dir = runner.RESULTS_DIR / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.RESULTS_DIR = out_dir

    print("=" * 60)
    print(" M14 BASELINE (Pr_t=0.9, derived Re, unified setup)")
    print("=" * 60)
    print(flow.summary())
    print(f"\nITER={args.iter} | CFL_max={args.cfl} | cores={args.cores}")
    print(f"Output -> {out_dir}/Pr_0.9000/\n")

    cfg = runner.generate_config(0.9, "baseline")
    patch_cfl(cfg, args.cfl)

    for stale in ("flow.dat", "history.csv", "flow.vtu", "restart_flow.dat"):
        (SRC_DIR / stale).unlink(missing_ok=True)

    t0 = time.time()
    ok = runner.run_su2(str(cfg))
    dt = time.time() - t0
    print(f"\nSU2 finished in {dt/60:.1f} min (success={ok}).")

    if ok:
        rmse = runner.calculate_loss("flow")
        print(f"Baseline T-u RMSE vs DNS (matched Re) = {rmse:.4f}")
        runner.plot_results("flow", "0.9000")
        runner.organize_files("0.9000")

        # Report convergence + QoI stationarity from the moved history file.
        hist = out_dir / "Pr_0.9000" / "history.csv"
        if hist.exists():
            df = pd.read_csv(hist)
            df.columns = [c.strip().strip('"').strip() for c in df.columns]
            last = df.iloc[-1]
            tail = df.tail(min(1000, len(df)))
            hf_drift = (abs(tail["HF"].iloc[-1] - tail["HF"].iloc[0])
                        / abs(tail["HF"].iloc[-1])) if "HF" in df.columns else float("nan")
            print(f"\nFinal residuals: rms[Rho]={last['rms[Rho]']:.2f}, "
                  f"rms[RhoE]={last['rms[RhoE]']:.2f}")
            print(f"Wall heat flux HF={last.get('HF', float('nan')):.3e}, "
                  f"drift over last {len(tail)} it = {hf_drift:.2e}")

    runner.cleanup("baseline")
    print("\nDone.")


if __name__ == "__main__":
    main()
