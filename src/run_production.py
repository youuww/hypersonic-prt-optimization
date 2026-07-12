"""Production driver: matched-Re baseline + calibrated runs per DNS case.

For each selected DNS case this runs SU2 twice at a fixed (production)
iteration count:
    1. Pr_t = 0.9         (standard Spalart-Allmaras baseline)
    2. Pr_t = <optimal>   (from the local 6000-iter Brent calibration)

Both runs share the identical mesh / CFL / iteration count, so the
downstream Check 2 level-offset comparison reflects only Pr_t (and not a
difference in convergence state).

This is the deliverable intended for the lab / professor machine: the 10
runs (5 cases x {baseline, calibrated}) executed at a production iteration
count.  See ``RUN.md`` at the repo root for reproducibility instructions
and the HF-drift stationarity check used to justify ``--iter``.

Usage (from src/):
    python run_production.py                       # all 5 cases, 30000 iter
    python run_production.py --cases M14Tw018      # a single case
    python run_production.py --iter 40000 --cores 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from case_config import FlowCondition
from su2_interface import SU2Interface

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

# Standard Spalart-Allmaras baseline Pr_t.
BASELINE_PRT = 0.9

# Case name -> FlowCondition factory.
CASE_FACTORIES = {
    "M2p5":     FlowCondition.dns_M2p5,
    "M6Tw025":  FlowCondition.dns_M6Tw025,
    "M6Tw076":  FlowCondition.dns_M6Tw076,
    "M8Tw048":  FlowCondition.dns_M8Tw048,
    "M14Tw018": FlowCondition.dns_M14Tw018,
}

# Optimal Pr_t from the local 6000-iter x6 Brent calibrations (matched
# derived Re, CFL 250) -- the argmin-RMSE row of each case's
# optimization_log.csv.  SINGLE SOURCE OF TRUTH for the calibration targets:
# run_active_learning.get_seed_data() imports this dict for the GP seed.
# Production runs at higher iteration count refine these; the two-point
# {0.9, optimal} pair also validates that the optimum stays bracketed.
OPTIMAL_PRT = {
    "M2p5":     0.7360,
    "M6Tw025":  0.6204,
    "M6Tw076":  0.7880,
    "M8Tw048":  0.7761,
    "M14Tw018": 0.6505,
}


def run_case(case: str, n_iter: int, cores: int) -> None:
    """Run the baseline (Pr_t=0.9) and calibrated (optimal Pr_t) pair for one case."""
    flow = CASE_FACTORIES[case]()
    runner = SU2Interface(flow=flow, num_cores=cores)
    runner.ITERATIONS = n_iter

    out_dir = RESULTS_DIR / "production" / f"{case}_{n_iter}iter"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.RESULTS_DIR = out_dir

    print(f"\n{'=' * 60}")
    print(f" Production case: {case}  (iter={n_iter}, cores={cores})")
    print(f"{'=' * 60}")
    print(flow.summary())

    prt_values = [BASELINE_PRT, OPTIMAL_PRT[case]]
    rows = []
    for i, pr in enumerate(prt_values, start=1):
        pr_str = f"{pr:.4f}"
        run_id = f"prod_{case}_Pr{pr_str}"
        tag = "baseline" if pr == BASELINE_PRT else "calibrated"
        print(f"\n>>> [{case}] Run {i}/2 ({tag}): Pr_t={pr_str}")

        cfg = runner.generate_config(pr, run_id)
        (runner.SCRIPT_DIR / "flow.dat").unlink(missing_ok=True)

        t0 = time.time()
        ok = runner.run_su2(cfg)
        rmse = runner.calculate_loss("flow") if ok else 999.0
        elapsed = time.time() - t0
        print(f"    RMSE={rmse:.5f} | time={elapsed:.0f}s | success={ok}")

        if ok:
            runner.plot_results("flow", pr_str)
            runner.organize_files(pr_str)
        runner.cleanup(run_id)

        rows.append({
            "Pr_t": pr, "RMSE": rmse, "Iterations": n_iter,
            "Time_Sec": elapsed, "Role": tag,
        })

    # Two-row log (baseline 0.9 + optimal) so the diagnostics can resolve
    # detect_optimal_pr() (min RMSE) and baseline_rmse() (nearest to 0.9).
    log_path = out_dir / "optimization_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    print(f"\n[Log] {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Production baseline+calibrated SU2 runs per DNS case."
    )
    ap.add_argument(
        "--cases", nargs="+", choices=sorted(CASE_FACTORIES),
        default=sorted(CASE_FACTORIES),
        help="DNS cases to run (default: all 5).",
    )
    ap.add_argument(
        "--iter", type=int, default=30000,
        help="Production SU2 iterations (default 30000; see RUN.md stationarity note).",
    )
    ap.add_argument(
        "--cores", type=int, default=4,
        help="MPI cores (max 4 physical slots on this machine).",
    )
    args = ap.parse_args()

    print(f"Production plan: cases={args.cases} | iter={args.iter} | cores={args.cores}")
    print(f"Total SU2 runs: {2 * len(args.cases)}  (baseline 0.9 + optimal per case)")

    for case in args.cases:
        run_case(case, args.iter, args.cores)

    print("\n=== Production runs complete. ===")


if __name__ == "__main__":
    main()
