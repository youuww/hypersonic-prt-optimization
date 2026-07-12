"""Orchestrator: Active Learning loop for Pr_t calibration.

This is the top-level entry point that wires together:
  - FlowCondition      (what physics to simulate)
  - SU2Interface        (how to run CFD)
  - PrtSurrogate        (the GP "brain")
  - ActiveCalibrationLoop  (the decision engine)

It replaces the old Brent-based run_optimization.py with a
Bayesian Optimization workflow that strategically picks which
flow conditions to simulate next.

Usage:
    cd src/
    python run_active_learning.py

The script will:
    1. Load any existing training data (from previous runs)
    2. Build / restore the GP surrogate
    3. Run N active-learning iterations
    4. Save all results to checkpoints/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

from scipy.optimize import minimize_scalar

from case_config import FlowCondition
from su2_interface import SU2Interface
from surrogate import PrtSurrogate
from active_loop import ActiveCalibrationLoop, Decision
from run_production import OPTIMAL_PRT, CASE_FACTORIES

# ------------------------------------------------------------------ #
#                         Configuration                                #
# ------------------------------------------------------------------ #

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# Pr_t search bounds for the inner Brent optimizer (per-point calibration)
PRT_BOUNDS = (0.3, 1.2)
PRT_TOLERANCE = 1e-3
PRT_MAX_ITER = 8

# Active-learning settings
UNCERTAINTY_THRESHOLD = 0.03
UCB_BETA = 2.0
DEFAULT_BUDGET = 10


def setup_logging() -> None:
    """Configure console + file logging."""
    fmt = "[%(asctime)s] %(name)-20s %(levelname)s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(CHECKPOINT_DIR / "active_learning.log"),
        ],
    )

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#     CFD callback: run Brent optimization at a given flow condition   #
# ------------------------------------------------------------------ #

def _dns_case_for(x_candidate: Tensor) -> FlowCondition:
    """Map a (discrete) acquisition point back to its DNS FlowCondition.

    Acquisition is restricted to the 5 DNS feature vectors, so every proposed
    point equals one of the DNS cases.  Recovering the exact
    :class:`FlowCondition` gives the real NASA freestream (t_inf, p_inf,
    derived Re) and the matching DNS profile for ``calculate_loss`` — no
    placeholder freestream and no interpolated target.  We match to the nearest
    DNS case by feature-vector distance (an exact hit under discrete acquisition).
    """
    x = x_candidate.squeeze().tolist()
    best, best_d = None, float("inf")
    for flow in FlowCondition.all_dns_cases():
        d = sum((a - b) ** 2 for a, b in zip(flow.feature_vector, x))
        if d < best_d:
            best, best_d = flow, d
    return best


def run_cfd_at_point(x_candidate: Tensor) -> tuple[float, float, float]:
    """Run the full SU2 optimization pipeline at a candidate point.

    Given X = [Mach, Tw/Taw, theta_pg], this function:
      1. Recovers the DNS FlowCondition for the (discrete) candidate point
      2. Creates an SU2Interface for that condition
      3. Runs Brent's method to find the optimal Pr_t
      4. Returns (optimal_Pr_t, best_RMSE, obs_var)

    ``obs_var`` is the per-point GP observation variance
    (``FlowCondition.observation_noise_var``): air cases get the trusted noise
    tier, while a non-air case (M8 nitrogen) is automatically down-weighted via
    its ``fluid`` field.

    NOTE: This function currently supports flat-plate cases only
    (pg_angle = 0).  Ramp support requires a mesh generator and
    will be added in Phase 3 of the thesis.

    Freestream consistency: because acquisition is DISCRETE over the 5 DNS
    feature vectors (see ``ActiveCalibrationLoop.choices``), the candidate maps
    back to a real DNS ``FlowCondition`` with the accurate NASA freestream and a
    real DNS target — resolving the former placeholder-freestream limitation.
    """
    x = x_candidate.squeeze().tolist()

    flow = _dns_case_for(x_candidate)
    logger.info(
        "=== CFD CALLBACK: M=%.2f, Tw/Taw=%.3f, theta=%.1f -> DNS case %s ===",
        x[0], x[1], x[2], flow.label,
    )
    logger.info("Flow condition:\n%s", flow.summary())

    runner = SU2Interface(flow=flow, num_cores=4)
    runner.ITERATIONS = 6000 # 30000 production iteration count. 6000 to find Opt_Prt and than do 30000.

    best_prt = None
    best_rmse = 999.0
    
    def objective(pr_t: float) -> float:
        """Objective function for Brent optimization: runs SU2 at given Pr_t and returns RMSE.t
        to use in minimize_scalar.  Updates best_prt and best_rmse if this is the best result so far."""
        nonlocal best_prt, best_rmse
        pr_t = float(pr_t)
        run_id = f"AL_{flow.label}_Pr{pr_t:.4f}"

        cfg = runner.generate_config(pr_t, run_id)

        # Clean previous output
        flow_file = runner.SCRIPT_DIR / "flow.dat"
        flow_file.unlink(missing_ok=True)

        success = runner.run_su2(cfg)
        if not success:
            runner.cleanup(run_id)
            return 100.0

        rmse = runner.calculate_loss("flow")

        if rmse < best_rmse:
            best_rmse = rmse
            best_prt = pr_t
            runner.organize_files(f"{pr_t:.4f}")

        runner.cleanup(run_id)
        logger.info("  Pr_t=%.4f -> RMSE=%.5f", pr_t, rmse)
        return rmse

    t0 = time.time()
    res = minimize_scalar(
        objective,
        bounds=PRT_BOUNDS,
        method="bounded",
        options={"xatol": PRT_TOLERANCE, "maxiter": PRT_MAX_ITER},
    )
    elapsed = time.time() - t0

    optimal_prt = res.x if best_prt is None else best_prt
    obs_var = flow.observation_noise_var
    logger.info(
        "=== CFD COMPLETE: Pr_t*=%.4f, RMSE=%.5f, obs_var=%.2e, time=%.0fs ===",
        optimal_prt, best_rmse, obs_var, elapsed,
    )

    return optimal_prt, best_rmse, obs_var


# ------------------------------------------------------------------ #
#                         Seed data                                    #
# ------------------------------------------------------------------ #

def get_seed_data() -> tuple[Tensor, Tensor, Tensor]:
    """Return the initial training data from completed calibrations.

    Each entry pairs a :class:`FlowCondition` (the physics) with its
    RANS-optimal Pr_t.  Per-point observation variance comes from
    ``FlowCondition.observation_noise_var`` (single source of truth), so a
    non-air case (e.g. M8 nitrogen) is automatically down-weighted.

    All five DNS flat-plate cases are now real SU2 Brent calibrations at the
    matched derived Re (``run_production.OPTIMAL_PRT`` is the single source of
    truth for the targets).  The M14 optimum is 0.650 (matched-Re), which
    SUPERSEDES the old provisional 0.566 (fit to the earlier, worse fixed-Re
    baseline).  Production runs at higher iteration count may refine these.
    """
    calibrated: list[tuple[FlowCondition, float]] = [
        (factory(), OPTIMAL_PRT[case]) for case, factory in CASE_FACTORIES.items()
    ]

    X = torch.tensor(
        [flow.feature_vector for flow, _ in calibrated], dtype=torch.float64
    )
    Y = torch.tensor(
        [[prt] for _, prt in calibrated], dtype=torch.float64
    )
    Yvar = torch.tensor(
        [[flow.observation_noise_var] for flow, _ in calibrated],
        dtype=torch.float64,
    )
    return X, Y, Yvar


# ------------------------------------------------------------------ #
#                           Main                                       #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Active Learning for Pr_t calibration"
    )
    parser.add_argument(
        "--budget", type=int, default=DEFAULT_BUDGET,
        help="Number of active-learning iterations",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without CFD (GP suggestions only)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint",
    )
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()

    logger.info("=" * 60)
    logger.info("  Active Learning for Pr_t Calibration")
    logger.info("  Budget: %d iterations | Dry-run: %s", args.budget, args.dry_run)
    logger.info("=" * 60)

    # --- Load or initialize data ---
    gp_dir = CHECKPOINT_DIR / "gp_model"
    if args.resume and gp_dir.exists():
        logger.info("Resuming from checkpoint: %s", gp_dir)
        surrogate = PrtSurrogate.load(gp_dir)
    else:
        X_seed, Y_seed, Yvar_seed = get_seed_data()
        logger.info("Initializing GP with %d seed points", X_seed.shape[0])
        surrogate = PrtSurrogate(
            train_X=X_seed, train_Y=Y_seed, train_Yvar=Yvar_seed
        )

    logger.info("\n%s", surrogate.summary())

    # --- Discrete design space: acquire only over the 5 DNS feature vectors ---
    # Real ground truth exists only at the DNS cases, so we restrict acquisition
    # to them (avoids baked interpolation error; keeps freestream consistent).
    choices = torch.tensor(
        [flow.feature_vector for flow in FlowCondition.all_dns_cases()],
        dtype=torch.float64,
    )

    # --- Build the active learning loop ---
    loop = ActiveCalibrationLoop(
        surrogate=surrogate,    # the GP "brain" that makes predictions and uncertainty estimates
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
        beta=UCB_BETA,
        checkpoint_dir=CHECKPOINT_DIR,
        choices=choices,
    )

    # --- CFD callback (None if dry-run) ---
    callback = None if args.dry_run else run_cfd_at_point

    # --- Run the loop ---
    for i in range(args.budget):
        logger.info("\n--- Active Learning Iteration %d / %d ---", i + 1, args.budget)
        result = loop.step(run_cfd_callback=callback)

        logger.info(
            "  Result: %s | Pr_t=%.4f +/- %.4f | CFD Pr_t=%s",
            result.decision.value,  # Whether we trusted the GP or ran CFD
            result.predicted_prt,   # GP mean prediction at x_candidate
            result.predicted_std,   # GP uncertainty at x_candidate
            f"{result.observed_prt:.4f}" if result.observed_prt else "N/A", # observed Pr_t only available if CFD was run
        )

    # --- Final report ---
    logger.info("\n" + "=" * 60)
    logger.info(loop.status())
    logger.info(surrogate.summary())
    logger.info("Checkpoints saved to: %s", CHECKPOINT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
