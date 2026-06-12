# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Automated calibration of the turbulent Prandtl number (Pr_t) in RANS turbulence models for hypersonic cold-wall boundary layers. The pipeline wraps the SU2 CFD solver in Python, runs simulations, computes RMSE against DNS data, and optimizes Pr_t. Validated result: Pr_t=0.566 (vs default 0.9) cuts temperature RMSE by 61.3% at Mach 13.6.

> **GP + Active Learning status (Jun 2026):** Mode 2 code is a first-pass scaffold — written but **not yet run end-to-end**. Restore point: git tag `v0.1-gp-scaffold`.

## Running the Pipeline

```bash
pip install -r requirements.txt

# Classical single-condition optimization (Brent's method, Mach 14 flat plate)
cd src && python run_optimization.py

# Active learning across multiple flow conditions (Bayesian Optimization)
cd src && python run_active_learning.py --budget 10
cd src && python run_active_learning.py --dry-run    # GP only, no CFD
cd src && python run_active_learning.py --resume     # from checkpoints/
```

SU2 must be on PATH. Parallel: `mpirun -n 4 SU2_CFD`. Set `num_cores=1` for serial.

## Architecture

Two execution modes share a common foundation:

**`case_config.py` — `FlowCondition`**: Dataclass holding complete case physics (Mach, T_inf, P_inf, Tw/Taw, Re, pressure-gradient angle, mesh, DNS path). Computes derived quantities (U_inf, T_wall, T_aw). Its `.feature_vector` property `[Mach, Tw/Taw, theta_pg]` is the GP surrogate input. Factory: `FlowCondition.mach14_flat_plate()`.

**`su2_interface.py` — `SU2Interface`**: Accepts an optional `FlowCondition` (defaults to Mach 14). `generate_config()` patches the base SU2 template at `config/turb_SA_flatplate_M14Tw018.cfg`, injecting Pr_t and freestream overrides. `run_su2()` invokes MPI+SU2 via subprocess. `calculate_loss()` extracts a wall-normal profile at `X_STATION=1.5m` from Tecplot output, normalizes, interpolates onto DNS data from `data/DNS Dataset.csv`, and returns RMSE.

**Mode 1 — `run_optimization.py`**: Single-condition Brent optimization. `scipy.optimize.minimize_scalar(bounds=(0.5, 0.95), method='bounded')` calls `objective_function(pr_t)` which runs the full generate→simulate→loss pipeline per trial.

**Mode 2 — Active Learning** (3 modules):
- `surrogate.py` — `PrtSurrogate`: BoTorch `SingleTaskGP`, Matern-5/2 ARD kernel. Maps `[Mach, Tw/Taw, theta_pg]` → optimal Pr_t. `predict()` returns mean + uncertainty. Saves/loads from `checkpoints/gp_model/`.
- `active_loop.py` — `ActiveCalibrationLoop`: Each `step()`: UCB acquisition → GP uncertainty check → if `std > threshold` call CFD callback → update GP. Logs to `checkpoints/active_learning_log.json`.
- `run_active_learning.py`: Orchestrator. `run_cfd_at_point()` is the callback — builds `FlowCondition` from tensor, creates `SU2Interface`, runs inner Brent optimization, returns `(optimal_prt, rmse)`.

## Critical Technical Gotchas

- **SU2 `Heat_Flux` column bug**: `flow.dat` has zeros at ~50% of wall nodes. Compute q_w from temperature gradient (`q_w = k_lam * |dT/dy|` at y=0) instead.
- **DNS scope**: The DNS profile is at one streamwise station. T-U comparison is valid (self-similar); direct q_w comparison is NOT.
- **Iteration count**: `SU2Interface.ITERATIONS = 51` is for quick tests. Production runs need 15,000 — set `runner.ITERATIONS = 15000`.
- **Path resolution**: All `src/` scripts resolve paths from `SCRIPT_DIR = Path(__file__).resolve().parent`. Run them from `src/`.
- **WSL2**: `matplotlib.use('Agg')` is forced in `su2_interface.py` before any pyplot import.
- **`generate_ramp.py`**: Has Hebrew comments, gitignored — do not commit without cleanup.

## Conventions

- All code comments and documentation in English only (no Hebrew — causes BiDi rendering issues).
- Figure colors: blue solid = calibrated RANS (Pr_t=0.566), red dashed = standard RANS (Pr_t=0.9), black hollow circles = DNS.
- Results layout: `results/{geometry}_{n_iter}iter_{date}/Pr_{val}/` contains `flow.dat`, `flow.vtu`, `history.csv`, `restart_flow.dat`, plot PNG. Active learning checkpoints in `checkpoints/`.
