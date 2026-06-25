# Hypersonic Turbulent Prandtl Number Optimization

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![SU2](https://img.shields.io/badge/SU2-CFD-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![MPI](https://img.shields.io/badge/MPI-Parallel-purple)

<p align="center">
  <img src="post_processing/optimization_final_flow.gif" alt="Optimization Animation" width="600">
</p>

Automated calibration of RANS turbulence models for hypersonic cold-wall boundary layers using data-driven inverse modeling.

---

## Key Results

| Metric | Baseline (Pr_t=0.9) | Calibrated (Pr_t=0.566) | Improvement |
|--------|----------------------|--------------------------|-------------|
| Temperature RMSE | 0.739 | 0.286 | **-61.3%** |
| Turbulent Prandtl Number | 0.9 (SA default) | 0.566 (optimized) | Data-driven calibration |

<p align="center">
  <img src="post_processing/optimization_profile_physics.gif" alt="T-U Profile Evolution" width="500">
</p>

---

## The Problem

Standard RANS turbulence models assume a constant turbulent Prandtl number (Pr_t = 0.9). This assumption breaks down for hypersonic flows with cold walls, leading to significant errors in heat flux prediction - critical for thermal protection system design.

## The Solution

This project implements an **automated calibration pipeline** with two execution modes:

**Mode 1 — Single-condition optimization (validated at Mach 14)**
1. Wraps the SU2 CFD solver in a Python interface
2. Runs parametric simulations automatically
3. Computes loss against DNS ground truth
4. Optimizes Pr_t using Brent's method (SciPy)

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   SciPy         │────>│  SU2 RANS    │────>│  Loss Function  │
│   Optimizer     │     │  Solver      │     │  (RMSE vs DNS)  │
│  (Brent's)      │<────│  (Mach 14)   │<────│                 │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

**Mode 2 — GP surrogate + active learning (machinery validated; SU2 calibration pending)**

Extends Mode 1 to multiple flow conditions using a Gaussian Process (BoTorch) that maps `[Mach, Tw/Taw, pressure-gradient angle]` → optimal Pr_t, with UCB acquisition to decide where to run new SU2 simulations. The GP/active-learning machinery is **validated** against a 5-case DNS family (`validate_surrogate.py`), including per-point observation noise to down-weight lower-fidelity data. The remaining step is replacing the DNS placeholder targets with **real RANS-optimal Pr_t** from per-case SU2 calibration.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ FlowCondition│────>│  GP Surrogate│────>│  UCB         │
│ (case_config)│     │  (BoTorch)   │     │  Acquisition │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │ high uncertainty
                                                  v
                                          ┌──────────────┐
                                          │ SU2 + Brent  │
                                          │ (inner loop) │
                                          └──────────────┘
```

---

## Test Case

| Parameter | Value |
|-----------|-------|
| Mach Number | 13.6 |
| Wall Temperature Ratio (Tw/Taw) | 0.186 |
| Reynolds Number | 5×10⁶ |
| Turbulence Model | Spalart-Allmaras |
| Validation Data | DNS (Murphy & Agarwal, 2025) |

---

## Technologies

- **CFD Solver:** [SU2](https://su2code.github.io/) (open-source, MPI-parallel)
- **Optimization:** SciPy (Brent's bounded method)
- **Surrogate Model:** PyTorch + GPyTorch + BoTorch (GP, active learning)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib (AIAA publication style)
- **Environment:** Linux/WSL2, Python 3.9+

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Mode 1: Single-condition optimization (validated)
cd src
python run_optimization.py

# Mode 2: Active learning (scaffold — not yet validated)
python run_active_learning.py --dry-run    # GP logic only, no SU2
python run_active_learning.py --budget 10    # full loop (requires SU2 + MPI)
python run_active_learning.py --resume       # resume from checkpoints/
```

SU2 must be on PATH. Parallel runs use `mpirun -n 4 SU2_CFD`.

---

## Project Structure

```
├── src/
│   ├── run_optimization.py       # Mode 1: Brent optimization (validated)
│   ├── run_active_learning.py    # Mode 2: GP + active learning orchestrator
│   ├── case_config.py            # FlowCondition dataclass + 5 DNS case factories
│   ├── surrogate.py              # GP surrogate (BoTorch, per-point noise support)
│   ├── active_loop.py            # UCB acquisition loop
│   ├── su2_interface.py          # SU2 wrapper (accepts FlowCondition)
│   ├── extract_dns_profiles.py   # NASA TMR .dat -> normalized T-u profile CSVs
│   ├── validate_surrogate.py     # GP/AL machinery sanity check (placeholder data)
│   └── generate_ramp.py          # Mesh generator (future work)
├── checkpoints/                  # GP model + AL log (runtime, gitignored)
├── config/                       # SU2 configuration files
├── data/
│   └── dns_database/             # 5-case DNS family (Mach 2.5-14): raw + profiles
└── post_processing/              # Visualization scripts
```

---

## References

- Murphy, A. R. & Agarwal, R. K. (2025). *Application and Evaluation of the Wray-Agarwal Turbulence Model with Compressibility Corrections in SU2 for RANS Hypersonic Flow Prediction.* AIAA Aviation Forum.

---

## Author

**Matar Hedi**  
M.Sc. Researcher, Technion | Thermal Engineer, Elbit Systems  
[LinkedIn](https://www.linkedin.com/in/matar-hedi)

---

## License

MIT
