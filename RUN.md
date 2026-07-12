# Production Run Guide (Lab Machine)

One-page reproducibility guide for the **10 production runs** = 5 DNS cases x
{`Pr_t=0.9` baseline, optimal `Pr_t`}, at a converged production iteration count.
These produce the final matched-Re figures (Check 2/3) and the real GP training
targets. The cheap 6000-iter Brent calibrations that located each optimal `Pr_t`
were already run locally; this guide is only for the expensive precision pass.

## 1. Environment


| Component | Version / location                                             |
| --------- | -------------------------------------------------------------- |
| Python    | 3.9 (conda env `thesis_env`)                                   |
| SU2_CFD   | build at `~/SU2_install/bin/SU2_CFD` (match this build/commit) |
| MPI       | Open MPI 4.1.2 (`mpirun`)                                      |
| ML stack  | torch 2.8.0+cpu, gpytorch 1.11, botorch 0.10.0                 |


```bash
conda activate thesis_env
which SU2_CFD mpirun          # both must be on PATH
```

- **MPI slots:** this machine has **4 physical slots** — do NOT exceed `--cores 4`.
- **WSL sandbox gotcha:** run these commands in a normal shell WITHOUT the agent
sandbox (Landlock/bwrap is unsupported here) or the subprocess silently no-ops.



## 2. Choosing the iteration count (stationarity)

The 15k baseline was under-converged: wall heat flux `HF` was still drifting
3.3%/1000 iter at iter 15000 (`rms[RhoE]` only -1.63). Extrapolating the
decay, `HF` drift falls below ~0.5%/1000 near **24k iterations**.

- **Default:** `--iter 30000` (comfortable margin).
- **Ceiling:** `--iter 40000` if drift is still high.
- **Verify per run:** in each `history.csv`, check `HF` drift over the last 1000
iters is `< ~0.5%/1000`. If not, re-run that case with a higher `--iter`.



## 3. Run

```bash
cd src
python run_production.py --iter 30000 --cores 4          # all 5 cases (10 runs)
# or a subset / single case:
python run_production.py --cases M14Tw018 --iter 30000 --cores 4
```

Each case writes to `results/production/<case>_<iter>iter/`:

- `Pr_0.9000/` and `Pr_<optimal>/` — each with `flow.dat`, `history.csv`, plot PNG
- `optimization_log.csv` — two rows (baseline 0.9 + optimal) for the diagnostics

The optimal `Pr_t` per case is baked into `run_production.OPTIMAL_PRT`
(M2p5 0.736, M6Tw025 0.620, M6Tw076 0.788, M8Tw048 0.776, M14Tw018 0.650).

## 4. After the runs (back on the dev machine)

1. Copy the `results/production/` folders back to the repo.
2. Regenerate diagnostics against the matched-Re pair, e.g. for M14:
  ```bash
   cd post_processing/calibration_diagnostics
   python plot_calibration_diagnostics.py \
       --baseline   ../../results/production/M14Tw018_30000iter/Pr_0.9000/flow.dat \
       --calibrated ../../results/production/M14Tw018_30000iter/Pr_0.6505/flow.dat \
       --log        ../../results/production/M14Tw018_30000iter/optimization_log.csv
  ```
3. Recompile the memo (`tectonic calibration_diagnostics_summary.tex`) and update
  the Section 1 headline + Check 2/3 tables with the production numbers.

