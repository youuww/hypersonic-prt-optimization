# Pr_t Calibration Diagnostics

Self-contained diagnostic suite that stress-tests the Mach 14 cold-wall flat-plate
turbulent-Prandtl-number (`Pr_t`) calibration. It answers one question: **is the
calibrated `Pr_t = 0.566` a physical result, or a fudge factor hiding a momentum-model
error?**

This folder is **isolated** from the rest of the pipeline — it only *reads* existing
results and DNS data, and writes its own figures/PDF here. No existing code is modified
and no CFD is re-run.

> Full write-up with figures, tables, and the plan:
> **[`calibration_diagnostics_summary.pdf`](calibration_diagnostics_summary.pdf)**.

---

## TL;DR verdict

- **Check 2 (T–u space): clean pass.** Calibrated RANS collapses onto DNS; the
  `Pr_t = 0.9` error is almost entirely a uniform level offset that `Pr_t` removes.
- **Check 3 (Pr_t recovery): supportive.** Calibrated `0.566` matches the near-wall DNS
  `Pr_t` and is within 11.5% of the DNS boundary-layer average (`0.639`). Both far below `0.9`.
- **Check 1 (van Driest u+): did not collapse — but explained.** RANS `u_tau ≈ 14.7 m/s`
  vs DNS `67.6`. Forensics show this is a Reynolds-number/flow-state difference + the
  Mach-14 strong viscous-interaction regime (+ incomplete energy convergence), **not** a
  turbulence-model failure.
- **Bottom line:** the velocity mismatch is real but does not invalidate the calibration,
  because calibration is done in the station-invariant, self-similar `T(u)` space. `Pr_t`
  is a genuine physical correction.

---

## Contents

| File | What it is |
|------|------------|
| `plot_calibration_diagnostics.py` | The three checks (van Driest `u+`, `T(u)` shape-vs-level, `Pr_t` recovery). |
| `investigate_wall_shear.py` | Wall-shear / boundary-layer forensics along the plate (`Cf(x)`, `δ(x)`, profiles). |
| `calibration_diagnostics_summary.tex` / `.pdf` | Technical memo summarizing everything + next steps. |
| `check1_van_driest_uplus.{png,pdf}` | Check 1 figure. |
| `check2_tu_shape.{png,pdf}` | Check 2 figure. |
| `check3_prt_recovery.{png,pdf}` | Check 3 figure. |
| `diag_cf_vs_x.{png,pdf}` | `Cf(x)` vs laminar/turbulent references. |
| `diag_delta_vs_x.{png,pdf}` | Boundary-layer growth `δ99(x)`. |
| `diag_profiles_vs_x.{png,pdf}` | Velocity profiles at several stations. |

---

## How to run

From this folder, with the project Python environment active (`thesis_env`):

```bash
# 1. Generate the three calibration checks
python plot_calibration_diagnostics.py

# 2. Generate the wall-shear forensics (loads the full baseline flow.dat once)
python investigate_wall_shear.py

# 3. Rebuild the PDF memo (tectonic is installed in the conda base env)
tectonic calibration_diagnostics_summary.tex
```

Both scripts force the `Agg` matplotlib backend (WSL-safe) and write PNG + PDF here.

---

## Data sources (read-only)

- **RANS baseline:** `../../results/Pr_0.9000/flow.dat`
- **RANS calibrated:** `../../results/Pr_<optimal>/flow.dat` (optimum auto-detected from
  `../../results/optimization_log.csv`, currently `Pr_0.5660`)
- **DNS:** `../../data/dns_database/M14Tw018_Stat.dat`
  (NASA TMR / Zhang, Duan & Choudhari 2018). Columns used (1-indexed):
  3 = `y+`, 5 = `u/U∞`, 7 = `T/T∞`, 25 = `Uvd` (van Driest), 27 = `Pr_t`.

---

## Key numbers

| Quantity | Value |
|----------|-------|
| Optimal `Pr_t` (RMSE) | 0.566 (0.286) |
| Baseline `Pr_t` (RMSE) | 0.9 (0.739) |
| T/U RMSE reduction | −61.3% |
| DNS `Pr_t` (BL-avg) | 0.639 |
| RANS `u_tau` vs DNS | 14.7 vs 67.6 m/s |
| `Re_tau` (x=1.5 m) vs DNS | 182 vs 646 |
| Wall pressure vs freestream | 244 vs 109 Pa (2.2×) |
| Convergence `rms[ρ]` / `rms[ρE]` | −6.1 / −0.70 |

---

## Next steps (see memo §6)

1. Re-run the `Pr_t = 0.9` baseline to a converged energy residual (`rms[ρE] ≲ −5`) and
   regenerate Check 1.
2. Compare at matched Reynolds number / `Re_tau` to remove the flow-state confound.
3. Optionally use the Trettel–Larsson transform (DNS column 26) alongside van Driest.
4. Resume the core thesis track: GP + Active Learning multi-point calibration (M2.5–M14).
5. Extend to the compression-ramp / SBLI geometry and test `Pr_t` transferability.
