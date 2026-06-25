"""Sanity-check the GP + Active Learning machinery (Option 2).

PURPOSE
-------
Before spending days of SU2 compute, we validate that the GP surrogate and
the active-learning loop behave CORRECTLY with multiple data points:
  - uncertainty is ~zero at data, grows away from data
  - the M8 (nitrogen) point gets elevated noise => GP does NOT interpolate
    it exactly (visible larger posterior std there)
  - the acquisition function proposes sensible high-information points

IMPORTANT — PLACEHOLDER DATA
----------------------------
This script uses the DNS boundary-layer-averaged Pr_t (column 27) as a
STAND-IN for the training target.  The REAL target is the RANS-optimal
constant Pr_t obtained by SU2 calibration (not yet run for 4/5 cases).
=> Results here validate the CODE, not the science.  Do not cite numbers.

Usage:
    cd src/
    python validate_surrogate.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from case_config import FlowCondition
from surrogate import PrtSurrogate
from active_loop import ActiveCalibrationLoop

OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# DNS BL-averaged Pr_t per case (from extract_dns_profiles.py).
# PLACEHOLDER targets — see module docstring.
DNS_PRT_PLACEHOLDER = {
    "M2.5_Tw1.00": 0.944,
    "M5.86_Tw0.25": 0.947,
    "M5.86_Tw0.76": 0.854,
    "M7.86_Tw0.48": 0.929,    # nitrogen -> elevated noise
    "M13.68_Tw0.18": 0.639,
}

# Observation noise (std) per data quality tier.
SIGMA_AIR = 0.01     # trustworthy air cases
SIGMA_N2 = 0.05      # M8: nitrogen-vs-air mismatch => trust less (5x std, 25x var)


def build_placeholder_dataset() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble (X, Y, Yvar) tensors from the 5 DNS cases."""
    X_rows, Y_rows, Yvar_rows = [], [], []
    for flow in FlowCondition.all_dns_cases():
        prt = DNS_PRT_PLACEHOLDER[flow.label]
        X_rows.append(flow.feature_vector)            # [Mach, Tw/Taw, theta]
        Y_rows.append([prt])
        sigma = SIGMA_N2 if "7.86" in flow.label else SIGMA_AIR
        Yvar_rows.append([sigma ** 2])

    X = torch.tensor(X_rows, dtype=torch.float64)
    Y = torch.tensor(Y_rows, dtype=torch.float64)
    Yvar = torch.tensor(Yvar_rows, dtype=torch.float64)
    return X, Y, Yvar


def check_uncertainty_behavior(gp: PrtSurrogate, X: torch.Tensor) -> None:
    """Verify uncertainty is low at data and higher away / at the N2 point."""
    print("\n--- Uncertainty behavior ---")

    # 1. At each training point
    res = gp.predict(X)
    for flow, mean, std in zip(
        FlowCondition.all_dns_cases(), res.mean, res.std
    ):
        tag = "  <- N2 (elevated noise)" if "7.86" in flow.label else ""
        print(
            f"  {flow.label:16s}: Pr_t={mean:.3f} +/- {std:.3f}{tag}"
        )

    # 2. Far from any data (Mach 10, hot wall, large angle — unsampled corner)
    x_far = torch.tensor([[10.0, 0.95, 20.0]], dtype=torch.float64)
    far = gp.predict(x_far)
    print(
        f"  {'FAR (M10,hot,ramp)':16s}: Pr_t={far.mean.item():.3f} "
        f"+/- {far.std.item():.3f}  <- should be LARGEST"
    )


def check_active_learning(gp: PrtSurrogate) -> None:
    """Confirm the acquisition proposes a real (non-random) next point."""
    print("\n--- Active learning suggestion ---")
    loop = ActiveCalibrationLoop(surrogate=gp, uncertainty_threshold=0.03)
    x_next = loop.suggest_next_point()
    decision, mean, std = loop.evaluate_point(x_next)
    m, tw, th = [round(v, 2) for v in x_next.squeeze().tolist()]
    print(f"  Next query: Mach={m}, Tw/Taw={tw}, theta={th}")
    print(f"  Decision:   {decision.value}  (Pr_t={mean:.3f} +/- {std:.3f})")


def plot_gp_slice(gp: PrtSurrogate, X: torch.Tensor, Y: torch.Tensor) -> Path:
    """Plot a 1-D GP slice (Pr_t vs Mach) with the 2-sigma band.

    The slice fixes Tw/Taw and theta at representative values; the 5 data
    points (which have their own Tw/Taw) are overlaid and colored by Tw/Taw.
    """
    tw_fixed, theta_fixed = 0.55, 0.0
    mach_grid = torch.linspace(2.5, 14.0, 120, dtype=torch.float64)
    X_grid = torch.stack(
        [mach_grid,
         torch.full_like(mach_grid, tw_fixed),
         torch.full_like(mach_grid, theta_fixed)],
        dim=1,
    )
    pred = gp.predict(X_grid)
    mean = pred.mean.numpy()
    lower = pred.lower.numpy()
    upper = pred.upper.numpy()
    mg = mach_grid.numpy()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(mg, mean, "b-", lw=2, label=f"GP mean (Tw/Taw={tw_fixed}, slice)")
    ax.fill_between(
        mg, lower, upper, color="b", alpha=0.15,
        label=r"GP 95% CI ($\pm2\sigma$)",
    )

    # Data points, colored by their actual Tw/Taw
    x_np = X.numpy()
    y_np = Y.numpy().ravel()
    sc = ax.scatter(
        x_np[:, 0], y_np, c=x_np[:, 1], cmap="coolwarm",
        s=140, edgecolors="k", zorder=5, label="DNS placeholder points",
    )
    plt.colorbar(sc, ax=ax, label="Tw/Taw of data point")

    # Mark the N2 point
    for i, flow in enumerate(FlowCondition.all_dns_cases()):
        if "7.86" in flow.label:
            ax.annotate(
                "M8 (N$_2$, elevated noise)",
                (x_np[i, 0], y_np[i]),
                textcoords="offset points", xytext=(10, 15),
                fontsize=9, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred"),
            )

    ax.axhline(0.9, color="gray", ls=":", alpha=0.7, label="Standard Pr_t=0.9")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Turbulent Prandtl number  Pr$_t$")
    ax.set_title(
        "GP Surrogate Sanity Check  —  PLACEHOLDER DATA (DNS BL-avg Pr_t)\n"
        "Validates code behavior only; NOT a scientific result"
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "validation_gp_placeholder.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    print("=" * 64)
    print("  GP + Active Learning MACHINERY VALIDATION  (Option 2)")
    print("  *** PLACEHOLDER DATA — validates code, not science ***")
    print("=" * 64)

    X, Y, Yvar = build_placeholder_dataset()
    print(f"\nBuilt dataset: {X.shape[0]} points, 3 features")
    print(f"  M8 noise std = {SIGMA_N2} (air cases = {SIGMA_AIR})")

    gp = PrtSurrogate(train_X=X, train_Y=Y, train_Yvar=Yvar)
    print("\n" + gp.summary())

    check_uncertainty_behavior(gp, X)
    check_active_learning(gp)

    out = plot_gp_slice(gp, X, Y)
    print(f"\n[Plot] Saved GP visualization -> {out}")

    print("\n" + "=" * 64)
    print("  VALIDATION COMPLETE.  If uncertainty grows away from data and")
    print("  the M8 point is NOT interpolated exactly, the machinery works.")
    print("=" * 64)


if __name__ == "__main__":
    main()
