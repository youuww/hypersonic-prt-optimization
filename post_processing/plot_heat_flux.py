"""
Fig 5: Wall Heat Flux Comparison — Pr_t=0.9 (standard) vs Pr_t=0.566 (calibrated)

NOTE: The Heat_Flux column in SU2's volume output (flow.dat) has zeros at
      random wall nodes. We compute q_w from the temperature gradient instead:
          q_w = k_lam * dT/dy |_wall   (k_lam = mu_lam * c_p / Pr_lam)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ==========================================
#              CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

CASES = {
    "Baseline": {
        "folder": "Pr_0.9000",
        "label": r"Standard RANS ($Pr_t = 0.9$)",
        "color": "#d62728",
        "style": "--",
    },
    "Calibrated": {
        "folder": "Pr_0.5660",
        "label": r"Calibrated RANS ($Pr_t = 0.566$)",
        "color": "#1f77b4",
        "style": "-",
    },
}

OUTPUT_FILE = SCRIPT_DIR / "comparison_heat_flux.png"
C_P = 1005.0       # J/(kg·K)
PR_LAM = 0.72       # Laminar Prandtl number (from SU2 config)
SMOOTHING_WIN = 15  # Rolling window for light noise reduction

# ---- Adjustable plot range ----
X_START = 0.6       # Change this to trim leading-edge transient
X_END   = 2.0

# ==========================================
#          AIAA PLOTTING STYLE
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",  # Ticks pointing inside
    "ytick.direction": "in",
    "lines.linewidth": 1.8,
    
    "figure.figsize": (5, 3.5),
    "figure.dpi": 300,
    "mathtext.fontset": "stix",
    "savefig.bbox": "tight",
})


def load_volume_data(folder_name: str):
    """Load the TECPLOT ASCII volume file."""
    dat_path = RESULTS_DIR / folder_name / "flow.dat"
    if not dat_path.exists():
        print(f"  !! MISSING: {dat_path}")
        return None

    with open(dat_path, "r") as fh:
        lines = fh.readlines()

    header_rows, col_names = 0, []
    for i, line in enumerate(lines):
        if "VARIABLES" in line:
            col_names = re.findall(r'"(.*?)"', line)
        if "ZONE" in line:
            header_rows = i + 1
            break

    df = pd.read_csv(
        dat_path, skiprows=header_rows, sep=r"\s+",
        names=col_names, on_bad_lines="skip", low_memory=False,
    )
    for c in col_names:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x", "y"])


def compute_wall_heat_flux(df: pd.DataFrame) -> pd.DataFrame:
    """Compute q_w = k_lam * |dT/dy| at y=0 using the first grid point above."""
    wall = (
        df[np.abs(df["y"]) < 1e-8]
        .sort_values("x")
        .drop_duplicates(subset="x", keep="first")
    )

    records: list[dict] = []
    for _, wp in wall.iterrows():
        x0 = wp["x"]
        if x0 < 0.05:
            continue  # skip inflow region
        # Find the first interior point directly above this wall node
        above = df[(df["x"] == x0) & (df["y"] > 1e-8) & (df["y"] < 5e-4)]
        if above.empty:
            continue
        near = above.sort_values("y").iloc[0]

        dTdy = (near["Temperature"] - wp["Temperature"]) / (near["y"] - wp["y"])
        mu_lam = wp["Laminar_Viscosity"]
        k_lam = mu_lam * C_P / PR_LAM
        # Heat into the wall is positive (cold wall, dT/dy < 0 → q_w > 0)
        qw = -k_lam * dTdy
        records.append({"x": x0, "qw": qw})

    return pd.DataFrame(records).sort_values("x").reset_index(drop=True)


def plot_heat_flux() -> None:
    fig, ax = plt.subplots()

    # Uses module-level X_START / X_END (easy to tweak at the top of the file)

    for name, cfg in CASES.items():
        print(f"[{name}] Loading {cfg['folder']}...")
        df = load_volume_data(cfg["folder"])
        if df is None:
            continue

        qw_df = compute_wall_heat_flux(df)
        mask = (qw_df["x"] >= X_START) & (qw_df["x"] <= X_END)
        qw_slice = qw_df[mask].copy()

        # Convert W/m² → kW/m² and light smoothing
        qw_slice["qw_kW"] = (
            qw_slice["qw"].abs().rolling(window=SMOOTHING_WIN, center=True).mean()
            / 1000.0
        )

        ax.plot(
            qw_slice["x"], qw_slice["qw_kW"],
            label=cfg["label"], color=cfg["color"], linestyle=cfg["style"],
        )
        mean_kw = qw_slice["qw_kW"].mean()
        print(f"  → {len(qw_slice)} pts, mean |q_w| = {mean_kw:.2f} kW/m²")

    # --- Formatting ---
    ax.set_xlabel(r"Position $x$ [m]")
    ax.set_ylabel(r"Wall Heat Flux $|q_w|$ [kW/m$^2$]")
    ax.set_xlim(X_START, X_END)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=600)
    plt.savefig(str(OUTPUT_FILE).replace(".png", ".pdf"))
    print(f"Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    plot_heat_flux()