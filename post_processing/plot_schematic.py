"""
Fig 1: Hypersonic Flat Plate — Computational Domain Schematic
=============================================================
AIAA-format schematic for the Turbulence Course Project paper.
Draws a 2D computational domain (2 m x 1 m) with labeled boundary
conditions, representative flow features (shock wave, boundary layer),
a mesh-clustering inset (y+ < 1), and dimensional annotations.

Output: domain_schematic.png / .pdf   (600 DPI, AIAA single-column)

Author : Matar Hedi
Date   : Feb 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# ==========================================
#           OUTPUT CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PNG = SCRIPT_DIR / "domain_schematic.png"
OUTPUT_PDF = SCRIPT_DIR / "domain_schematic.pdf"

# ==========================================
#           PLOTTING STYLE (AIAA)
# ==========================================
# Matches style defined in plot_AIAA_publication.py
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "lines.linewidth": 1.5,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 600,
    "mathtext.fontset": "stix",            # LaTeX-like math rendering
    "savefig.bbox": "tight",               # Trim white margins on save
})

# ==========================================
#       DOMAIN / PHYSICS PARAMETERS
# ==========================================
DOMAIN_W: float = 2.0      # Domain width  [m]  (streamwise)
DOMAIN_H: float = 1.0      # Domain height [m]  (wall-normal)
MACH: float = 13.6              # Freestream Mach number
T_WALL: int = 302           # Isothermal wall temperature [K]


# ==========================================
#           DRAWING FUNCTION
# ==========================================
def draw_domain_schematic() -> None:
    """Create an AIAA-quality schematic of the flat plate domain."""

    # ------------------------------------------------------------------
    # 1.  Canvas & Coordinate System
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.5))       # AIAA single-column

    # Padding around the domain rectangle for labels & arrows
    pad_l, pad_r = 0.55, 0.45                         # left / right margins
    pad_b, pad_t = 0.35, 0.30                         # bottom / top  margins
    ax.set_xlim(-pad_l, DOMAIN_W + pad_r)
    ax.set_ylim(-pad_b, DOMAIN_H + pad_t)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # 2.  Domain Rectangle
    # ------------------------------------------------------------------
    domain_rect = patches.Rectangle(
        (0, 0), DOMAIN_W, DOMAIN_H + pad_t/2,
        linewidth=1.8, edgecolor="black", facecolor="#f5f5f5", zorder=2,
    )
    ax.add_patch(domain_rect)

    # ------------------------------------------------------------------
    # 3.  Wall Hatching  (engineering convention — inclined lines below wall)
    # ------------------------------------------------------------------
    hatch_len = 0.06                                   # length of each hatch mark
    n_hatch = 30
    for i in range(n_hatch + 1):
        x_h = i * DOMAIN_W / n_hatch
        ax.plot(
            [x_h, x_h - hatch_len * 0.7],
            [0, -hatch_len],
            color="black", linewidth=0.5, zorder=2,
        )

    # ------------------------------------------------------------------
    # 4.  Boundary-Condition Labels
    # ------------------------------------------------------------------
    bc_fontsize = 8

    # Bottom — Isothermal Wall
    ax.text(
        DOMAIN_W / 2, -0.14,
        r"Isothermal Wall ($T_w = %d\,$K)" % T_WALL,
        ha="center", va="top", fontsize=bc_fontsize, style="italic",
    )

    # Left — Supersonic Inlet
    ax.text(
        -0.08, DOMAIN_H / 2,
        r"Inlet" + "\n" + r"($M_\infty = %.1f$)" % MACH,
        ha="right", va="center", fontsize=bc_fontsize, style="italic",
    )

    # Top — Farfield  (shifted up to separate from inset title)
    ax.text(
        DOMAIN_W * 0.40, DOMAIN_H + 0.22,
        "Farfield",
        ha="center", va="bottom", fontsize=bc_fontsize, style="italic",
    )

    # Right — Outlet
    ax.text(
        DOMAIN_W + 0.08, DOMAIN_H / 2,
        "Outlet",
        ha="left", va="center", fontsize=bc_fontsize, style="italic",
        rotation=-90,
    )

    # ------------------------------------------------------------------
    # 5.  Flow Direction Arrow  (freestream, left → right)
    # ------------------------------------------------------------------
    arrow_y = DOMAIN_H * 0.80
    ax.annotate(
        "", xy=(0.25, arrow_y), xytext=(-0.35, arrow_y),
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"),
    )
    ax.text(
        -0.30, arrow_y + 0.06, "Flow", fontsize=7,
        ha="left", va="bottom", weight="bold",
    )

    # ------------------------------------------------------------------
    # 6.  Shock Wave  (representative oblique shock — red dashed)
    # ------------------------------------------------------------------
    shock_angle = np.deg2rad(15)                       # representative angle
    x_shock = np.linspace(0, DOMAIN_W, 200)
    y_shock = np.tan(shock_angle) * x_shock

    # Clip to domain height
    mask = y_shock <= DOMAIN_H
    ax.plot(
        x_shock[mask], y_shock[mask],
        color="red", linestyle="--", linewidth=1.0, zorder=3,
    )
    # Label — positioned along the shock line, left of the inset
    lbl_x = 0.60
    lbl_y = np.tan(shock_angle) * lbl_x + 0.07
    ax.text(
        lbl_x, lbl_y, "Shock Wave",
        fontsize=6.5, color="red", style="italic",
        rotation=np.degrees(shock_angle),
    )

    # ------------------------------------------------------------------
    # 7.  Boundary Layer  (representative δ(x) ~ √x — blue solid)
    # ------------------------------------------------------------------
    x_bl = np.linspace(0, DOMAIN_W, 200)
    y_bl = 0.12 * np.sqrt(x_bl)                       # δ ∝ √x  (laminar-like growth)
    ax.plot(x_bl, y_bl, color="blue", linewidth=0.9, zorder=3)

    # Label with δ symbol
    ax.text(
        1.05, 0.15, r"Boundary Layer $\delta(x)$",
        fontsize=6, color="blue", style="italic", rotation=4,
    )

    # ------------------------------------------------------------------
    # 8.  Mesh Detail Inset  (wall-normal clustering, y+ < 1)
    # ------------------------------------------------------------------
    # Position the inset — shifted left to clear right edge & separate from Farfield
    ax_ins = ax.inset_axes([0.28, 0.55, 0.22, 0.28])  # [x0, y0, w, h] in axes coords
    ax_ins.set_xlim(0, 1)
    ax_ins.set_ylim(0, 1)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.set_title(r"Near-wall mesh ($y^+\!<\!1$)", fontsize=5.5, pad=2)
    for spine in ax_ins.spines.values():
        spine.set_linewidth(0.6)

    # Horizontal lines — exponentially clustered near the wall (bottom)
    y_cluster = np.array([0.0, 0.015, 0.04, 0.08, 0.14, 0.22, 0.33, 0.47, 0.64, 0.82, 1.0])
    for y in y_cluster:
        ax_ins.axhline(y, color="black", linewidth=0.35)

    # Vertical lines — uniform spacing
    for x in np.linspace(0, 1, 7):
        ax_ins.axvline(x, color="black", linewidth=0.35)


    # ------------------------------------------------------------------
    # 9.  Dimension Annotations  (L and H)
    # ------------------------------------------------------------------
    dim_color = "dimgray"
    dim_fs = 6.5

    # Horizontal dimension — L = 2 m  (below the hatching)
    y_dim_h = -0.28
    ax.annotate(
        "", xy=(DOMAIN_W, y_dim_h), xytext=(0, y_dim_h),
        arrowprops=dict(arrowstyle="<->", color=dim_color, lw=0.7),
    )
    ax.text(
        DOMAIN_W / 2, y_dim_h - 0.04,
        r"$L = 2\,$m", ha="center", va="top",
        fontsize=dim_fs, color=dim_color,
    )

    # Vertical dimension — H = 1 m  (to the right of outlet label)
    x_dim_v = DOMAIN_W + 0.28
    ax.annotate(
        "", xy=(x_dim_v, DOMAIN_H), xytext=(x_dim_v, 0),
        arrowprops=dict(arrowstyle="<->", color=dim_color, lw=0.7),
    )
    ax.text(
        x_dim_v + 0.04, DOMAIN_H / 2,
        r"$H = 1\,$m", ha="left", va="center",
        fontsize=dim_fs, color=dim_color, rotation=90,
    )

    # ------------------------------------------------------------------
    # 10. Internal Annotation  (solver info — upper-left open area)
    # ------------------------------------------------------------------
    ax.text(
        DOMAIN_W * 0.75, DOMAIN_H * 0.80,
        "SA turbulence model\nStructured grid",
        fontsize=5.5, ha="center", va="center",
        color="dimgray", linespacing=1.4,
    )

    # ------------------------------------------------------------------
    # 11. Save
    # ------------------------------------------------------------------
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"[Fig 1] Saved: {OUTPUT_PNG.name}, {OUTPUT_PDF.name}")
    plt.close(fig)


# ==========================================
#                  MAIN
# ==========================================
if __name__ == "__main__":
    draw_domain_schematic()
