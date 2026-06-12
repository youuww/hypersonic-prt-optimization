"""Flow condition definitions for hypersonic RANS calibration.

Each FlowCondition encapsulates the complete physics of a simulation case:
Mach number, freestream thermodynamics, wall temperature ratio, and geometry.

It also provides the feature vector [Mach, Tw/Taw, theta_pg] consumed by
the GP surrogate model.

Usage:
    from case_config import FlowCondition

    # Predefined case (backward-compatible with existing Mach 14 pipeline)
    case = FlowCondition.mach14_flat_plate()

    # Custom case for multi-Mach data generation
    case = FlowCondition(mach=8.0, t_inf=50.0, p_inf=500.0,
                         tw_ratio=0.3, re=5e6, pg_angle=0.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---- Calorically perfect air (standard values) ----
GAMMA: float = 1.4
R_AIR: float = 287.058      # Specific gas constant [J/(kg*K)]
PR_LAM: float = 0.72        # Laminar Prandtl number


@dataclass
class FlowCondition:
    """Complete thermodynamic + geometric definition of a hypersonic case.

    Attributes
    ----------
    mach        : Freestream Mach number.
    t_inf       : Freestream static temperature [K].
    p_inf       : Freestream static pressure [Pa].
    tw_ratio    : Wall-to-adiabatic-wall temperature ratio (Tw / Taw).
    re          : Reynolds number (per REYNOLDS_LENGTH in the SU2 config).
    pg_angle    : Pressure-gradient angle [deg].
                  0 for flat plates; ramp deflection angle for compression
                  corners.  Named generically so it extends to future
                  geometries (cylinder-flare, etc.).
    mesh_file   : SU2 mesh filename (resolved relative to run directory).
    dns_data_path : Optional path to a two-column DNS CSV (u/U_inf, T/T_inf)
                    used for loss computation.
    label       : Human-readable tag for logs and plots.
    """

    mach: float
    t_inf: float
    p_inf: float
    tw_ratio: float
    re: float
    pg_angle: float = 0.0
    mesh_file: str = "mesh_flatplate_turb_545x385.su2"
    dns_data_path: Optional[Path] = None
    label: str = ""

    # ------------------------------------------------------------------ #
    #                        Validation                                    #
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.mach <= 0:
            raise ValueError(f"Mach must be positive, got {self.mach}")
        if self.t_inf <= 0:
            raise ValueError(f"T_inf must be positive, got {self.t_inf}")
        if not 0.0 < self.tw_ratio <= 1.5:
            raise ValueError(
                f"Tw/Taw = {self.tw_ratio} outside physical range (0, 1.5]"
            )
        if not self.label: # Label Construction
            self.label = (
                f"M{self.mach:.1f}_Tw{self.tw_ratio:.3f}_PG{self.pg_angle:.0f}"
            )

    # ------------------------------------------------------------------ #
    #              Derived thermodynamic quantities                         #
    # ------------------------------------------------------------------ #
    @property
    def a_inf(self) -> float:
        """Speed of sound in the freestream [m/s]."""
        return math.sqrt(GAMMA * R_AIR * self.t_inf)

    @property
    def u_inf(self) -> float:
        """Freestream velocity [m/s]:  U = M * a."""
        return self.mach * self.a_inf

    @property
    def recovery_factor(self) -> float:
        """Turbulent recovery factor  r = Pr_lam^(1/3)  (~ 0.896 for air)."""
        return PR_LAM ** (1.0 / 3.0)

    @property
    def t_aw(self) -> float:
        """Adiabatic wall (recovery) temperature [K].

        T_aw = T_inf * [1 + r * (gamma-1)/2 * M^2]
        """
        r = self.recovery_factor
        return self.t_inf * (1.0 + r * (GAMMA - 1.0) / 2.0 * self.mach ** 2)

    @property
    def t_wall(self) -> float:
        """Isothermal wall temperature [K]:  Tw = (Tw/Taw) * Taw."""
        return self.tw_ratio * self.t_aw

    @property
    def feature_vector(self) -> list[float]:
        """GP surrogate input:  [Mach, Tw/Taw, theta_pg]."""
        return [self.mach, self.tw_ratio, self.pg_angle]

    # ------------------------------------------------------------------ #
    #                   Predefined reference cases                         #
    # ------------------------------------------------------------------ #
    @classmethod
    def mach14_flat_plate(cls) -> FlowCondition:
        """M = 13.6, Tw/Taw ~ 0.186  (Zhang, Duan & Choudhari, 2018).

        This is the baseline case validated in the AIAA paper.
        Freestream: T_inf = 47.4 K, P_inf = 1122 Pa, Re = 5e6.
        """
        return cls(
            mach=13.6,
            t_inf=47.4,
            p_inf=1122.0,
            tw_ratio=0.186,
            re=5_000_000.0,
            pg_angle=0.0,
            mesh_file="mesh_flatplate_turb_545x385.su2",
            label="M13.6_Flat_Plate",
        )

    # ------------------------------------------------------------------ #
    #                       Display helpers                                #
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        """Multi-line summary for logging / console output."""
        return (
            f"  Case:        {self.label}\n"
            f"  Mach:        {self.mach}\n"
            f"  T_inf:       {self.t_inf:.1f} K\n"
            f"  P_inf:       {self.p_inf:.1f} Pa\n"
            f"  Tw/Taw:      {self.tw_ratio:.4f}\n"
            f"  T_aw:        {self.t_aw:.1f} K  (computed)\n"
            f"  T_wall:      {self.t_wall:.1f} K  (computed)\n"
            f"  U_inf:       {self.u_inf:.1f} m/s (computed)\n"
            f"  Re:          {self.re:.2e}\n"
            f"  theta_pg:    {self.pg_angle:.1f} deg\n"
            f"  Mesh:        {self.mesh_file}\n"
            f"  Feature X:   {self.feature_vector}"
        )

    def __repr__(self) -> str:
        return (
            f"FlowCondition({self.label}: "
            f"M={self.mach}, Tw/Taw={self.tw_ratio}, "
            f"theta_pg={self.pg_angle} deg)"
        )
