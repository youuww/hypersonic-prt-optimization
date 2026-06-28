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
                         tw_ratio=0.3, pg_angle=0.0)
    # Re is derived from (mach, t_inf, p_inf): case.re
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

# ---- Sutherland's law for air (must match the SU2 config) ----
MU_REF_SUTH: float = 1.716e-5   # Reference dynamic viscosity [Pa*s]
T_REF_SUTH: float = 273.15      # Reference temperature [K]
S_SUTH: float = 110.4           # Sutherland constant [K]

# Characteristic length used by SU2's REYNOLDS_LENGTH (matches the config).
REYNOLDS_LENGTH: float = 1.0    # [m]


def sutherland_viscosity(t: float) -> float:
    """Dynamic viscosity of air via Sutherland's law [Pa*s].

    mu(T) = mu_ref * (T / T_ref)^1.5 * (T_ref + S) / (T + S)
    """
    return (
        MU_REF_SUTH
        * (t / T_REF_SUTH) ** 1.5
        * (T_REF_SUTH + S_SUTH)
        / (t + S_SUTH)
    )


@dataclass
class FlowCondition:
    """Complete thermodynamic + geometric definition of a hypersonic case.

    Attributes
    ----------
    mach        : Freestream Mach number.
    t_inf       : Freestream static temperature [K].
    p_inf       : Freestream static pressure [Pa].
    tw_ratio    : Wall-to-adiabatic-wall temperature ratio (Tw / Taw).
    pg_angle    : Pressure-gradient angle [deg].
                  0 for flat plates; ramp deflection angle for compression
                  corners.  Named generically so it extends to future
                  geometries (cylinder-flare, etc.).
    mesh_file   : SU2 mesh filename (resolved relative to run directory).
    dns_data_path : Optional path to a two-column DNS CSV (u/U_inf, T/T_inf)
                    used for loss computation.
    label       : Human-readable tag for logs and plots.

    Note
    ----
    The Reynolds number is NOT stored.  It is a derived property
    (``re``) computed from ``(mach, t_inf, p_inf)`` together with
    ``rho_inf`` (ideal gas) and ``mu_inf`` (Sutherland), so it is
    always physically consistent with the freestream state.
    """

    mach: float
    t_inf: float
    p_inf: float
    tw_ratio: float
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
    def rho_inf(self) -> float:
        """Freestream density [kg/m^3] from the ideal gas law: rho = P / (R*T)."""
        return self.p_inf / (R_AIR * self.t_inf)

    @property
    def mu_inf(self) -> float:
        """Freestream dynamic viscosity [Pa*s] via Sutherland's law."""
        return sutherland_viscosity(self.t_inf)

    @property
    def re(self) -> float:
        """Reynolds number (derived), per REYNOLDS_LENGTH.

        Re = rho_inf * U_inf * REYNOLDS_LENGTH / mu_inf

        Derived from (mach, t_inf, p_inf) so it is always consistent with
        the DNS freestream state, instead of a hardcoded constant.
        """
        return self.rho_inf * self.u_inf * REYNOLDS_LENGTH / self.mu_inf

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
        """Deprecated alias for :meth:`dns_M14Tw018` (single M14 source of truth).

        Kept for backward compatibility with older scripts.  The unified M14
        baseline uses the DNS-consistent freestream (T_inf = 47.1 K,
        P_inf = 229.8 Pa) and a derived Reynolds number.
        """
        return cls.dns_M14Tw018()

    # ------------------------------------------------------------------ #
    #   DNS database cases (Zhang, Duan & Choudhari, AIAA J. 2018)         #
    # ------------------------------------------------------------------ #
    #  Freestream values (Mach, T_inf, Tw/Tr) are taken from Table 1 of
    #  the paper.  P_inf = rho_inf * R * T_inf (air).
    #
    #  METHODOLOGY NOTE — derived-Re convention:
    #  The Reynolds number is NOT hardcoded.  It is derived per case from
    #  (mach, t_inf, p_inf) via the `re` property
    #  (Re = rho_inf * U_inf * REYNOLDS_LENGTH / mu_inf), so each case's
    #  freestream is internally consistent and matches its DNS state.
    #  SU2 reconstructs the freestream density from REYNOLDS_NUMBER, so
    #  injecting this derived Re makes SU2 reproduce the DNS freestream.
    #
    #  Why the normalized comparison is still robust across Re:
    #  the calibration compares the *normalized* T-u profile, which is
    #  self-similar via the Crocco-Busemann relation / Strong Reynolds
    #  Analogy (SRA) — T/T_inf is a function of u/U_inf that is only
    #  weakly Re-dependent.  (This is a stronger statement than Morkovin's
    #  hypothesis, which concerns compressibility scaling of turbulence.)

    @staticmethod
    def _dns_profile_path(case_tag: str) -> Path:
        """Resolve the extracted DNS profile CSV for a given case tag."""
        base = Path(__file__).resolve().parent.parent / "data" / "dns_database"
        return base / f"{case_tag}_profile.csv"

    @classmethod
    def dns_M2p5(cls) -> FlowCondition:
        """M = 2.5, Tw/Tr = 1.0 (near-adiabatic).  Air."""
        return cls(
            mach=2.5, t_inf=270.0, p_inf=7749.0, tw_ratio=1.0,
            pg_angle=0.0,
            dns_data_path=cls._dns_profile_path("M2p5"),
            label="M2.5_Tw1.00",
        )

    @classmethod
    def dns_M6Tw025(cls) -> FlowCondition:
        """M = 5.86, Tw/Tr = 0.25 (strongly cooled).  Air."""
        return cls(
            mach=5.86, t_inf=55.0, p_inf=694.5, tw_ratio=0.25,
            pg_angle=0.0,
            dns_data_path=cls._dns_profile_path("M6Tw025"),
            label="M5.86_Tw0.25",
        )

    @classmethod
    def dns_M6Tw076(cls) -> FlowCondition:
        """M = 5.86, Tw/Tr = 0.76 (mildly cooled).  Air."""
        return cls(
            mach=5.86, t_inf=55.0, p_inf=678.7, tw_ratio=0.76,
            pg_angle=0.0,
            dns_data_path=cls._dns_profile_path("M6Tw076"),
            label="M5.86_Tw0.76",
        )

    @classmethod
    def dns_M8Tw048(cls) -> FlowCondition:
        """M = 7.86, Tw/Tr = 0.48.

        PHYSICS CAVEAT: the DNS working fluid for this case is NITROGEN,
        not air.  Our SU2 config uses air (R=287, Sutherland for air).
        The normalized T-u profile is only weakly fluid-dependent, but
        treat this point with extra scrutiny during validation.
        """
        return cls(
            mach=7.86, t_inf=51.8, p_inf=386.5, tw_ratio=0.48,
            pg_angle=0.0,
            dns_data_path=cls._dns_profile_path("M8Tw048"),
            label="M7.86_Tw0.48",
        )

    @classmethod
    def dns_M14Tw018(cls) -> FlowCondition:
        """M = 13.68, Tw/Tr = 0.18.  Air.  (DNS-consistent freestream.)"""
        return cls(
            mach=13.68, t_inf=47.1, p_inf=229.8, tw_ratio=0.18,
            pg_angle=0.0,
            dns_data_path=cls._dns_profile_path("M14Tw018"),
            label="M13.68_Tw0.18",
        )

    @classmethod
    def all_dns_cases(cls) -> list[FlowCondition]:
        """Return all five DNS flat-plate cases (Mach 2.5 -> 14)."""
        return [
            cls.dns_M2p5(),
            cls.dns_M6Tw025(),
            cls.dns_M6Tw076(),
            cls.dns_M8Tw048(),
            cls.dns_M14Tw018(),
        ]

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
