"""Extract (u/U_inf, T/T_inf) profiles from the NASA TMR DNS database.

The Zhang, Duan & Choudhari (2018) DNS files (`*_Stat.dat`) are Tecplot
POINT-format files with ~28 columns of wall-normal turbulence statistics.
Our calibration pipeline (su2_interface.calculate_loss) only needs the
boundary-layer T-u coupling:

    column  5  ->  <u>/U_inf      (velocity ratio,    0 -> 1)
    column  7  ->  <T>/T_inf      (temperature ratio)
    column 27  ->  Pr_t           (DNS turbulent Prandtl, for reference)

This script parses each raw file and writes a clean 2-column CSV
(`{case}_profile.csv`) plus reports the DNS bulk-averaged Pr_t — a
useful physical sanity check for the calibration target.

Usage:
    cd src/
    python extract_dns_profiles.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Column indices (0-based) within the DNS data block
COL_U_NORM = 4    # <u>/U_inf
COL_T_NORM = 6    # <T>/T_inf
COL_PRT = 26      # Pr_t

DNS_DIR = Path(__file__).resolve().parent.parent / "data" / "dns_database"

# All five cases of the DNS family
CASES = ["M2p5", "M6Tw025", "M6Tw076", "M8Tw048", "M14Tw018"]


def parse_dns_block(dat_file: Path) -> np.ndarray:
    """Parse the numeric data block of a Tecplot POINT-format DNS file.

    Strategy: skip everything up to and including the 'DATAPACKING=POINT'
    line, then read whitespace-separated floats until the rows stop
    parsing as numbers.

    Returns
    -------
    np.ndarray, shape (n_points, n_columns)
    """
    with open(dat_file, "r") as f:
        lines = f.readlines()

    # Find where the numeric data starts
    start_idx = None
    for i, line in enumerate(lines):
        if "DATAPACKING" in line.upper():
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(f"No DATAPACKING marker found in {dat_file.name}")

    rows: list[list[float]] = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        # Skip blanks and comment lines (e.g. the '#variables=...' line
        # that sits between DATAPACKING and the first data row).
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            # A non-numeric line AFTER data has started means a new ZONE
            # block — stop.  Before any data, keep scanning the header.
            if rows:
                break
            continue

    if not rows:
        raise ValueError(f"No numeric data parsed from {dat_file.name}")
    return np.array(rows)


def extract_case(case: str) -> dict:
    """Extract one case: write profile CSV, return summary stats."""
    dat_file = DNS_DIR / f"{case}_Stat.dat"
    data = parse_dns_block(dat_file)

    u_norm = data[:, COL_U_NORM]
    t_norm = data[:, COL_T_NORM]
    prt = data[:, COL_PRT]

    # Sort by u_norm (required for np.interp in the loss function) and
    # drop duplicates that would break interpolation.
    order = np.argsort(u_norm)
    u_sorted = u_norm[order]
    t_sorted = t_norm[order]
    _, unique_idx = np.unique(u_sorted, return_index=True)
    u_clean = u_sorted[unique_idx]
    t_clean = t_sorted[unique_idx]

    out_csv = DNS_DIR / f"{case}_profile.csv"
    pd.DataFrame({"u_norm": u_clean, "T_norm": t_clean}).to_csv(
        out_csv, index=False
    )

    # DNS Pr_t reference: averaged across the boundary layer (0.1 < u < 0.9
    # to avoid wall/edge singularities) — a physical anchor for the GP target.
    bl_mask = (u_norm > 0.1) & (u_norm < 0.9)
    prt_bl = prt[bl_mask]
    prt_mean = float(np.mean(prt_bl)) if prt_bl.size else float("nan")

    return {
        "case": case,
        "n_points": data.shape[0],
        "u_range": (float(u_norm.min()), float(u_norm.max())),
        "T_peak": float(t_norm.max()),
        "Prt_DNS_mean": prt_mean,
        "csv": out_csv.name,
    }


def main() -> None:
    print("=" * 64)
    print("  Extracting DNS T-u profiles  (Zhang, Duan & Choudhari 2018)")
    print("=" * 64)

    summaries = []
    for case in CASES:
        try:
            s = extract_case(case)
            summaries.append(s)
            print(
                f"  [OK] {s['case']:10s} | {s['n_points']:4d} pts | "
                f"T_peak={s['T_peak']:6.2f} | "
                f"Pr_t(DNS,BL-avg)={s['Prt_DNS_mean']:.3f} | "
                f"-> {s['csv']}"
            )
        except Exception as e:
            print(f"  [FAIL] {case}: {e}")

    print("-" * 64)
    print(
        "  NOTE: Pr_t(DNS,BL-avg) is the boundary-layer-averaged turbulent\n"
        "  Prandtl from DNS.  It is a physical reference, NOT necessarily\n"
        "  the optimal constant Pr_t for RANS (which the calibration finds).")
    print("=" * 64)


if __name__ == "__main__":
    main()
