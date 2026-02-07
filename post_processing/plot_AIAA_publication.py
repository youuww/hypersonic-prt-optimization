import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import re

# ==========================================
#              CONFIGURATION
# ==========================================
# Basic Paths
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR.parent / "results"

def _get_run_dir(root: Path) -> Path:
    """Use flat results/ if legacy; else latest run folder (geometry_niter_date)."""
    if (root / "optimization_log.csv").exists():
        return root
    run_dirs = [d for d in root.iterdir() if d.is_dir() and "iter" in d.name]
    if not run_dirs:
        return root
    return max(run_dirs, key=lambda d: d.stat().st_mtime)

RESULTS_DIR = _get_run_dir(RESULTS_ROOT)
LOG_FILE = RESULTS_DIR / "optimization_log.csv"

# Log DataFrame
df_log = pd.read_csv(LOG_FILE)

# Optimal Pr
best_row_idx = df_log["RMSE"].idxmin()
optimal_pr = df_log.loc[best_row_idx, "Pr_t"]

# 4 points after the dot
optimal_pr_str = f"{optimal_pr:.4f}"

print(f">>> Auto-detected Optimal Pr_t: {optimal_pr_str} (RMSE: {df_log.loc[best_row_idx, 'RMSE']:.5f})")

DATA_FILE = RESULTS_DIR / f"Pr_{optimal_pr_str}" / "flow.dat"  # Optimized SU2 Result
DNS_FILE = SCRIPT_DIR.parent / "data" / "DNS Dataset.csv"      # Benchmark Data

# Safety Check - Data
if not DATA_FILE.exists():
    raise FileNotFoundError(f"CRITICAL: Could not find result file at {DATA_FILE}")

# Physics Constants (Mach 14 Case)
U_INF = 1882.0
T_INF = 47.4
PR_T = 0.566

# Output Configuration
OUTPUT_PNG = "aiaa_plot_M14.png"
OUTPUT_PDF = "aiaa_plot_M14.pdf"

# ==========================================
#           PLOTTING STYLE (AIAA)
# ==========================================
# Configure Matplotlib to use DejaVu Serif fonts (Times New Roman) and LaTeX-style math
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "figure.figsize": (3.5, 2.5),   # Single-column figure size (3.5 inch)
    "figure.dpi": 300,
    "mathtext.fontset": "stix",  # LaTeX-like math rendering
    "savefig.bbox": "tight" # Cutting white margins
})

# ==========================================
#             DATA HANDLING
# ==========================================
def load_robust_data(dat_file):
    """
    Parses SU2 Tecplot (.dat) files robustly, handling variable headers 
    and ensuring numeric conversion.
    """
    print(f"[IO] Loading file: {dat_file}")
    with open(dat_file, 'r') as f:
        lines = f.readlines()
    
    # Dynamic header detection
    header_rows = 0
    col_names = []
    for i, line in enumerate(lines):
        if "VARIABLES" in line: 
            col_names = re.findall(r'"(.*?)"', line)
        if "ZONE" in line: 
            header_rows = i + 1
            break
            
    # Load data into DataFrame
    df = pd.read_csv(dat_file, skiprows=header_rows, sep='\s+', names=col_names, on_bad_lines='skip')
    
    # Force numeric conversion to handle potential formatting errors
    for col in df.columns: 
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop invalid rows
    df.dropna(inplace=True)

    # Standardize column names
    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if 'x' == c or "coordinatex" in c: rename_map[col] = 'x'
        if "temperature" in c: rename_map[col] = 'T'
        if "momentum" in c and "x" in c: rename_map[col] = 'mom_x'
        if "density" in c: rename_map[col] = 'rho'
    
    df.rename(columns=rename_map, inplace=True)
    
    # Calculate Velocity if missing (u = momentum / density)
    if 'u' not in df.columns and 'mom_x' in df.columns:
        df['u'] = df['mom_x'] / df['rho']
        
    return df

# ==========================================
#             PLOTTING LOGIC
# ==========================================
def plot_aiaa_style():
    # 1. Load Datasets
    su2_df = load_robust_data(DATA_FILE)
    dns_df = pd.read_csv(DNS_FILE)
    
    # 2. Extract Profile at Validation Station (x = 1.5m)
    # Using a small tolerance window to capture the slice
    slice_df = su2_df[ (su2_df['x'] > 1.495) & (su2_df['x'] < 1.505) ].copy()
    
    # 3. Normalize Variables
    # Velocity normalized by Freestream Velocity (u_inf)
    # Temperature normalized by Freestream Temperature (T_inf)
    slice_df['u_norm'] = slice_df['u'] / U_INF
    slice_df['t_norm'] = slice_df['T'] / T_INF
    
    # Sort by velocity for clean plotting lines
    slice_df = slice_df.sort_values(by='u_norm')
    
    # 4. Generate Plot
    fig, ax = plt.subplots()
    
    # --- Plot DNS Data ---
    # Style: Black circles with white fill (Standard for experimental/DNS data)
    ax.plot(dns_df.iloc[:,0], dns_df.iloc[:,1], 'ok', 
            label='DNS', 
            markersize=5, markerfacecolor='white', markeredgewidth=1.2, zorder=5)
    
    # --- Plot Calibrated RANS ---
    # Style: Solid blue line
    ax.plot(slice_df['u_norm'], slice_df['t_norm'], '-b', 
            label=f'Calibrated RANS ($Pr_t = {PR_T:.3f}$)', 
            zorder=10) 
    
    # 5. Formatting (LaTeX Labels)
    ax.set_xlabel(r'$u / u_{\infty}$')
    ax.set_ylabel(r'$T / T_{\infty}$')
    
    # Set Axis Limits for cleaner view
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 11.5]) 
    
    # Add subtle grid
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    
    # Legend formatting
    ax.legend(loc='lower center', frameon=True, fancybox=False, edgecolor='black')

    # 6. Save High-Resolution Output
    plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches='tight')
    plt.savefig(OUTPUT_PDF, bbox_inches='tight') # Vector format for LaTeX papers
    
    print(f"[Success] Plots generated:\n  - {OUTPUT_PNG}\n  - {OUTPUT_PDF}")

if __name__ == "__main__":
    plot_aiaa_style()