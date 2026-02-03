import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import re

# --- CONFIGURATION ---
plt.style.use('dark_background')
RESULTS_DIR = Path("../results")
DATA_DIR = Path("../data")
DNS_FILE = DATA_DIR / "DNS Dataset.csv"
OUTPUT_GIF = "optimization_profile_physics.gif"
LOG_FILE = RESULTS_DIR / "optimization_log.csv"
X_STATION = 1.5
X_TOL = 0.01  # Increased slightly for stability
U_INF = 1882.0
T_INF = 47.4
FRAME_DURATION = 150 # ms

def load_dns():
    if not DNS_FILE.exists(): return None, None
    df = pd.read_csv(DNS_FILE)
    return df.iloc[:, 0].values, df.iloc[:, 1].values

def load_simulation_profile(folder_path):
    dat_file = folder_path / "flow.dat"
    if not dat_file.exists(): return None
    
    try:
        # Optimization: Read only header first to avoid memory overhead
        with open(dat_file, 'r') as f:
            lines = f.readlines(2000) # Read first 2000 lines for header detection
            
        header_rows = 0
        col_names = []
        for i, line in enumerate(lines):
            if "VARIABLES" in line: col_names = re.findall(r'"(.*?)"', line)
            if "ZONE" in line: header_rows = i + 1; break
        
        # Robust loading for large Mach 14 files
        df = pd.read_csv(dat_file, 
                         skiprows=header_rows, 
                         sep='\s+', 
                         names=col_names, 
                         on_bad_lines='skip', 
                         low_memory=False, 
                         engine='c')
        
        rename_map = {}
        for col in df.columns:
            c = col.lower()
            if 'x' == c or "coordinatex" in c: rename_map[col] = 'x'
            if "temperature" in c: rename_map[col] = 'T'
            if "velocity" in c and "x" in c: rename_map[col] = 'u'
            if "momentum" in c and "x" in c: rename_map[col] = 'mom_x'
            if "density" in c: rename_map[col] = 'rho'
            
        df.rename(columns=rename_map, inplace=True)
        if 'u' not in df.columns and 'mom_x' in df.columns: df['u'] = df['mom_x'] / df['rho']
        if 'u' not in df.columns or 'T' not in df.columns: return None

        slice_df = df[ (df['x'] > X_STATION - X_TOL) & (df['x'] < X_STATION + X_TOL) ].copy()
        if slice_df.empty: return None

        slice_df['u_norm'] = slice_df['u'] / U_INF
        slice_df['t_norm'] = slice_df['T'] / T_INF
        return slice_df.sort_values(by='u_norm')

    except Exception as e: 
        print(f"Error loading {dat_file}: {e}")
        return None

def create_frames():
    if not LOG_FILE.exists(): return [], None
    df_log = pd.read_csv(LOG_FILE)

    # === FIX 1: Find best by VALUE, not index (more robust) ===
    min_rmse_val = df_log['RMSE'].min()
    
    dns_u, dns_t = load_dns()
    
    # Load Baseline
    baseline_df = None
    first_run_folder = RESULTS_DIR / f"Pr_{df_log.iloc[0]['Pr_t']:.4f}"
    if first_run_folder.exists():
        baseline_df = load_simulation_profile(first_run_folder)

    image_paths = []
    best_frame_path = None

    print(f"Generating profiles based on Log Order ({len(df_log)} runs)...")

    for index, row in df_log.iterrows():
        iteration = int(row['Iteration'])
        pr_val = row['Pr_t']
        current_rmse = row['RMSE']
        
        folder_path = RESULTS_DIR / f"Pr_{pr_val:.4f}"
        
        df = load_simulation_profile(folder_path)
        if df is None: 
            print(f"Skipping Iter {iteration} (Load failed)")
            continue
        
        # === FIX 2: Compare floats with tolerance ===
        is_best = np.isclose(current_rmse, min_rmse_val, atol=1e-6)

        # --- PLOTTING ---
        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
        
        if dns_u is not None:
            ax.plot(dns_u, dns_t, 'o', color='white', markersize=4, alpha=0.6, label='DNS (Ground Truth)')

        if baseline_df is not None:
            ax.plot(baseline_df['u_norm'], baseline_df['t_norm'], 
                    color='#ff0055', linestyle='--', linewidth=1.5, alpha=0.5, label='Initial Guess')

        # Colors by status
        line_color = '#ffd700' if is_best else '#00ff9d'
        label_str = f'OPTIMAL RANS' if is_best else f'Iter {iteration}'
        z_order = 10 if is_best else 5 # Make sure best line is on top
        
        ax.plot(df['u_norm'], df['t_norm'], color=line_color, linewidth=3, label=label_str, zorder=z_order)

        ax.set_title(f"SciML Calibration: Matching Physics vs. DNS\nIteration {iteration} | Pr_t = {pr_val:.4f}", 
                     fontsize=16, color='white', pad=15)
        ax.set_xlabel(r"Normalized Velocity ($u/u_{\infty}$)", fontsize=14)
        ax.set_ylabel(r"Normalized Temperature ($T/T_{\infty}$)", fontsize=14)
        
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 14)
        
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.legend(loc='upper right', fontsize=12, facecolor='#111111', edgecolor='#333333')

        # HUD
        status_txt = "OPTIMAL SOLUTION" if is_best else "LEARNING"
        status_color = '#ffd700' if is_best else '#00ff9d'
        
        hud_txt = f"RMSE: {current_rmse:.4f}\nStatus: {status_txt}"
        ax.text(0.05, 0.95, hud_txt, transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', bbox=dict(facecolor='#222222', edgecolor=status_color, alpha=0.8), 
                family='monospace', color=status_color)

        fname = f"prof_iter_{iteration:03d}.png"
        plt.savefig(fname, bbox_inches='tight')
        image_paths.append(fname)
        plt.close()
        
        if is_best:
            best_frame_path = fname
            print(f" -> Iteration {iteration} (Pr={pr_val:.4f}) [BEST HIT - GOLD FRAME]")
        else:
            print(f" -> Iteration {iteration}")

    return image_paths, best_frame_path

def make_gif(paths, best_frame_path):
    if not paths: return
    
    frames = [Image.open(p) for p in paths]
    
    # Freeze logic: Make sure we explicitly load the BEST frame for the freeze
    if best_frame_path and Path(best_frame_path).exists():
        last = Image.open(best_frame_path)
        print(f"Freezing on BEST frame: {best_frame_path}")
    else:
        last = frames[-1]
        print("Warning: Best frame not found, freezing last frame.")
    
    # Add 20 frames of the winner at the end
    for _ in range(30): frames.append(last.copy())
    
    frames[0].save(OUTPUT_GIF, save_all=True, append_images=frames[1:], duration=FRAME_DURATION, loop=0)
    
    # Cleanup
    for p in paths: Path(p).unlink(missing_ok=True)
    print(f"Done! {OUTPUT_GIF}")

if __name__ == "__main__":
    paths, best_path = create_frames()
    make_gif(paths, best_path)