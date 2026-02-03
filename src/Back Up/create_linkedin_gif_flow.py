import matplotlib
matplotlib.use('Agg') # Mandatory for WSL
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from PIL import Image

# --- Global Design Settings ---
plt.style.use('dark_background')
OUTPUT_GIF = "optimization_final_flow.gif"
RESULTS_DIR = Path("../results")
LOG_FILE = RESULTS_DIR / "optimization_log.csv"
FRAME_DURATION_MS = 150 # Slower for better readability

def load_data(folder_path):
    """ Load data with SU2 format support """
    dat_file = folder_path / "flow.dat"
    if not dat_file.exists(): return None, None, None

    try:
        with open(dat_file, 'r') as f:
            lines = f.readlines()
        
        header_rows = 0
        col_names = []
        for i, line in enumerate(lines):
            if "VARIABLES" in line: col_names = re.findall(r'"(.*?)"', line)
            if "ZONE" in line: header_rows = i + 1; break
        
        df = pd.read_csv(dat_file, skiprows=header_rows, sep='\s+', names=col_names, on_bad_lines='skip')
        
        rename_map = {}
        for col in df.columns:
            c = col.lower()
            if 'x' == c or "coordinatex" in c: rename_map[col] = 'x'
            if 'y' == c or "coordinatey" in c: rename_map[col] = 'y'
            if "temperature" in c: rename_map[col] = 'T'
        df.rename(columns=rename_map, inplace=True)
        
        if 'x' in df.columns and 'y' in df.columns and 'T' in df.columns:
            return df, 'x', 'T'
    except: return None, None, None
    return None, None, None

def create_frames_from_log():
    # 1. Read the Log
    if not LOG_FILE.exists():
        print("Error: Log file not found.")
        return [], None
        
    df_log = pd.read_csv(LOG_FILE)
    
    # Find the optimal run
    best_idx = df_log['RMSE'].idxmin()
    best_pr_global = df_log.loc[best_idx, 'Pr_t']
    
    image_paths = []
    best_frame_path = None

    print(f"Rendering {len(df_log)} frames based on Log...")

    for index, row in df_log.iterrows():
        iteration = int(row['Iteration'])
        pr_val = row['Pr_t']
        rmse_val = row['RMSE']
        
        # Construct folder name
        folder_name = f"Pr_{pr_val:.4f}" # Folder name must match previous code output
        folder_path = RESULTS_DIR / folder_name
        
        if not folder_path.exists():
            print(f"Skipping Iter {iteration} (Folder missing)")
            continue

        df_flow, x_col, t_col = load_data(folder_path)
        if df_flow is None: continue

        # Prepare data for plotting
        df_wall = df_flow[df_flow['y'] < 0.0001].sort_values(by=x_col)

        # --- PLOTTING DESIGN (Original design restored) ---
        fig, ax = plt.subplots(figsize=(12, 7), dpi=200)

        # Color logic: Gold for optimal, Green for others
        is_best = (index == best_idx)
        
        main_color = '#ffd700' if is_best else '#00ff9d' # Gold vs Neon Green
        status_txt = "OPTIMAL SOLUTION" if is_best else "OPTIMIZING..."
        status_color = '#ffd700' if is_best else '#00ff9d'
        
        # 1. Main Line (RANS)
        ax.plot(df_wall[x_col], df_wall[t_col], color=main_color, linewidth=3, label=f'RANS Prediction')

        # 2. Reference Line (Wall BC)
        ax.axhline(y=300, color='white', linestyle='--', linewidth=1.5, alpha=0.6, label='Wall BC ($T_{wall}=300K$)')

        # 3. Limits and Grid
        ax.set_xlim(0.0001, 1.0) 
        ax.set_ylim(280, 650) 
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.3)
        
        # 4. Titles and Texts
        title_str = f"Automated Calibration Loop | Iteration {iteration:02d}"
        if is_best: title_str += " [CONVERGED]"
        
        ax.set_title(title_str, fontsize=18, color='white', pad=20)
        ax.set_xlabel("Position along Plate [m]", fontsize=14)
        ax.set_ylabel("Temperature [K]", fontsize=14)

        # 5. Legend (Exact location from previous design)
        ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.1), fontsize=12, frameon=True, facecolor='#111111', edgecolor='#333333')

        # 6. Styled Text Box (The Visual "Trojan Horse")
        # Added RMSE to show real-time improvement
        text_str = f"MACH: 14.0\nPR_T: {pr_val:.4f}\nRMSE: {rmse_val:.4f}\nSTATUS: {status_txt}"
        
        props = dict(boxstyle='round,pad=0.5', facecolor='#222222', alpha=0.9, edgecolor=status_color, linewidth=1.5)
        ax.text(0.70, 0.95, text_str, transform=ax.transAxes, fontsize=13, 
                verticalalignment='top', bbox=props, family='monospace', color=status_color)

        # Save
        filename = f"frame_{iteration:03d}.png"
        plt.savefig(filename, bbox_inches='tight')
        image_paths.append(filename)
        plt.close()

        if is_best:
            best_frame_path = filename
            print(f" -> Frame {iteration} (BEST) Created.")
        else:
            print(f" -> Frame {iteration} Created.")

    return image_paths, best_frame_path

def make_gif_pillow(image_paths, best_frame_path):
    if not image_paths: return
    
    print("Stitching GIF...")
    frames = [Image.open(f) for f in image_paths]
    
    # Dramatic freeze at the end on the best frame
    final_frame = Image.open(best_frame_path) if best_frame_path else frames[-1]
    
    # Duplicate last frame 20 times to keep it on screen
    for _ in range(20):
        frames.append(final_frame.copy())

    frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=frames[1:], 
        duration=FRAME_DURATION_MS, 
        loop=0 
    )
            
    # Clean temporary files
    for f in image_paths: 
        if os.path.exists(f): os.remove(f)
    print(f"Done! Saved as {OUTPUT_GIF}")

if __name__ == "__main__":
    paths, best = create_frames_from_log()
    make_gif_pillow(paths, best)