import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from su2_interface import SU2Interface 
from pathlib import Path
import time
import shutil
from datetime import datetime

# ==========================================
#              CONFIGURATION
# ==========================================
BOUNDS = (0.5, 0.95) # Bounding Pr_t
TOLERANCE = 1e-3
MAX_ITER = 5
LOG_FILE = "optimization_log.csv" # Log File - Csv

# ==========================================
#              GLOBAL OBJECTS
# ==========================================
runner = SU2Interface(num_cores=4)

iteration = 0
history = []

# ==========================================
#              OPTIMIZATION ENGINE
# ==========================================

def objective_function(pr_t):
    global iteration
    iteration += 1
    
    # Formatting
    current_pr = float(pr_t)
    run_id = f"Iter_{iteration}_Pr{current_pr:.4f}"
    
    print(f"\n>>> [Optimizer] Iteration {iteration}: Testing Pr_t = {current_pr:.4f}")
    start_time = time.time() # Time
    
    # --- 1. Run Pipeline ---
    try:
        cfg_file = runner.generate_config(current_pr, run_id)
        
        # Delete leftovers
        flow_file = runner.SCRIPT_DIR / "flow.dat"
        flow_file.unlink(missing_ok=True)
        
        success = runner.run_su2(cfg_file)
        
        if not success:
            print("!!! CFD Simulation Crashed. Applying Penalty.")
            loss = 100.0
        else:
            loss = runner.calculate_loss("flow")

    except Exception as e:
        print(f"!!! Critical Error in execution: {e}")
        loss = 100.0

    elapsed = time.time() - start_time
    print(f"   [Result] RMSE: {loss:.5f} | Time: {elapsed:.2f}s | Prandtl: {current_pr}")
    
    # --- 2. Save Data ---
    history.append({
        'Iteration': iteration,
        'Pr_t': current_pr,
        'RMSE': loss,
        'Time_Sec': elapsed
    })

    pd.DataFrame(history).to_csv(LOG_FILE, index=False)
    
    # --- 3. Visualize & Save per Iteration ---
    if loss < 50.0:
        pr_str = f"{current_pr:.4f}"
        runner.plot_results("flow", pr_str)
        runner.organize_files(pr_str)
    
    # Clean up
    runner.cleanup(run_id)
    
    return loss

# ==========================================
#              MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== 🚀 Starting SciML Optimization Loop ===")
    
    runner = SU2Interface(num_cores=4)
    # Per-run folder: all Pr_* and summary files go here; renamed at end to geometry_niter_date
    ts = datetime.now().strftime("%y%m%d_%H%M")
    run_dir = runner.RESULTS_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    runner.RESULTS_DIR = run_dir
    print(f"[Results] Run folder: {run_dir} (will be renamed at end)\n")
    
    # method='bounded': Brent's Method
    res = minimize_scalar(
        objective_function, 
        bounds=BOUNDS, 
        method='bounded',
        options={'xatol': TOLERANCE, 'maxiter': MAX_ITER, 'disp': 3}
    )

    # --- OPTIMIZATION COMPLETE ---
    print("\n" + "="*40)
    print(f" OPTIMIZATION COMPLETE")
    print(f" Best Pr_t Found: {res.x:.5f}")
    print(f" Minimum RMSE:    {res.fun:.5f}")
    print("="*40)

    # 1. Iterations Log History
    pd.DataFrame(history).to_csv(LOG_FILE, index=False)
    print(f"[Log] History saved to {LOG_FILE}")
    
    # 2. Convergence Plot
    hist_df = pd.DataFrame(history)
    
    # Vaild runs
    valid_runs = hist_df[hist_df['RMSE'] < 20]
    crashed_runs = hist_df[hist_df['RMSE'] >= 20]
   
    # Plot:
    plt.figure(figsize=(10,6))

    # Real Convergence - Vaild runs
    if not valid_runs.empty:
        plt.plot(valid_runs['Iteration'], valid_runs['RMSE'], 'b-o', label='Optimization Path')
        
        # Min from Valid runs
        best_run_val = valid_runs['RMSE'].min()
        best_run_idx = valid_runs['RMSE'].idxmin()
        best_iter = valid_runs.loc[best_run_idx, 'Iteration']
        
        plt.plot(best_iter, best_run_val, 'g*', markersize=20, markeredgecolor='k', label=f'Best (RMSE={best_run_val:.4f})', zorder=10)

        # Ylim [0.9 - 1.1]
        y_min = valid_runs['RMSE'].min()
        y_max = valid_runs['RMSE'].max()
        plt.ylim(y_min * 0.9, y_max * 1.1)

    # Marking Crashed runs
    if not crashed_runs.empty:
        # Location at the Ceiling of the Plot
        y_ceiling = plt.ylim()[1]
        plt.scatter(crashed_runs['Iteration'], [y_ceiling * 0.95] * len(crashed_runs), 
                   c='red', marker='x', s=50, label='Crash Penalty')
    
    # --- Plotting the LAST iteration ---
    plt.plot(hist_df.iloc[-1]['Iteration'], hist_df.iloc[-1]['RMSE'], 'r*', markersize=15,markeredgecolor='k', label='Last Iteration', zorder=10)

    plt.xlabel('Iteration')
    plt.ylabel('RMSE (Temperature Error)')
    plt.title('Convergence of Hypersonic Turbulence Calibration')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save & Print
    plt.savefig("optimization_convergence.png", dpi=150)
    print("[Log] Convergence plot saved.")

    # 3. Final Verification Run - OPTIMAL Pr_t
    print("\n>>> Running Validation Case with OPTIMAL Parameters...")
    optimal_pr = res.x
    optimal_pr_str = f"{optimal_pr:.4f}"
    
    final_cfg = runner.generate_config(optimal_pr, optimal_pr_str)
    runner.run_su2(final_cfg)
    runner.plot_results("flow", optimal_pr_str)
    
    # Organize files & Clean
    runner.organize_files(optimal_pr_str) # Make Dir
    runner.cleanup(optimal_pr_str)
    
    # --- Move Summary Files to [Results \ Run Folder] ---
    print(f"\n>>> Archiving summary files to run folder...")
    for f_name in [LOG_FILE, "optimization_convergence.png"]:
        src = Path(f_name)
        dst = runner.RESULTS_DIR / f_name
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"       -> Moved: {f_name}")

    # --- Rename run folder to: geometry_niter_date (e.g. flatplate_M14_5iter_260207) ---
    n_iter = len(history)
    geometry = runner.base_config.stem  # e.g. turb_SA_flatplate_M14Tw018
    date_short = datetime.now().strftime("%y%m%d")
    final_name = f"{geometry}_{n_iter}iter_{date_short}"
    run_dir = runner.RESULTS_DIR
    results_root = run_dir.parent
    final_path = results_root / final_name
    if final_path.exists():
        # avoid overwrite: append time
        final_name = f"{geometry}_{n_iter}iter_{date_short}_{datetime.now().strftime('%H%M')}"
        final_path = results_root / final_name
    run_dir.rename(final_path)
    print(f"\n>>> Results saved under: {final_path}")

    print("\n=== 🏁 Mission Accomplished. ===")