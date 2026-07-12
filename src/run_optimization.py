import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from su2_interface import SU2Interface 
from case_config import FlowCondition
from pathlib import Path
import argparse
import time
import shutil
from datetime import datetime

# ==========================================
#              CONFIGURATION
# ==========================================
# Pr_t search bounds. Widened from the old (0.5, 0.95) to (0.3, 1.2) so the
# optimizer is not clipped for low-Mach cases whose optimum sits near ~0.9.
BOUNDS = (0.3, 1.2)
TOLERANCE = 1e-3
MAX_ITER = 6
LOG_FILE = "optimization_log.csv" # Log File - Csv

# Maps a --case name to its FlowCondition factory. Default is the unified
# DNS-consistent Mach 14 baseline (M14Tw018).
CASE_FACTORIES = {
    "M2p5":     FlowCondition.dns_M2p5,
    "M6Tw025":  FlowCondition.dns_M6Tw025,
    "M6Tw076":  FlowCondition.dns_M6Tw076,
    "M8Tw048":  FlowCondition.dns_M8Tw048,
    "M14Tw018": FlowCondition.dns_M14Tw018,
}

# ==========================================
#              GLOBAL OBJECTS
# ==========================================
# The runner is assigned in the __main__ block once CLI args are known.
# objective_function reads this module-level global at call time.
runner = None

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
    # ---------------- CLI ----------------
    parser = argparse.ArgumentParser(
        description="Calibrate optimal Pr_t for one flow case via Brent's method."
    )
    parser.add_argument(
        "--case", choices=sorted(CASE_FACTORIES.keys()), default="M14Tw018",
        help="DNS case to calibrate (default: M14Tw018, the unified Mach 14 baseline).",
    )
    parser.add_argument(
        "--iter", type=int, default=15000,
        help="SU2 inner iterations per run (use ~3000 for a quick smoke test).",
    )
    parser.add_argument(
        "--cores", type=int, default=4,
        help="MPI cores for SU2 (lower this to reduce machine load).",
    )
    args = parser.parse_args()

    # ---------------- Build the runner for the chosen case ----------------
    flow = CASE_FACTORIES[args.case]()
    runner = SU2Interface(flow=flow, num_cores=args.cores)
    print(f"=== 🚀 Calibrating case: {args.case} ===")
    print(flow.summary())

    runner.ITERATIONS = args.iter
    print(f"[Config] SU2 iterations={args.iter} | cores={args.cores} | "
          f"Pr_t bounds={BOUNDS}\n")

    # Use global runner so objective_function writes into the same run_dir
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

    # --- Rename run folder to: case_niter_date (e.g. M14Tw018_5iter_260207) ---
    n_iter = len(history)
    geometry = args.case  # case-specific (e.g. M14Tw018, M6Tw025) - NOT the shared config filename
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